import random
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit
from sklearn.mixture import GaussianMixture
from sam import SAM

class HLoss(nn.Module):
    def __init__(self, temp_factor=1.0):
        super(HLoss, self).__init__()
        self.temp_factor = temp_factor

    def forward(self, x):
        softmax = F.softmax(x / self.temp_factor, dim=1)
        entropy = - softmax * torch.log(softmax + 1e-6)
        b = entropy.mean()

        return b

loss_fn = HLoss()

class HUS:
    def __init__(self, capacity, num_class, threshold=None):
        self.num_class = num_class
        self.data = [[[], [], []] for _ in range(self.num_class)]  # feat, pseudo_cls, domain, conf
        self.counter = [0] * self.num_class
        self.capacity = capacity
        self.threshold = threshold

    def get_memory(self):
        data = []
        for x in self.data: data.extend(x[0])
        data = torch.stack(data)
        return data

    def get_occupancy(self):
        occupancy = 0
        for data_per_cls in self.data: occupancy += len(data_per_cls[0])
        return occupancy

    def get_occupancy_per_class(self):
        occupancy_per_class = [0] * self.num_class
        for i, data_per_cls in enumerate(self.data): occupancy_per_class[i] = len(data_per_cls[0])
        return occupancy_per_class

    def add_instance(self, instance):
        assert (len(instance) == 3)
        cls = instance[1]
        self.counter[cls] += 1
        is_add = True

        if self.threshold is not None and instance[2] < self.threshold: is_add = False
        elif self.get_occupancy() >= self.capacity: is_add = self.remove_instance(cls)

        if is_add:
            for i, dim in enumerate(self.data[cls]):
                dim.append(instance[i])

    def get_largest_indices(self):
        occupancy_per_class = self.get_occupancy_per_class()
        max_value = max(occupancy_per_class)
        largest_indices = []
        for i, oc in enumerate(occupancy_per_class):
            if oc == max_value:
                largest_indices.append(i)
        return largest_indices

    def get_target_index(self, data):
        return random.randrange(0, len(data))

    def remove_instance(self, cls):
        largest_indices = self.get_largest_indices()
        if cls not in largest_indices:  # instance is stored in the place of another instance that belongs to the largest class
            largest = random.choice(largest_indices)  # select only one largest class
            tgt_idx = self.get_target_index(self.data[largest][1])
            for dim in self.data[largest]:
                dim.pop(tgt_idx)
        else:  # replaces a randomly selected stored instance of the same class
            tgt_idx = self.get_target_index(self.data[cls][1])
            for dim in self.data[cls]:
                dim.pop(tgt_idx)
        return True

    def reset_value(self, feats, cls, aux):
        self.data = [[[], [], []] for _ in range(self.num_class)]  # feat, pseudo_cls, domain, conf

        for i in range(len(feats)):
            tgt_idx = cls[i]
            self.data[tgt_idx][0].append(feats[i])
            self.data[tgt_idx][1].append(cls[i])
            self.data[tgt_idx][3].append(aux[i])

class SoTTA(nn.Module):
    def __init__(self, model, optimizer, memory_size=64, threshold=.33, num_class=1000, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer  
        assert (isinstance(self.optimizer, SAM))
        self.memory_size = memory_size
        self.threshold = threshold
        self.num_class = num_class
        self.memory = HUS(memory_size, num_class, threshold)

        self.episodic = episodic
        self.steps = steps
        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)


    def forward(self, x):
        if self.episodic: self.reset()
        for _ in range(self.steps): outputs = forward_and_adapt(x, self.model, self.memory, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        
        load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)
        self.memory = HUS(self.memory_size, self.num_class, self.threshold)

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, memory, optimizer):
    model.eval()
    outputs = model(x)
    outputs_ = outputs.softmax(dim=1)
    outputs_ = outputs_.detach().to('cpu')
    confidences, pseudo_labels = torch.max(outputs_, dim=1)
    
    for i in range(x.shape[0]): 
        memory.add_instance([x[i].detach(), pseudo_labels[i], confidences[i]])

    feats = memory.get_memory()
    try:
        if feats.shape[0] == 0: return outputs
        elif feats.shape[0] == 1: model.eval()
        else: model.train()
    except:
        logger.warning("feats.shape[0] == 0")
        return outputs
    
    model.train()
    optimizer.zero_grad()
    _feats_logits = model(feats)
    loss = loss_fn(_feats_logits)
    loss.backward()
    optimizer.first_step(zero_grad=True)

    model.train()
    _feats_logits = model(feats)
    loss = loss_fn(_feats_logits)
    loss.backward()
    optimizer.second_step(zero_grad=True)

    return outputs

def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")

    return params, names

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None

    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"
