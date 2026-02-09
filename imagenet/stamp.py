import math
import random
from copy import deepcopy
import my_transforms as my_transforms
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import torch.nn.functional as F
from time import time
import PIL
def entropy(p): return -torch.sum(p * torch.log(p + 1e-5), dim=1)

def get_tta_transforms(gaussian_std: float=0.005, soft=False, clip_inputs=False):
    transform = transforms.Compose([transforms.RandomCrop(224, padding=4), transforms.RandomHorizontalFlip()])
    return transform

class RBM:
    def __init__(self, max_len, num_class):
        self.num_class = num_class
        self.count_class = torch.zeros(num_class)
        self.data = [[] for _ in range(num_class)]
        self.max_len = max_len
        self.total_num = 0

    def remove_item(self):
        max_count = 0
        for i in range(self.num_class):
            if len(self.data[i]) == 0: continue
            if self.count_class[i] > max_count: max_count = self.count_class[i]
        max_classes = []
        for i in range(self.num_class):
            if self.count_class[i] == max_count and len(self.data[i]) > 0:
                max_classes.append(i)
        remove_class = random.choice(max_classes)
        self.data[remove_class].pop(0)

    def append(self, items, class_ids):
        for item, class_id in zip(items, class_ids):
            if self.total_num < self.max_len:
                self.data[class_id].append(item)
                self.total_num += 1
            else:
                self.remove_item()
                self.data[class_id].append(item)

    def get_data(self):
        data = []
        for cls in range(self.num_class):
            data.extend(self.data[cls])
            self.count_class[cls] = 0.9 * self.count_class[cls] + 0.1 * len(self.data[cls])

        return torch.stack(data)

    def __len__(self):
        return self.total_num

    def reset(self):
        self.count_class = torch.zeros(self.num_class)
        self.data = [[] for _ in range(self.num_class)]
        self.total_num = 0


class STAMP(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, num_class, steps=1, episodic=False, alpha=[0.5]):
        super().__init__()
        self.model = model
        self.norm_model = deepcopy(self.model).train()
        self.optimizer = optimizer
        self.num_class = num_class
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.n_view = 32
        self.alpha = alpha[0]
        self.margin = self.alpha * math.log(num_class)
        self.mem = RBM(64, num_class)

        self._inlier_sum = 0
        self._outlier_sum = 0
        self.transform = get_tta_transforms()    
        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic: self.reset()
        for _ in range(self.steps): output = forward_and_adapt(x, self)
        return output

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)
        self.mem.reset()
    
    def update_memory(self, x):
        with torch.no_grad():
            if self.num_class == 1000:
                outputs = []
                output_origin = self.model(x)
                outputs.append(output_origin.softmax(dim=1))
                
                for i in range(self.n_view):
                    x_aug = self.transform(x)
                    outputs.append(self.model(x_aug.cuda()).softmax(dim=1))
                    
                output = torch.stack(outputs, dim=0)
                output = torch.mean(output, dim=0)
                entropys = entropy(output)
                filter_ids = torch.where(entropys < self.margin)
                x_append = x[filter_ids]
                self.mem.append(x_append, output_origin.max(dim=1)[1][filter_ids])

            else:
                outputs = []
                self.model.train()
                output_origin = self.model(x)
                output_norm = self.norm_model(x)
                filter_ids_0 = torch.where(output_origin.max(dim=1)[1] == output_norm.max(dim=1)[1])
                outputs.append(output_origin.softmax(dim=1))

                for i in range(self.n_view):
                    augmented_x = self.transform(x)
                    outputs.append(self.model(augmented_x).softmax(dim=1))

                output = torch.stack(outputs, dim=0)
                output = torch.mean(output, dim=0)
                entropys = entropy(output)[filter_ids_0]
                filter_ids = torch.where(entropys < self.margin)
                x_append = x[filter_ids_0][filter_ids]
                self.mem.append(x_append, output_origin.max(dim=1)[1][filter_ids_0][filter_ids])
            
            """temp = torch.arange(0, 200)
            filtered_ = temp[filter_ids_0][filter_ids]

            self._inlier_sum += len(filtered_[filtered_ < 100])
            self._outlier_sum += len(filtered_[filtered_ >= 100])"""
            return output      

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.jit.script
def softmax_mean_entropy(x: torch.Tensor) -> torch.Tensor:
    """Mean entropy of softmax distribution from logits."""
    x = x.softmax(1).mean(0)
    return -(x * torch.log(x)).sum()


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, stamp):
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    output = stamp.update_memory(x)
    if len(stamp.mem) != 0:
        data = stamp.mem.get_data()
        stamp.optimizer.zero_grad()
        # data = x_origin
        if len(data) > 0:
            output_1 = stamp.model(data)
            entropys = softmax_entropy(output_1)
            # coeff = 1 / (torch.exp(entropys.clone().detach() - self.margin))
            inv_entropy = 1 / torch.exp(entropys)
            coeff = inv_entropy / inv_entropy.sum() * 64
            entropys = entropys.mul(coeff)
            loss = entropys.mean()
            loss.backward()
            stamp.optimizer.first_step(zero_grad=True)

            # second time forward
            output_1 = stamp.model(data)
            entropys = softmax_entropy(output_1)
            inv_entropy = 1 / torch.exp(entropys)
            coeff = inv_entropy / inv_entropy.sum() * 64
            entropys = entropys.mul(coeff)
            loss = entropys.mean()
            loss.backward()
            stamp.optimizer.second_step(zero_grad=True)
    
    return output

def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
        if 'layer4' in nm:
            continue
        if 'conv5_x' in nm:
            continue
        if 'blocks.9' in nm:
            continue
        if 'blocks.10' in nm:
            continue
        if 'blocks.11' in nm:
            continue
        if 'norm.' in nm:
            continue
        if nm in ['norm']:
            continue

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
    model.requires_grad_(False)
    # configure norm for eata updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            # m.momentum = 0.2
            m.running_mean = None
            m.running_var = None
        if isinstance(m, nn.LayerNorm):
            m.requires_grad_(True)
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
