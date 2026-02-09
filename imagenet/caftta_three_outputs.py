from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
from sklearn.mixture import GaussianMixture
import PIL
import torchvision.transforms as transforms
import my_transforms as my_transforms
from time import time
import logging
import torch.nn.functional as F
import math
from sklearn import metrics
from sam import SAM
import norm 
def get_tta_transforms():
    transform = transforms.Compose([transforms.RandomCrop(224, padding=4), transforms.RandomHorizontalFlip()])
    return transform

def collect_params(model):
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names

def configure_model(model):
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        # else: m.requires_grad_(True)
    return model

class CAFTTA(nn.Module):
    def __init__(self, model, optimizer, steps=1, episodic=False, teacher_ema=0.999, n_aug=1, hyperparameters=[0.3, 0.5, 0.5]):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.model_state, self.optimizer_state, self.model_ema, _ = copy_model_and_optimizer(self.model, self.optimizer)
        self._inlier_sum = 0
        self._outlier_sum = 0
        self.mt = teacher_ema
        self.model0 = norm.Norm(deepcopy(model), reset_stats=True, no_stats=True) 
        # self.model0.eval()
        self.hyperparameters = hyperparameters
        self.transform = get_tta_transforms()
        self.n_aug = n_aug  
        
    def forward(self, x):
        if self.episodic: self.reset()
        for _ in range(self.steps): outputs = self.forward_and_adapt(x, self.model, self.model0, self.optimizer)
        return outputs

    @torch.enable_grad()  # NO augmentation!
    def forward_and_adapt(self, x, model, model0, optimizer):
        # outputs = self.model(self.transform(x))
        outputs = self.model(self.transform(x))
        N = self.n_aug
        outputs_emas = []
        with torch.no_grad():
            for i in range(N):
                outputs_ = self.model_ema(self.transform(x)).detach() # if ema1_ent < ema2_ent else self.model_ema2(self.transform(x)).detach()
                outputs_emas.append(outputs_)

        outputs_ema = torch.stack(outputs_emas)
        outputs_ema = outputs_ema.mean(0)
        ema_ent = softmax_entropy(outputs_ema)

        scores_reverse = entropy(outputs_ema.softmax(1))
        scores_reverse = entropy(outputs.softmax(1))
        # coeff = get_coeff(softmax_entropy(outputs_ema), self.hyperparameters[0], 1000)
        score_reverse = entropy(outputs.softmax(1))
        filter_idx = torch.where(scores_reverse < self.hyperparameters[0] * math.log(1000), 0, 1) 
        filter_idx2 = torch.where(scores_reverse > self.hyperparameters[1] * math.log(1000), 0, 1) 

        with torch.no_grad():
            caftta_outputs = self.model0(x)
            thr = self.hyperparameters[0] * math.log(1000)
            transformed_x_1 = self.transform(x)
            transformed_x_2 = self.transform(x)
            output_aug_main = self.model(transformed_x_1)
            output_aug_ema = self.model_ema(transformed_x_2)
            filter_main_aug = torch.where(entropy(output_aug_main.softmax(1)) < thr, 0, 1)
            filter_ema_aug = torch.where(entropy(output_aug_ema.softmax(1)) < thr, 0, 1)
            # filter_main = torch.where(softmax_entropy(outputs) < thr, 0, 1)
 
            filter_00 = (filter_main_aug == 0) & (filter_ema_aug == 0)
            filter_01 = (filter_main_aug == 0) & (filter_ema_aug == 1)  # 흠..
            
            # filter_01_use = filter_01 & (outputs.argmax(dim=1) == output_aug_main.argmax(dim=1)) # & (torch.where(softmax_entropy(outputs) < thr, 0, 1) == 0)
            filter_00_or_01_use = filter_00 | filter_01
            
            filter_10 = (filter_main_aug == 1) & (filter_ema_aug == 0)  # 버려
            # filter_10_use = filter_10 & (ema_output_original.argmax(dim=1) == output_aug_ema.argmax(dim=1)) & (torch.where(softmax_entropy(ema_output_original) < thr, 0, 1) == 0)
        
            filter_11 = (filter_main_aug == 1) & (filter_ema_aug == 1)
            filter_11 = filter_11
        
        loss_ind = softmax_entropy(outputs)[filter_00_or_01_use == True] 
        loss_ind_weighted = loss_ind * get_coeff(softmax_entropy(outputs_ema), self.hyperparameters[0], 1000)[filter_00_or_01_use == True]
        
        loss_ood = softmax_entropy(outputs)[filter_11 == True]
        loss_ood_weighted = loss_ood #* get_reversed_coeff(softmax_entropy(outputs_ema), self.hyperparameters[0], 1000)[filter_11 == True] # !!!

        loss = loss_ind_weighted.mean(0) - self.hyperparameters[2] * loss_ood_weighted.mean(0) # + loss_001.mean(0)

        self._inlier_sum += 200 - filter_idx[100:].sum()  # 100 - filter_idx[:100].sum()
        self._outlier_sum += 200 - filter_idx2[:100].sum()  # 100 - filter_idx[100:].sum()
        # print(32 - filter_idx[:32].sum(), 32 - filter_idx[32:].sum(), 32 - filter_idx2[:32].sum(), 32 - filter_idx2[32:].sum())
        

        # onehot_ema = F.one_hot(output_aug_ema.argmax(dim=1), num_classes=1000).float()
        # loss += softmax_entropy_ema(outputs, onehot_ema)[filter_10_use == True].mean(0) / len(loss_ind_weighted)

        # 🔥 그냥 ema로 filtering하는 loss
        # loss = (softmax_entropy(outputs)[filter_idx == 0]).mean(0)  - self.hyperparameters[2] * softmax_entropy(outputs)[filter_idx2 == 0].mean(0)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        final_outputs = outputs + outputs_ema + caftta_outputs

        self.model_ema = update_ema_variables(ema_model=self.model_ema, model=self.model, alpha_teacher=0.999)

        return (outputs, outputs_ema, caftta_outputs, outputs.logsumexp(1))

    def reset(self):
        print('resetting model and optimizer')
        if self.model_state is None or self.optimizer_state is None: raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)                 
        self.model_state, self.optimizer_state, self.model_ema, self.model0 = copy_model_and_optimizer(self.model, self.optimizer)

def get_coeff(ent_list, hyperparameters, n_class):
    return 1 / (torch.exp(ent_list.clone().detach() - hyperparameters * math.log(n_class)))

def get_reversed_coeff(ent_list, hyperparameters, n_class):
    return 1 / (torch.exp(hyperparameters * math.log(n_class) - ent_list.clone().detach()))

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    model0 = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return model_state, optimizer_state, ema_model, model0


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

def entropy(x): return -(x * torch.log(x)).sum(1)  # x = x.softmax(1)

def softmax_entropy(x):
    return -(x.softmax(1) * (x).log_softmax(1)).sum(1)

def softmax_entropy_ema(x, x_ema): 
    return -(x_ema * x.log_softmax(1)).sum(1)

@torch.jit.script
def softmax_mean_entropy(x: torch.Tensor) -> torch.Tensor:
    """Mean entropy of softmax distribution from logits."""
    x = x.softmax(1).mean(0)
    return -(x * torch.log(x)).sum()
