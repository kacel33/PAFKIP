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
        self.model_state, self.optimizer_state, self.model_ema, self.model0 = copy_model_and_optimizer(self.model, self.optimizer)
        self._inlier_sum = 0
        self._outlier_sum = 0
        self.mt = teacher_ema
        self.model_ema2 = deepcopy(self.model_ema)
        self.model0 = deepcopy(model)
        self.model0.eval()
        self.hyperparameters = hyperparameters
        self.transform = get_tta_transforms()
        self.loss_ind_ema = None 
        self.n_aug = n_aug  
        self.original_model = deepcopy(self.model_ema)

    def forward(self, x):
        if self.episodic: self.reset()
        for _ in range(self.steps): outputs = self.forward_and_adapt(x, self.model, self.model0, self.optimizer)
        return outputs

    @torch.enable_grad()  # NO augmentation!
    def forward_and_adapt(self, x, model, model0, optimizer):
        # outputs = self.model(self.transform(x))
        outputs = self.model(self.transform(x))
        outputs_ema = self.model_ema(self.transform(x))
        ent_list = softmax_entropy(outputs_ema)
        
        filter_idx = torch.where(ent_list < self.hyperparameters[0] * 1000, 0, 1)  # 기존 caftta

        loss = (softmax_entropy(outputs)[filter_idx == 0]).mean(0) - self.hyperparameters[2] * softmax_entropy(outputs)[filter_idx == 1].mean(0)
      
        self._inlier_sum += 100 - filter_idx[:100].sum() # close-set을 잘 학습
        self._outlier_sum += filter_idx[100:].sum()  # open-set을 잘 학습
        # wandb.log({"inlier-true": 100 - filter_idx[:100].sum(), "inlier-false": filter_idx[:100].sum(), 
        #             "outlier-true": filter_idx[100:].sum(), "outlier-false": 100 - filter_idx[100:].sum()})
    
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        self.model_ema = update_ema_variables(ema_model=self.model_ema, model=self.model, alpha_teacher=0.999)
        return ((outputs, outputs, outputs, outputs, outputs,outputs, outputs), outputs.logsumexp(1))

    def reset(self):
        if self.model_state is None or self.optimizer_state is None: raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)                 
        self.model_state, self.optimizer_state, self.model_ema, self.model0 = copy_model_and_optimizer(self.model, self.optimizer)

def get_coeff(ent_list, hyperparameters, n_class):
    return 1 / (torch.exp(ent_list.clone().detach() - hyperparameters * math.log(n_class)))

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
