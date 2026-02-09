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
        outputs = self.model(self.transform(x))
        N = self.n_aug
        outputs_emas = []
        """x_aug = self.transform(x)
        ema1_ent_list = softmax_entropy(self.model_ema(x_aug))
        ema2_ent_list = softmax_entropy(self.model_ema2(x_aug))

        filter_ema1 = torch.where(ema1_ent_list < self.hyperparameters[0] * math.log(1000), 0, 1)  # 기존 caftta
        filter_ema2 = torch.where(ema2_ent_list < self.hyperparameters[0] * math.log(1000), 0, 1)  # 기존 caftta

        ema1_ent = ema1_ent_list[filter_ema2 == 0].mean().item()
        ema2_ent = ema2_ent_list[filter_ema2 == 0].mean().item()
        print("ema_ent", ema1_ent, ema2_ent, 'ratio', ema1_ent/ema2_ent)
    
        # EMA 값보다 현재 loss_ind 값이 작을 때만 model_ema 업데이트
        # print(logit_similarity(outputs, outputs_ema).max(), logit_similarity(outputs, outputs_ema).min())
        self.model_ema = update_ema_variables(ema_model=self.model_ema, model=self.model, alpha_teacher=0.99)
        self.model_ema2 = update_ema_variables(ema_model=self.model_ema2, model=self.model, alpha_teacher=0.999)
        if ema1_ent > ema2_ent * 1.1:
            print('reset ema to original')
            # del self.model_ema
            self.model_ema = deepcopy(self.model_ema2)"""

        with torch.no_grad():
            for i in range(N):
                outputs_ = self.model_ema(self.transform(x)).detach() # if ema1_ent < ema2_ent else self.model_ema2(self.transform(x)).detach()
                outputs_emas.append(outputs_)

        self.model_ema = update_ema_variables(ema_model=self.model_ema, model=self.model, alpha_teacher=0.999)
        outputs_ema = torch.stack(outputs_emas)
        outputs_ema = outputs_ema.mean(0)
        ema_ent = softmax_entropy(outputs_ema)
        
        # CAFTTA
        with torch.no_grad():
            ema_ent = softmax_entropy(outputs_ema)
            caftta_outputs = self.model0(x)
            caftta_ent = softmax_entropy(caftta_outputs)
            tta_ent = softmax_entropy(outputs)
            entropys = torch.stack([ema_ent, caftta_ent], dim=1)
            reversed_final_outputs = (outputs_ema).softmax(1)
            entropys = torch.stack([ema_ent, caftta_ent, tta_ent], dim=1)
            sharpened_values = (entropys / 1.).softmax(1)
            
            final_outputs = (1 - sharpened_values[:, 0].unsqueeze(1)) * outputs_ema.softmax(1) \
                            + (1 - sharpened_values[:, 1].unsqueeze(1)) * caftta_outputs.softmax(1) + \
                            (1 - sharpened_values[:, 2].unsqueeze(1))* outputs.softmax(1)

            self.model0[1].conv1.weight.copy_(model[1].conv1.weight)
            self.model0[1].bn1.weight.copy_(model[1].bn1.weight)
            self.model0[1].bn1.bias.copy_(model[1].bn1.bias)
            for a_layer, b_layer in zip(model[1].layer1, model0[1].layer1):
                if isinstance(a_layer, nn.Conv2d) or isinstance(a_layer, nn.BatchNorm2d):
                    b_layer.weight.copy_(a_layer.weight)
                    b_layer.bias.copy_(a_layer.bias)

            
        scores_reverse = entropy(reversed_final_outputs)
        coeff = 1 / (torch.exp(scores_reverse.clone().detach() - self.hyperparameters[0] * math.log(1000)))
        coeff2 = 1 / (torch.exp(self.hyperparameters[0] * math.log(outputs.shape[-1]) - scores_reverse.clone().detach()))
        filter_idx = torch.where(scores_reverse < self.hyperparameters[0] * math.log(1000), 0, 1) 
        filter_idx2 = torch.where(scores_reverse > self.hyperparameters[1] * math.log(1000), 0, 1) 

        with torch.no_grad():
            transformed_x_1 = self.transform(x)
            transformed_x_2 = self.transform(x)
            output_aug_main = self.model(transformed_x_1)
            output_aug_ema = self.model_ema(transformed_x_2)
            filter_main_aug = torch.where(entropy(output_aug_main.softmax(1)) < self.hyperparameters[0] * math.log(1000), 0, 1)
            filter_ema_aug = torch.where(entropy(output_aug_ema.softmax(1)) < self.hyperparameters[0] * math.log(1000), 0, 1)
            filter_main = torch.where(softmax_entropy(outputs) < self.hyperparameters[0] * math.log(1000), 0, 1)

        filter_00 = (filter_main_aug == 0) & (filter_ema_aug == 0)
        filter_01 = (filter_main_aug == 0) & (filter_ema_aug == 1)  # 흠..
        filter_01_use = filter_01 & outputs.argmax(dim=1) == output_aug_main.argmax(dim=1)

        filter_10 = (filter_main_aug == 1) & (filter_ema_aug == 0)  # 버려
        filter_11 = (filter_main_aug == 1) & (filter_ema_aug == 1)

        loss_ind = (softmax_entropy(outputs))[filter_main_aug == 0] 
        loss_ind_weighted = loss_ind * coeff[filter_main_aug == 0]
        # loss_001 = (softmax_entropy(outputs))[filter_01 == True] * (coeff[filter_01 == True] - self.hyperparameters[2])
        # print(loss_001)
        loss_ood = softmax_entropy(outputs)[filter_ema_aug == 1]
        # loss_ood = ((softmax_entropy(outputs))[filter_idx2 == 0] * coeff2[filter_idx2 == 0]).mean(0)
        self._inlier_sum += 64 - filter_idx[32:].sum()  # 100 - filter_idx[:100].sum()
        self._outlier_sum += 64 - filter_idx2[:32].sum()  # 100 - filter_idx[100:].sum()
        # print(32 - filter_idx[:32].sum(), 32 - filter_idx[32:].sum(), 32 - filter_idx2[:32].sum(), 32 - filter_idx2[32:].sum())
        
        loss_ind_weighted = loss_ind_weighted
    
        loss = loss_ind_weighted.mean(0) - self.hyperparameters[2] * loss_ood.mean(0) # + loss_001.mean(0)
        
        if isinstance(optimizer, SAM):
            loss.backward()
            optimizer.first_step(zero_grad=True)
            outputs = self.model(x)
            loss_ind = (softmax_entropy(outputs))[filter_idx == 0] * coeff[filter_idx == 0]
            loss_ind = loss_ind.mean(0)
            loss_ood = softmax_entropy(outputs)[filter_idx2 == 0].mean(0)   
            loss = loss_ind - self.hyperparameters[2] * loss_ood
            loss.backward()
            optimizer.second_step(zero_grad=True)
        else:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        final_outputs = torch.where(
            (torch.argmax(outputs_ema, dim=1) == torch.argmax(outputs, dim=1)).unsqueeze(1), 
            outputs, 
            outputs + outputs_ema + caftta_outputs
        )

        return ((outputs, outputs_ema, caftta_outputs, outputs_ema + outputs, outputs_ema + caftta_outputs, outputs + caftta_outputs, final_outputs), outputs.logsumexp(1))

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
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)

@torch.jit.script
def softmax_mean_entropy(x: torch.Tensor) -> torch.Tensor:
    """Mean entropy of softmax distribution from logits."""
    x = x.softmax(1).mean(0)
    return -(x * torch.log(x)).sum()
