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
@torch.no_grad()
def entropy_from_logits(logits):
    p = F.softmax(logits, dim=-1)
    return -(p * p.log()).sum(dim=-1)    
"""def _get_tta_transforms(gaussian_std: float=0.005, soft=False, clip_inputs=False):
    # same as STAMP
    img_shape = (32, 32, 3) 
    n_pixels = img_shape[0]

    tta_transforms = transforms.Compose([
        my_transforms.Clip(0.0, 1.0), 
        my_transforms.ColorJitterPro(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        ),
        transforms.Pad(padding=int(n_pixels / 2), padding_mode='edge'),  
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1/16, 1/16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            resample=PIL.Image.BILINEAR,
            fillcolor=None
        ),
        transforms.CenterCrop(size=n_pixels),
        transforms.RandomHorizontalFlip(p=0.5),
        my_transforms.Clip(0.0, 1.0)
    ])
    return tta_transforms"""
def confidence_weighted_ensemble(outputs, outputs_ema, caftta_outputs, alpha=0.1):
    """
    각 모델의 confidence에 따라 가중 평균 앙상블을 수행하는 함수.

    Args:
        outputs (Tensor): main model의 logits, shape (batch, class)
        outputs_ema (Tensor): EMA model의 logits, shape (batch, class)
        caftta_outputs (Tensor): CAFTTA model의 logits, shape (batch, class)
        alpha (float): 중심화된 confidence에 곱해지는 계수

    Returns:
        Tensor: 가중 평균된 softmax 확률, shape (batch, class)
    """
    # 각 모델 confidence 계산
    conf_outputs = outputs.softmax(1).max(1).values  # (batch,)
    conf_outputs_ema = outputs_ema.softmax(1).max(1).values
    conf_caftta = caftta_outputs.softmax(1).max(1).values
    probs = outputs.softmax(1)
    probs_ema = outputs_ema.softmax(1)
    probs_caftta = caftta_outputs.softmax(1)


    # adaptive alpha: 예를 들어 최대값이 alpha_scale
    # alpha = disagreement * alpha  # (B,) → confidence 편차에 곱해질 α
    
    # 평균 confidence
    conf_mean = (conf_outputs + conf_outputs_ema + conf_caftta) / 3
    # print(alpha.mean())
    # 중심화된 confidence (편차)
    delta_outputs = conf_outputs - conf_mean
    delta_ema = conf_outputs_ema - conf_mean
    delta_caftta = conf_caftta - conf_mean

    # 가중치 계산: 1/3 ± α * 편차
    w_outputs = 1/3 + alpha * delta_outputs
    w_ema = 1/3 + alpha * delta_ema
    w_caftta = 1/3 + alpha * delta_caftta
    weights = torch.stack([w_outputs, w_ema, w_caftta], dim=1)  # (batch, 3)

    # 확률 앙상블
    logit_final = (
        weights[:, [0]] * outputs +
        weights[:, [1]] * outputs_ema +
        weights[:, [2]] * caftta_outputs
    )

    return logit_final
def softmax_based_confidence_weighted_ensemble(outputs, outputs_ema, caftta_outputs, alpha=0.1):
    """
    3개의 Logit에 대해 Confidence-Weighted Ensemble을 수행합니다.
    
    Args:
        outputs (torch.Tensor): Model 1 Logits (B, C)
        outputs_ema (torch.Tensor): Model 2 Logits (B, C)
        caftta_outputs (torch.Tensor): Model 3 Logits (B, C)
        
    Returns:
        ensemble_probs (torch.Tensor): Ensemble된 확률 분포 (B, C)
    """
    
    # 1. Logit을 Probability로 변환 (Softmax)
    p1 = F.softmax(outputs, dim=1)
    p2 = F.softmax(outputs_ema, dim=1)
    p3 = F.softmax(caftta_outputs, dim=1)
    
    # 2. 각 모델의 Confidence (Max Probability) 추출 -> Shape: (B,)
    # values, indices 중 values만 필요
    conf1, _ = p1.max(dim=1)
    conf2, _ = p2.max(dim=1)
    conf3, _ = p3.max(dim=1)
    
    # 3. 가중치 Stack 및 정규화 (Normalization)
    # 각 샘플별로 가중치의 합이 1이 되도록 조정해야 올바른 확률 분포가 나옵니다.
    weights = torch.stack([conf1, conf2, conf3], dim=1)  # (B, 3)
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8) # Zero division 방지
    
    # 4. 가중 평균 계산 (Weighted Average)
    # (B, 3, 1) * (B, 3, C) 형태의 연산을 위해 차원 확장
    weights = weights.unsqueeze(2) 
    
    stacked_probs = torch.stack([p1, p2, p3], dim=1) # (B, 3, C)
    
    # 가중치 적용 후 모델 차원(dim=1)에 대해 합산
    ensemble_probs = (weights * stacked_probs).sum(dim=1) # (B, C)
    
    return ensemble_probs
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
        N = self.n_aug
        outputs_emas = []
        with torch.no_grad():
            for i in range(N):
                outputs_ = self.model_ema(self.transform(x)).detach() # if ema1_ent < ema2_ent else self.model_ema2(self.transform(x)).detach()
                outputs_emas.append(outputs_)

        outputs_ema = torch.stack(outputs_emas)
        outputs_ema = outputs_ema.mean(0)
        ema_ent = softmax_entropy(outputs_ema)
        
        # scores_reverse = entropy(outputs_ema.softmax(1))
        # scores_reverse = entropy(outputs.softmax(1))
        # score_reverse = entropy(outputs.softmax(1))
        # filter_idx = torch.where(scores_reverse < self.hyperparameters[0] * math.log(1000), 0, 1) 
        # filter_idx2 = torch.where(scores_reverse > self.hyperparameters[1] * math.log(1000), 0, 1) 

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
            # filter_11 = filter_11 
        
        loss_ind = softmax_entropy(outputs)[filter_00_or_01_use == True] 
        loss_ind_weighted = loss_ind * get_coeff(softmax_entropy(outputs_ema), self.hyperparameters[0], 1000)[filter_00_or_01_use == True]
        
        loss_ood = softmax_entropy(outputs)[filter_11 == True]
        loss_ood_weighted = loss_ood # * get_reversed_coeff(softmax_entropy(outputs_ema), self.hyperparameters[0], 1000)[filter_11 == True] # !!!

        loss = loss_ind_weighted.mean(0) - self.hyperparameters[2] * loss_ood_weighted.mean(0) # + loss_001.mean(0)
    
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        final_outputs = confidence_weighted_ensemble(outputs, outputs_ema, caftta_outputs, alpha=0.10)
        # final_outputs = outputs + outputs_ema + caftta_outputs
        self.model_ema = update_ema_variables(ema_model=self.model_ema, model=self.model, alpha_teacher=0.999)

        return (final_outputs, outputs.logsumexp(1))

    def reset(self):
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
