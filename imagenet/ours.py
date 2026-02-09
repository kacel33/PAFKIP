from copy import deepcopy
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import math

# BF16 안전화를 위한 상수
DTYPE = torch.bfloat16
EPS = torch.tensor(1e-6, dtype=DTYPE)
EXP_CLAMP_MAX = torch.tensor(10.0, dtype=DTYPE)

def get_tta_transforms():
    transform = transforms.Compose([transforms.RandomCrop(224, padding=4),
                                    transforms.RandomHorizontalFlip()])
    return transform

def KIP(outputs, outputs_ema, caftta_outputs, alpha=0.1):
    """Knowledge-Informed Prediction: confidence 기반 가중 앙상블"""
    conf_outputs = outputs.softmax(1).max(1).values
    conf_outputs_ema = outputs_ema.softmax(1).max(1).values
    conf_caftta = caftta_outputs.softmax(1).max(1).values

    three = torch.tensor(3.0, dtype=DTYPE, device=outputs.device)
    one_third = torch.tensor(1.0/3.0, dtype=DTYPE, device=outputs.device)
    alpha_t = torch.tensor(alpha, dtype=DTYPE, device=outputs.device)

    conf_mean = (conf_outputs + conf_outputs_ema + conf_caftta) / three

    delta_outputs = conf_outputs - conf_mean
    delta_ema = conf_outputs_ema - conf_mean
    delta_caftta = conf_caftta - conf_mean

    w_outputs = one_third + alpha_t * delta_outputs
    w_ema = one_third + alpha_t * delta_ema
    w_caftta = one_third + alpha_t * delta_caftta
    weights = torch.stack([w_outputs, w_ema, w_caftta], dim=1)

    logit_final = (
        weights[:, [0]] * outputs +
        weights[:, [1]] * outputs_ema +
        weights[:, [2]] * caftta_outputs
    )

    return logit_final

class PAFKIP(nn.Module):
    """PAF-KIP: Prediction Aggregation Filtering + Knowledge-Informed Prediction for ImageNet"""
    def __init__(self, model, optimizer, steps=1, episodic=False, n_aug=1,
                teacher_ema=0.999, ent_thr_ratio=0.4, alpha=2.0):
        super().__init__()
        # 모델을 BF16으로 변환
        self.model = model.to(dtype=DTYPE, memory_format=torch.channels_last)
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic

        self.model_state, self.optimizer_state, self.model_ema, self.model0 = copy_model_and_optimizer(self.model, self.optimizer)

        self.ema_decay = teacher_ema
        self.model0 = deepcopy(model).to(dtype=DTYPE, memory_format=torch.channels_last)
        self.model0.eval()

        self.ent_thr_ratio = ent_thr_ratio
        self.alpha = alpha
        self.n_class = 1000  # ImageNet

        self.transform = get_tta_transforms()
        self.n_aug = n_aug

        self._t = 0

    def forward(self, x):
        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.optimizer)
        if self.episodic:
            self.reset()
        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)
        self.model_state, self.optimizer_state, self.model_ema, self.model0 = copy_model_and_optimizer(self.model, self.optimizer)

    @torch.enable_grad()
    def forward_and_adapt(self, x, optimizer):
        self._t += 1
        # 입력을 BF16으로 변환
        x = x.to(DTYPE)
        outputs = self.model(self.transform(x))
        n_class = outputs.shape[-1]
        N = self.n_aug
        outputs_emas = []

        with torch.no_grad():
            for _ in range(N):
                outputs_ = self.model_ema(self.transform(x)).detach()
                outputs_emas.append(outputs_)

        outputs_ema = torch.stack(outputs_emas)
        outputs_ema = outputs_ema.mean(0)

        with torch.no_grad():
            caftta_outputs = self.model0(x)

        # PAF: Prediction Aggregation Filtering
        with torch.no_grad():
            thr = torch.tensor(self.ent_thr_ratio * math.log(n_class), dtype=DTYPE, device=x.device)

            output_aug_main = self.model(self.transform(x))
            output_aug_ema = self.model_ema(self.transform(x))

            zero = torch.tensor(0, dtype=torch.int64, device=x.device)
            one = torch.tensor(1, dtype=torch.int64, device=x.device)

            filter_main_aug = torch.where(entropy(output_aug_main.softmax(1)) < thr, zero, one)
            filter_ema_aug = torch.where(entropy(output_aug_ema.softmax(1)) < thr, zero, one)

            filter_ent_min = filter_main_aug == zero
            filter_ent_max = (filter_main_aug == one) & (filter_ema_aug == one)

        # Loss 계산
        loss_ind = (softmax_entropy(outputs))[filter_ent_min]
        loss_ind_weighted = loss_ind * get_coeff(softmax_entropy(output_aug_ema), self.ent_thr_ratio, n_class)[filter_ent_min]
        loss_ood = softmax_entropy(outputs)[filter_ent_max]

        loss = torch.tensor(0.0, dtype=DTYPE, device=x.device)
        if len(loss_ind_weighted) > 0:
            loss = loss + loss_ind_weighted.mean(0)

        if len(loss_ood) > 0:
            alpha_t = torch.tensor(self.alpha, dtype=DTYPE, device=x.device)
            loss = loss - alpha_t * loss_ood.mean(0)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # KIP: Knowledge-Informed Prediction
        final_outputs = KIP(outputs, output_aug_ema, caftta_outputs)
        self.model_ema = update_ema_variables(ema_model=self.model_ema, model=self.model, alpha_teacher=self.ema_decay)

        return (final_outputs, logsumexp_safe(outputs, dim=1))

def get_coeff(ent_list, ent_thr_ratio, n_class):
    """엔트로피 기반 적응적 가중치 계산"""
    thr = torch.tensor(ent_thr_ratio * math.log(n_class), dtype=DTYPE, device=ent_list.device)
    clamp_max = EXP_CLAMP_MAX.to(ent_list.device)
    diff = (ent_list.clone().detach() - thr).clamp(-clamp_max, clamp_max)
    return torch.exp(-diff)

def configure_model(model):
    """BatchNorm만 학습 가능하도록 모델 설정"""
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
            m.to(DTYPE)
    return model

def update_ema_variables(ema_model, model, alpha_teacher):
    """EMA 모델 파라미터 업데이트"""
    alpha_t = torch.tensor(alpha_teacher, dtype=DTYPE)
    one = torch.tensor(1.0, dtype=DTYPE)
    one_minus_alpha = one - alpha_t
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_t * ema_param[:].data[:] + one_minus_alpha * param[:].data[:]
    return ema_model

def collect_params(model):
    """BatchNorm 파라미터만 수집"""
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names

def copy_model_and_optimizer(model, optimizer):
    """모델과 옵티마이저 상태 복사"""
    model_state = deepcopy(model.state_dict())
    model0 = deepcopy(model).to(dtype=DTYPE, memory_format=torch.channels_last)
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model).to(dtype=DTYPE, memory_format=torch.channels_last)
    for param in ema_model.parameters():
        param.detach_()
    return model_state, optimizer_state, ema_model, model0

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """모델과 옵티마이저 상태 복원"""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

def entropy(x):
    """Shannon Entropy (BF16 안전화)"""
    eps = EPS.to(x.device)
    x_clamped = x.clamp(min=eps)
    return -(x_clamped * torch.log(x_clamped)).sum(1)

def softmax_entropy(x):
    """Softmax Entropy (BF16 안전화)"""
    eps = EPS.to(x.device)
    p = x.softmax(1).clamp(min=eps)
    return -(p * torch.log(p)).sum(1)

def logsumexp_safe(x, dim=1):
    """LogSumExp (BF16 안전화)"""
    eps = EPS.to(x.device)
    x_max = x.max(dim=dim, keepdim=True).values
    return x_max.squeeze(dim) + torch.log(torch.exp(x - x_max).sum(dim).clamp(min=eps))
