import argparse
import math
import os
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from iopath.common.file_io import g_pathmgr
from prettytable import PrettyTable
from scipy import interpolate
from sklearn import metrics
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset, ConcatDataset
from tqdm import tqdm
from robustbench.data import load_imagenetc
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from torchvision.models import resnet50 as resnet50_img, ResNet50_Weights
import cotta
import eata
import ostta
import norm
import tent
import caftta
import stamp, rotta, sotta, caftta_palm, filtering_with_main, filtering_with_ema, caftta_three_outputs
from sam import SAM
from data import load_imagenet_o_c, load_places_c, load_textures_c, load_gaussian, load_uniform
from utils import AverageMeter, get_logger, set_random_seed
import wandb
wandb.init(project='ostta')
wandb.run.name = 'unient'
parser = argparse.ArgumentParser()

# Model options
parser.add_argument("--arch", default="Hendrycks2020AugMix", choices=["Hendrycks2020AugMix", "Tian2022Deeper_DeiT-B", "Standard_R50"])
parser.add_argument("--adaptation", default="tent")
parser.add_argument("--episodic", action="store_true")
parser.add_argument("--rounds", default=1, type=int)

# Corruption options
parser.add_argument("--dataset", default="imagenet")
parser.add_argument("--type", default="gaussian_noise")
parser.add_argument("--severity", default=5, type=int)
parser.add_argument("--num_ex", default=5000, type=int)
parser.add_argument("--open_set_tta", default="True", type=str)
parser.add_argument("--ood_dataset", default="imagenet_o", choices=["imagenet_o", "places365", "textures", 'gaussian', 'uniform'])

# Optimizer options
parser.add_argument("--steps", default=1, type=int)
parser.add_argument("--lr", default=.00025, type=float)
parser.add_argument("--method", default="SGD", choices=["Adam", "SGD", "SAM"])
parser.add_argument("--momentum", default=0.9, type=float)

# Testing options
parser.add_argument("--batch_size", default=32, type=int) 
# Misc options
parser.add_argument("--rng_seed", default=1, type=int)
parser.add_argument("--save_dir", default="./output")
parser.add_argument("--data_dir", default=os.path.expanduser("~/imagenet"))
parser.add_argument("--ckpt_dir", default="./ckpt")
parser.add_argument("--log_dest", default="log.txt")

# Tent options
parser.add_argument("--alpha", nargs="+", default=[0.5], type=float)
parser.add_argument("--criterion", default="ent", choices=["ent", "ent_ind", "ent_ind_ood", "ent_unf"])

# EATA options
parser.add_argument("--fisher_size", default=2000, type=int)
parser.add_argument("--fisher_alpha", default=1., type=float)
parser.add_argument("--e_margin", default=0.40, type=float)
parser.add_argument("--d_margin", default=0.05, type=float)

parser.add_argument("--n_aug", default=32, type=int)
parser.add_argument("--mt", default=.999, type=float)
args = parser.parse_args()

args.type = ["gaussian_noise", "shot_noise", "impulse_noise",
             "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
             "snow", "frost", "fog", "brightness", "contrast",
             "elastic_transform", "pixelate", "jpeg_compression"]
args.severity = [5]
args.log_dest = "{}_{}_lr_{}_alpha_{}_{}.txt".format(
    args.adaptation, args.dataset, args.lr, "_".join(str(alpha) for alpha in args.alpha), args.criterion)
def softmax_entropy(x): return -(x.softmax(1) * (x).log_softmax(1)).sum(1)
g_pathmgr.mkdirs(args.save_dir)

set_random_seed(args.rng_seed)

logger = get_logger(__name__, args.save_dir, args.log_dest)
logger.info(f"args:\n{args}")

ent_list = []
def evaluate(hyperparameters=None, lr=None):
    # configure model
    base_model = load_model(args.arch, args.ckpt_dir, args.dataset, "corruptions").cuda()
    # base_model = resnet50_img(weights=ResNet50_Weights.IMAGENET1K_V1).cuda()
    if args.adaptation == "source":
        base_model.eval()
        model = base_model
    elif args.adaptation == "norm":
        model = norm.Norm(base_model)
    elif args.adaptation == "cotta":
        base_model = cotta.configure_model(base_model)
        params, param_names = cotta.collect_params(base_model)
        optimizer = setup_optimizer(params, lr)
        model = cotta.CoTTA(base_model, optimizer, steps=args.steps, episodic=args.episodic)
    elif args.adaptation == "tent":
        base_model = tent.configure_model(base_model)
        params, param_names = tent.collect_params(base_model)
        optimizer = setup_optimizer(params, .00025)
        if args.criterion == 'ent_unf': alpha = [.1, .2]
        elif args.criterion == 'ent_ind_ood': alpha = [.1, .5]
        else: alpha = [0., .0]
        model = tent.Tent(base_model, optimizer, steps=args.steps, episodic=args.episodic, alpha=alpha, criterion=args.criterion)

    elif args.adaptation == "eata":
        fisher_dataset = datasets.ImageFolder(args.data_dir + "/imagenet/images/train",
                                              transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), 
                                              transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
        sampled_indices = torch.randperm(len(fisher_dataset))[:args.fisher_size]
        sampler = SubsetRandomSampler(sampled_indices)
        fisher_loader = DataLoader(fisher_dataset, batch_size=args.batch_size * 2, sampler=sampler)
        base_model = eata.configure_model(base_model)
        params, param_names = eata.collect_params(base_model)
        ewc_optimizer = optim.SGD(params, 0.001)
        fishers = {}
        train_loss_fn = nn.CrossEntropyLoss().cuda()
        for iter_, (images, targets) in enumerate(fisher_loader, start=1):
            images, targets = images.cuda(), targets.cuda()
            outputs = base_model(images)
            _, targets = outputs.max(1)
            loss = train_loss_fn(outputs, targets)
            loss.backward()
            for name, param in base_model.named_parameters():
                if param.grad is not None:
                    if iter_ > 1: fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                    else: fisher = param.grad.data.clone().detach() ** 2
                    if iter_ == len(fisher_loader): fisher = fisher / iter_
                    fishers.update({name: [fisher, param.data.clone().detach()]})
            ewc_optimizer.zero_grad()
        del ewc_optimizer
        optimizer = setup_optimizer(params, lr)  
        if args.criterion == 'ent_unf': alpha = [1., .2]
        elif args.criterion == 'ent_ind_ood': alpha = [1., .2]
        else: alpha = [0.0, 0.0]
        
        model = eata.EATA(base_model, optimizer, fishers, args.fisher_alpha, e_margin=args.e_margin*math.log(1000), d_margin=args.d_margin, alpha=alpha, criterion=args.criterion)

    elif args.adaptation == "ostta":
        base_model = ostta.configure_model(base_model)
        params, param_names = ostta.collect_params(base_model)
        optimizer = setup_optimizer(params, lr)
        if args.criterion == 'ent_unf': alpha = [.2, .1]
        elif args.criterion == 'ent_ind_ood': alpha = [.2, .1]
        else: alpha = [1., .0]
        model = ostta.OSTTA(base_model, optimizer, steps=args.steps, episodic=args.episodic, alpha=alpha, criterion=args.criterion)
    
    elif args.adaptation == "caftta":
        base_model = caftta.configure_model(base_model)
        params, param_names = caftta.collect_params(base_model)
        optimizer = setup_optimizer(params, lr)
        model = caftta.CAFTTA(base_model, optimizer, steps=args.steps, episodic=False, n_aug=args.n_aug, hyperparameters=hyperparameters)

    elif args.adaptation == "filtering_with_main":
        base_model = filtering_with_main.configure_model(base_model)
        params, param_names = filtering_with_main.collect_params(base_model)
        optimizer = setup_optimizer(params, lr)
        model = filtering_with_main.CAFTTA(base_model, optimizer, steps=args.steps, episodic=args.episodic, n_aug=args.n_aug, hyperparameters=hyperparameters)
    
    elif args.adaptation == "filtering_with_ema":
        base_model = filtering_with_ema.configure_model(base_model)
        params, param_names = filtering_with_ema.collect_params(base_model)
        optimizer = setup_optimizer(params, lr)
        model = filtering_with_ema.CAFTTA(base_model, optimizer, steps=args.steps, episodic=args.episodic, n_aug=args.n_aug, hyperparameters=hyperparameters)

    elif args.adaptation == "caftta_palm":
        base_model = caftta_palm.configure_model(base_model)
        params, param_names = caftta_palm.collect_params(base_model)
        optimizer = setup_optimizer(params, args.lr)
        model = caftta_palm.CAFTTA(base_model, optimizer, steps=args.steps, episodic=args.episodic, 
                                teacher_ema=args.mt, hyperparameters=hyperparameters, n_class=1000)

    elif args.adaptation == "stamp":
        base_model = stamp.configure_model(base_model)
        params, param_names = stamp.collect_params(base_model)
        base_optimizer = optim.SGD
        optimizer = SAM(params, base_optimizer, lr=.01, rho=0.05)
        model = stamp.STAMP(base_model, optimizer, 1000, alpha=[0.8])

    elif args.adaptation == "rotta":
        base_model = rotta.configure_model(base_model, ALPHA=.05)
        params, param_names = rotta.collect_params(base_model)
        optimizer = setup_optimizer(params, lr)
        model = rotta.RoTTA(base_model, optimizer)
        
    elif args.adaptation == "caftta_three_outputs":
        base_model = caftta_three_outputs.configure_model(base_model)
        params, param_names = caftta_three_outputs.collect_params(base_model)
        optimizer = setup_optimizer(params, args.lr)
        model = caftta_three_outputs.CAFTTA(base_model, optimizer, steps=args.steps, episodic=False, n_aug=args.n_aug, hyperparameters=hyperparameters)
    elif args.adaptation == "sotta":
        base_model = sotta.configure_model(base_model)
        params, param_names = sotta.collect_params(base_model)
        base_optimizer = optim.Adam
        optimizer = SAM(params, base_optimizer, lr=.001, rho=0.5)
        model = sotta.SoTTA(base_model, optimizer)
    acc_arr_cum = torch.tensor([0., 0, 0, 0, 0, 0, 0, 0])
    for i in range(args.rounds):
        t = PrettyTable(["corruption", "acc", "auroc", "fpr95tpr", "oscr", "hscore"])
        top1, auroc, fpr95tpr, oscr, hscore = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for severity in args.severity:
            for corruption_type in args.type:
                # continual adaptation for all corruption
                logger.info("not resetting model")
                x_ind, y_ind = load_imagenetc(args.num_ex, severity, args.data_dir, True, [corruption_type])
                if args.open_set_tta == "True":
                    if args.ood_dataset == 'imagenet_o': x_ood, _ = load_imagenet_o_c(args.num_ex, severity, args.data_dir, True, [corruption_type])
                    elif args.ood_dataset == 'places365': x_ood, _ = load_places_c(args.num_ex, severity, args.data_dir, True, [corruption_type])
                    elif args.ood_dataset == 'textures': x_ood, _ = load_textures_c(args.num_ex, severity, args.data_dir, True, [corruption_type])
                    elif args.ood_dataset == 'gaussian': x_ood, _ = load_gaussian(args.num_ex)
                    elif args.ood_dataset == 'uniform': x_ood, _ = load_uniform(args.num_ex)

                    y_ood = torch.ones_like(y_ind).long() * -1
                    ind_dataset = TensorDataset(x_ind, y_ind)
                    ood_dataset = TensorDataset(x_ood, y_ood)
                    combined_dataset = ConcatDataset([ind_dataset, ood_dataset])
                    dataloader = DataLoader(combined_dataset, batch_size=2 * args.batch_size, shuffle=True)
                else:
                    ind_dataset = TensorDataset(x_ind, y_ind)
                    dataloader = DataLoader(ind_dataset, batch_size= args.batch_size, shuffle=True)

                acc, (auc, fpr), oscr_, acc_arr_cur = get_results(model, dataloader)
                acc_arr_cum += acc_arr_cur
                hscore_ = (2 * acc * auc) / (acc + auc)

                err = 1. - acc
                
                logger.info(f"[{corruption_type}{severity}] accuracy: {acc:.2%}, auroc: {auc:.2%}")
                t.add_row([f"{severity}/{corruption_type}", f"{acc:.2%}", f"{auc:.2%}", f"{fpr:.2%}", f"{oscr_:.2%}", f"{hscore_:.2%}"])
                top1.update(acc)
                auroc.update(auc)
                fpr95tpr.update(fpr)
                oscr.update(oscr_)
                hscore.update(hscore_)
                
        t.add_row(["mean", f"{top1.avg:.2%}", f"{auroc.avg:.2%}", f"{fpr95tpr.avg:.2%}", f"{oscr.avg:.2%}", f"{hscore.avg:.2%}"])
        logger.info(f"results of round {i}:\n{t}")

        print(acc_arr_cum / 15)
    return top1.avg, auroc.avg, hscore.avg

def setup_optimizer(params, lr):
    if args.method == "Adam": return optim.Adam(params, lr=lr)
    elif args.method == "SGD": return optim.SGD(params, lr, momentum=args.momentum)
    elif args.method == "SAM": return SAM(params, optim.SGD, lr=args.lr, rho=0.05)
    else: raise NotImplementedError

@torch.no_grad()
def entropy_from_logits(logits):
    p = F.softmax(logits, dim=-1)
    return -(p * p.log()).sum(dim=-1)          # (batch,)

import torch

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

import torch
import torch.nn.functional as F

def ccjw_ensemble(outputs, outputs_ema, caftta_outputs):
    probs = outputs.softmax(1)
    probs_ema = outputs_ema.softmax(1)
    probs_caftta = caftta_outputs.softmax(1)

    # confidence
    conf_outputs = probs.max(1).values
    conf_ema = probs_ema.max(1).values
    conf_caftta = probs_caftta.max(1).values

    # consensus (cosine similarity to others)
    def consensus(p, q1, q2):
        sim1 = F.cosine_similarity(p, q1, dim=1)
        sim2 = F.cosine_similarity(p, q2, dim=1)
        return (sim1 + sim2) / 2  # 평균 유사도

    cons_outputs = consensus(probs, probs_ema, probs_caftta)
    cons_ema = consensus(probs_ema, probs, probs_caftta)
    cons_caftta = consensus(probs_caftta, probs, probs_ema)

    # weight = confidence * consensus
    w_outputs = conf_outputs * cons_outputs
    w_ema = conf_ema * cons_ema
    w_caftta = conf_caftta * cons_caftta
    weights = torch.stack([w_outputs, w_ema, w_caftta], dim=1)  # (batch, 3)

    # normalize
    weights = weights / weights.sum(dim=1, keepdim=True)

    # final ensemble
    logit_final = (
        weights[:, [0]] * probs +
        weights[:, [1]] * probs_ema +
        weights[:, [2]] * probs_caftta
    )
    return logit_final

def get_results(model, dataloader):
    acc = 0.
    n_closed_set_data = 0
    y_true_auroc, y_true_knk_auroc, y_score, y_label = torch.zeros((0)), torch.zeros((0)), torch.zeros((0)), torch.zeros((20000))
    # score_ind, score_ood, pred = torch.zeros((0)), torch.zeros((0)), torch.zeros((0))
    # logits_main, logits_src, logits_ema = torch.zeros((20000, 10)), torch.zeros((20000, 10)), torch.zeros((20000, 10))
    total_inference_time = 0
    acc_arr = torch.tensor([0., 0, 0, 0, 0, 0, 0, 0])
    upper_bound_sum = 0
    alpha   = 0.05                                           # EMA 계수
    r_model = torch.tensor([0.34, 0.33, 0.33], device='cuda')# [adapt, EMA, source]
    with torch.no_grad():
        for data_ind, (x, y) in enumerate(tqdm(dataloader)):
            x, y = x.to('cuda'), y.to('cuda')
            output = model(x)
            if isinstance(output, tuple): (outputs, outputs_ema, caftta_outputs, energy) = output
            else: energy = output.logsumexp(1)
            
            # cos sim of outputs, outputs_ema, caftta_outputs
            """import torch.nn.functional as F
            # 1) logit → 확률
            probs         = F.softmax(outputs,         dim=1)
            probs_ema     = F.softmax(outputs_ema,     dim=1)
            probs_caftta  = F.softmax(caftta_outputs,  dim=1)

            # 2) 각 샘플별 코사인 유사도 계산  (결과 shape: (batch,))
            cos_ae = F.cosine_similarity(probs,        probs_ema,    dim=1)  # outputs ↔ outputs_ema
            cos_ac = F.cosine_similarity(probs,        probs_caftta, dim=1)  # outputs ↔ caftta_outputs
            cos_ec = F.cosine_similarity(probs_ema,    probs_caftta, dim=1)  # outputs_ema ↔ caftta_outputs

            # (선택) 배치 전체 평균이 필요하다면:
            print("mean cos(outputs, outputs_ema):", cos_ae.mean().item())
            print("mean cos(outputs, caftta):",      cos_ac.mean().item())
            print("mean cos(outputs_ema, caftta):",  cos_ec.mean().item())"""
            agree = outputs.argmax(-1) == caftta_outputs.argmax(-1)

            # ① 합의 O → 두 모델 평균
            logit_agree = outputs

            # ② 합의 X → 세 모델 평균 (혹은 source 단독)
            logit_disagree = (outputs + outputs_ema + caftta_outputs) / 3

            logit_final = torch.where(agree.unsqueeze(1), logit_agree, logit_disagree)

            conf_outputs = outputs.softmax(1).max(1).values  # (batch,)
            conf_outputs_ema = outputs_ema.softmax(1).max(1).values
            conf_caftta = caftta_outputs.softmax(1).max(1).values

            # 세 confidence 평균
            conf_mean = (conf_outputs + conf_outputs_ema + conf_caftta) / 3  # (batch,)

            # confidence deviation (중심화)
            delta_outputs = conf_outputs - conf_mean
            delta_ema = conf_outputs_ema - conf_mean
            delta_caftta = conf_caftta - conf_mean

            # 가중치: 1/3 ± α * 중심화된 confidence
            alpha = 0.1  # confidence 반영 정도 조절 (0이면 평균과 같음)
            w_outputs = 1/3 + alpha * delta_outputs
            w_ema = 1/3 + alpha * delta_ema
            w_caftta = 1/3 + alpha * delta_caftta
            weights = torch.stack([w_outputs, w_ema, w_caftta], dim=1)  # (batch, 3)
            # softmax된 확률 앙상블
            logit_final = (
                weights[:, [0]] * outputs.softmax(1) +
                weights[:, [1]] * outputs_ema.softmax(1) +
                weights[:, [2]] * caftta_outputs.softmax(1)
            )

            for o_ind, o in enumerate([outputs,
                                        outputs_ema,
                                        caftta_outputs,
                                        outputs + outputs_ema,
                                       confidence_weighted_ensemble(outputs, outputs_ema, caftta_outputs, alpha=0.15),
                                       confidence_weighted_ensemble(outputs, outputs_ema, caftta_outputs, alpha=0.20), 
                                       confidence_weighted_ensemble(outputs, outputs_ema, caftta_outputs, alpha=0.25),
                                       confidence_weighted_ensemble(outputs, outputs_ema, caftta_outputs, alpha=0.10)]):

                    # print(o[y.to('cpu') != -1].softmax(1).topk(3, dim=1)[0])
                max_logit, pred_ = o.max(1)
                prob = o.softmax(1) 
                max_prob, pred_ = prob.max(1)
                correct = pred_ == y
                acc_arr[o_ind] += correct.to('cpu').float().sum()
                if o_ind == 0:
                    n_closed_set_data += sum(y.to('cpu') != -1).item()

                if o_ind < 3:
                    mask_false = (~correct) & (y != -1)
                    mask_true = (correct) & (y != -1)
                    # print("틀린거", torch.max(o.softmax(dim=1)[mask_false], dim=1).values.mean().item())
                    if o_ind == 0:
                        wandb.log({"adapting_wrong_confidence": torch.max(o.softmax(dim=1)[mask_false], dim=1).values.mean().item()})
                    elif o_ind == 1:
                        wandb.log({"ema_wrong_confidence": torch.max(o.softmax(dim=1)[mask_false], dim=1).values.mean().item()})
                    elif o_ind == 2:
                        wandb.log({"source_wrong_confidence": torch.max(o.softmax(dim=1)[mask_false], dim=1).values.mean().item()})
                    # print("맞는거", torch.max(o.softmax(dim=1)[mask_true], dim=1).values.mean().item())
                    # print("="*20)

                if args.open_set_tta == "True" and o_ind == 7:
                    y_label = torch.cat((y_label, y.cpu().reshape(-1)))
                    y_true_auroc = torch.cat(( y_true_auroc , torch.where(y.to('cpu') == -1, torch.tensor(0), torch.tensor(1)) ), dim=0)
                    _condition = (correct.to('cpu') == False) | (y.to('cpu') == -1)
                    y_true_knk_auroc = torch.cat((y_true_knk_auroc, torch.where(_condition, torch.tensor(0), torch.tensor(1))), dim=0)
                    y_score = torch.cat((y_score, energy.cpu()), dim=0)
            even_one_correct = (outputs_ema.max(1)[1] == y) | (caftta_outputs.max(1)[1] == y) |  (outputs.max(1)[1] == y)
            upper_bound_sum += even_one_correct.sum().item()
    # print(n_closed_set_data)

    # print(total_inference_time / 100)
    print(acc_arr / n_closed_set_data)
    print('upper bound:', upper_bound_sum / n_closed_set_data)
    if args.open_set_tta == "True":
        return acc_arr[-1] / n_closed_set_data, get_ood_metrics(y_true_auroc.numpy(), y_score.numpy()), get_know_notknow_auroc(y_true_knk_auroc.numpy(), y_score.numpy()), acc_arr
    else:
        return acc_arr[-1] / n_closed_set_data, (0, 0), 0, acc_arr



def get_ood_metrics(y_true, y_score):
    auroc = metrics.roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    return auroc, float(interpolate.interp1d(tpr, fpr)(0.95))

def get_know_notknow_auroc(y_true, y_score):
    auroc = metrics.roc_auc_score(y_true, y_score)
    return auroc

def get_oscr(score_ind, score_ood, pred, y_ind):
    score = np.concatenate((score_ind, score_ood), axis=0)
    def get_fpr(t):
        return (score_ood >= t).sum() / len(score_ood)
    def get_ccr(t):
        return ((score_ind > t) & (pred == y_ind)).sum() / len(score_ind)
    fpr = [0.0]
    ccr = [0.0]
    for s in -np.sort(-score):
        fpr.append(get_fpr(s))
        ccr.append(get_ccr(s))
    fpr.append(1.0)
    ccr.append(1.0)
    roc = sorted(zip(fpr, ccr), reverse=True)
    oscr = 0.0
    for i in range(len(score)): oscr = oscr + (roc[i][0] - roc[i + 1][0]) * (roc[i][1] + roc[i + 1][1]) / 2.0
    return oscr



if __name__ == "__main__":
    best_hyperparameters = None
    best_hscore = 0.00
    best_top1_err = 0.00
    best_auroc = 0.00
    evaluate(args.alpha, args.lr)

