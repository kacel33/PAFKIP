import argparse
import math
import os
import sys
# Make repo-root `quantization/` importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
# Robustbench checkpoints contain numpy scalars; PyTorch 2.6+ rejects them
# under the new weights_only=True default. Allow-list and fall back if needed.
try:
    torch.serialization.add_safe_globals([np.core.multiarray.scalar, np.dtype, np.ndarray])
except Exception:
    pass
_orig_torch_load = torch.load
def _torch_load_allow(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_torch_load(*a, **kw)
torch.load = _torch_load_allow

import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from iopath.common.file_io import g_pathmgr
from prettytable import PrettyTable
from scipy import interpolate
from sklearn import metrics
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from tqdm import tqdm
from robustbench.data import load_imagenetc
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model

import cotta
import eata
import ostta
import norm
import tent
import caftta
import ours
from data import load_imagenet_o_c
from utils import AverageMeter, get_logger, set_random_seed


parser = argparse.ArgumentParser()

# Model options
parser.add_argument("--arch", default="Hendrycks2020AugMix",
                    choices=["Hendrycks2020AugMix", "Tian2022Deeper_DeiT-B"])
parser.add_argument("--adaptation", default="tent")
parser.add_argument("--episodic", action="store_true")
# Corruption options
parser.add_argument("--dataset", default="imagenet")
parser.add_argument("--type", default="gaussian_noise")
parser.add_argument("--severity", default=5, type=int)
parser.add_argument("--num_ex", default=5000, type=int)
parser.add_argument("--open_set_tta", default="True", type=str)
# Optimizer options
parser.add_argument("--steps", default=1, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--method", default="SGD", choices=["Adam", "SGD"])
parser.add_argument("--momentum", default=0.9, type=float)
# Testing options
parser.add_argument("--batch_size", default=32, type=int)  # default=32 but oom issue
# Misc options
parser.add_argument("--rng_seed", default=1, type=int)
parser.add_argument("--save_dir", default="./output")
parser.add_argument("--data_dir", default=os.path.expanduser("~/imagenet"))
parser.add_argument("--ckpt_dir", default="./ckpt")
parser.add_argument("--log_dest", default="log.txt")
# Tent options
parser.add_argument("--alpha", nargs="+", default=[0.5], type=float)
parser.add_argument("--criterion", default="ent", choices=["ent", "ent_ind", "ent_ind_ood", "ent_unf"])
parser.add_argument("--rounds", default=1, type=int)
# EATA options
parser.add_argument("--fisher_size", default=2000, type=int)
parser.add_argument("--fisher_alpha", default=2000., type=float)
parser.add_argument("--e_margin", default=math.log(1000)*0.40, type=float)
parser.add_argument("--d_margin", default=0.05, type=float)

# PAF-KIP options
parser.add_argument("--n_aug", default=1, type=int)
parser.add_argument("--mt", default=0.999, type=float)
parser.add_argument("--thr", default=0.4, type=float)
parser.add_argument("--ours_alpha", default=2.0, type=float)

# OOD source for open-set TTA
parser.add_argument("--ood_dataset", default="imagenet_o_c",
                    choices=["imagenet_o_c", "gaussian", "uniform"])

# MX (Microscaling) fake-quant
parser.add_argument("--mx_quantize", action="store_true")
parser.add_argument("--mx_format", default="int4", choices=["fp4", "int4"])
parser.add_argument("--mx_group_size", default=32, type=int)
parser.add_argument("--mx_no_e8m0", action="store_true")
parser.add_argument("--mx_no_act", action="store_true")
parser.add_argument("--mx_no_skip_first", action="store_true")

# Global (per-tensor act + per-channel weight) PTQ
parser.add_argument("--quantize", action="store_true",
                    help="Enable global INT4 PTQ (per-tensor act, per-channel weight).")
parser.add_argument("--w_bits", default=4, type=int)
parser.add_argument("--a_bits", default=4, type=int)
parser.add_argument("--act_percentile", default=0.999, type=float)
parser.add_argument("--mse_weight", action="store_true", default=True)
parser.add_argument("--skip_first_conv", action="store_true", default=True)
parser.add_argument("--act_granularity", default="per_tensor",
                    choices=["per_tensor", "per_channel", "per_sample"])

args = parser.parse_args()

args.type = ["gaussian_noise", "shot_noise", "impulse_noise",
             "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
             "snow", "frost", "fog", "brightness", "contrast",
             "elastic_transform", "pixelate", "jpeg_compression"]
args.severity = [5]
args.log_dest = "{}_{}_lr_{}_alpha_{}_{}.txt".format(
    args.adaptation, args.dataset, args.lr, "_".join(str(alpha) for alpha in args.alpha), args.criterion)

g_pathmgr.mkdirs(args.save_dir)

set_random_seed(args.rng_seed)

logger = get_logger(__name__, args.save_dir, args.log_dest)
logger.info(f"args:\n{args}")


def _load_ood(num_ex, severity, data_dir, corruption_type, dev):
    """Return [N, 3, 224, 224] OOD tensor on device per --ood_dataset selection."""
    if args.ood_dataset == "imagenet_o_c":
        x, _ = load_imagenet_o_c(num_ex, severity, data_dir, True, [corruption_type])
        return x.to(dev)
    # Synthetic OOD: noise images, ImageNet-normalized in [0,1] then standard mean/std.
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    if args.ood_dataset == "gaussian":
        x = torch.randn(num_ex, 3, 224, 224).clamp(-2, 2)
        x = (x * 0.25 + 0.5).clamp(0, 1)
    else:  # uniform
        x = torch.rand(num_ex, 3, 224, 224)
    x = (x - mean) / std
    return x.to(dev)


def evaluate(hyperparameters=None):
    # configure model
    base_model = load_model(args.arch, args.ckpt_dir, args.dataset, "corruptions").cuda()

    if args.mx_quantize:
        from quantization import apply_mx_quantization
        base_model = apply_mx_quantization(
            base_model,
            fp4=(args.mx_format == "fp4"),
            group_size=args.mx_group_size,
            use_e8m0=(not args.mx_no_e8m0),
            quant_act=(not args.mx_no_act),
            skip_first=(not args.mx_no_skip_first),
        )

    if args.quantize:
        from quantization import apply_quantization
        base_model = apply_quantization(
            base_model,
            w_bits=args.w_bits, a_bits=args.a_bits,
            act_percentile=args.act_percentile,
            mse_weight=args.mse_weight,
            skip_first=args.skip_first_conv,
            act_granularity=args.act_granularity,
        )

    if args.adaptation == "source":
        base_model.eval()
        model = base_model
    elif args.adaptation == "norm":
        model = norm.Norm(base_model)
    elif args.adaptation == "cotta":
        base_model = cotta.configure_model(base_model)
        params, param_names = cotta.collect_params(base_model)
        optimizer = setup_optimizer(params)
        model = cotta.CoTTA(base_model, optimizer, steps=args.steps, episodic=args.episodic)

    elif args.adaptation == "tent":
        base_model = tent.configure_model(base_model)
        params, param_names = tent.collect_params(base_model)
        optimizer = setup_optimizer(params)
        model = tent.Tent(base_model, optimizer, steps=args.steps, episodic=args.episodic, alpha=args.alpha, criterion=args.criterion)
    
    elif args.adaptation == "eata":
        fisher_dataset = datasets.ImageFolder(args.data_dir + "/imagenet/images/train",
                                              transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
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
                    if iter_ > 1:
                        fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                    else:
                        fisher = param.grad.data.clone().detach() ** 2
                    if iter_ == len(fisher_loader):
                        fisher = fisher / iter_
                    fishers.update({name: [fisher, param.data.clone().detach()]})
            ewc_optimizer.zero_grad()
        del ewc_optimizer
        optimizer = setup_optimizer(params)
        model = eata.EATA(base_model, optimizer, fishers, args.fisher_alpha, e_margin=args.e_margin, d_margin=args.d_margin, alpha=args.alpha, criterion=args.criterion)

    elif args.adaptation == "ostta":
        base_model = ostta.configure_model(base_model)
        params, param_names = ostta.collect_params(base_model)
        optimizer = setup_optimizer(params)
        model = ostta.OSTTA(base_model, optimizer, steps=args.steps, episodic=args.episodic, alpha=args.alpha, criterion=args.criterion)
    
    elif args.adaptation == "caftta":
        base_model = caftta.configure_model(base_model)
        params, param_names = caftta.collect_params(base_model)
        optimizer = setup_optimizer(params)
        model = caftta.CAFTTA(base_model, optimizer, steps=args.steps, episodic=args.episodic, hyperparameters=hyperparameters)

    elif args.adaptation == "ours":
        base_model = ours.configure_model(base_model)
        params, param_names = ours.collect_params(base_model)
        optimizer = setup_optimizer(params)
        model = ours.PAFKIP(base_model, optimizer, steps=args.steps, episodic=args.episodic,
                            n_aug=args.n_aug, teacher_ema=args.mt, ent_thr_ratio=args.thr, alpha=args.ours_alpha)

    for i in range(args.rounds):
        t = PrettyTable(["corruption", "acc", "auroc", "fpr95tpr", "oscr"])
        top1 = AverageMeter()
        auroc, fpr95tpr, oscr = AverageMeter(), AverageMeter(), AverageMeter()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for severity in args.severity:
            for corruption_type in args.type:
                # continual adaptation for all corruption
                logger.info("not resetting model")
                x_ind, y_ind = load_imagenetc(args.num_ex, severity, args.data_dir, True, [corruption_type])
                x_ind, y_ind = x_ind.to(device), y_ind.to(device)
                x_ood = _load_ood(args.num_ex, severity, args.data_dir, corruption_type, device)

                acc, (auc, fpr), oscr_ = get_results(model, x_ind, y_ind, x_ood, args.batch_size, device)
                err = 1. - acc
                logger.info(f"error % [{corruption_type}{severity}]: {err:.2%}")
                t.add_row([f"{severity}/{corruption_type}", f"{acc:.2%}", f"{auc:.2%}", f"{fpr:.2%}", f"{oscr_:.2%}"])
                top1.update(acc)
                auroc.update(auc)
                fpr95tpr.update(fpr)
                oscr.update(oscr_)
        t.add_row(["mean", f"{top1.avg:.2%}", f"{auroc.avg:.2%}", f"{fpr95tpr.avg:.2%}", f"{oscr.avg:.2%}"])
        logger.info(f"results of round {i}:\n{t}")


def setup_optimizer(params):
    if args.method == "Adam": return optim.Adam(params, lr=args.lr)
    elif args.method == "SGD": return optim.SGD(params, args.lr, momentum=args.momentum)
    else:
        raise NotImplementedError


def get_results(model: nn.Module,
                x_ind: torch.Tensor,
                y_ind: torch.Tensor,
                x_ood: torch.Tensor,
                batch_size: int = 100,
                device: torch.device = None):
    if device is None: device = x_ind.device
    acc = 0.
    # evaluate on each severity and type of corruption in turn
    # model = nn.DataParallel(model)
    model.to(device)
    y_true, y_score = torch.zeros((0)), torch.zeros((0))
    score_ind, score_ood, pred = torch.zeros((0)), torch.zeros((0)), torch.zeros((0))

    n_batches = math.ceil(x_ind.shape[0] / batch_size)
    in_domain_dataset = TensorDataset(x_ind, y_ind)
    in_domain_dataloader = DataLoader(in_domain_dataset, batch_size=batch_size, shuffle=False)
    out_domain_dataloader = DataLoader(x_ood, batch_size=batch_size, shuffle=False)

    dataloader_iterator = iter(in_domain_dataloader)

    with torch.no_grad():
        for x_ood_curr in tqdm(out_domain_dataloader):
            x_ind_curr, y_ind_curr = next(dataloader_iterator)
            x_curr = torch.cat((x_ind_curr, x_ood_curr), dim=0)
            output = model(x_curr)
            if isinstance(output, tuple): (output, energy) = output
            else: energy = output.logsumexp(1)
            max_logit, pred_ = output.max(1)
            prob = output.softmax(1)
            max_prob, pred_ = prob.max(1)

            acc = acc + (pred_[:x_ind_curr.shape[0]] == y_ind_curr).float().sum()

            y_true = torch.cat((y_true, torch.cat((torch.ones(x_ind_curr.shape[0]), torch.zeros(x_ood_curr.shape[0])), dim=0)), dim=0)
            y_score = torch.cat((y_score, energy.cpu()), dim=0)
            score_ind = torch.cat((score_ind, energy[:x_ind_curr.shape[0]].cpu()), dim=0)
            score_ood = torch.cat((score_ood, energy[x_ood_curr.shape[0]:].cpu()), dim=0)
            pred = torch.cat((pred, pred_[:x_ind_curr.shape[0]].cpu()), dim=0)

    return acc.item() / x_ind.shape[0], get_ood_metrics(y_true.numpy(), y_score.numpy()), \
           get_oscr(score_ind.numpy(), score_ood.numpy(), pred.numpy(), y_ind.cpu().numpy())


def get_ood_metrics(y_true, y_score):
    auroc = metrics.roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    return auroc, float(interpolate.interp1d(tpr, fpr)(0.95))


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
    evaluate(args.alpha)
