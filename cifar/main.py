import argparse
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from iopath.common.file_io import g_pathmgr
from prettytable import PrettyTable
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset, ConcatDataset
from tqdm import tqdm

from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model

from methods import cotta, eata, ostta, tent, sotta, stamp, rotta, ours
from data import load_svhn_c, load_tiny_imagenet_c, load_textures_c, load_places365_c
from utils import AverageMeter, set_random_seed
from sam import SAM
from robustbench.data import load_cifar10c, load_cifar100c

def load_args():
    parser = argparse.ArgumentParser()
    # Model options
    parser.add_argument("--arch", default="Hendrycks2020AugMix_WRN")  
    parser.add_argument("--adaptation", default="tent")
    parser.add_argument("--episodic", action="store_true")

    # Corruption options
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--ood_dataset", default="svhn", choices=["svhn", "tiny_imagenet", "textures", "places365", 'gaussian', 'uniform'])
    parser.add_argument("--open_set_tta", default="True", type=str)
    parser.add_argument("--open_set_ratio", default=1.0, type=float)
    parser.add_argument("--severity", default=5, type=int)
    parser.add_argument("--num_ex", default=10000, type=int)

    # Optimizer options
    parser.add_argument("--steps", default=1, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--method", default="Adam", choices=["Adam", "SGD", "SAM"])
    parser.add_argument("--momentum", default=0.9, type=float)

    # Testing options
    parser.add_argument("--batch_size", default=200, type=int)
    parser.add_argument("--continual", default="True", type=str)
    # Misc options
    parser.add_argument("--rng_seed", default=1, type=int)
    parser.add_argument("--save_dir", default="./output")
    parser.add_argument("--data_dir", default=os.path.expanduser("~"))
    parser.add_argument("--ckpt_dir", default="./ckpt")
    parser.add_argument("--log_dest", default="log.txt")

    # CoTTA options
    parser.add_argument("--mt", default=0.999, type=float)
    parser.add_argument("--rst", default=0.01, type=float)
    parser.add_argument("--ap", default=0.92, type=float)

    parser.add_argument("--criterion", default="ent", choices=["ent", "ent_ind", "ent_ind_ood", "ent_unf"])  # 일반 ent / ent_ind / UniEnt / UniEnt+
    parser.add_argument("--rounds", default=1, type=int)

    # EATA options
    parser.add_argument("--fisher_size", default=2000, type=int)
    parser.add_argument("--fisher_alpha", default=1., type=float)
    parser.add_argument("--e_margin", default=math.log(10)*0.40, type=float)
    parser.add_argument("--d_margin", default=0.4, type=float)

    parser.add_argument("--n_aug", default=1, type=int)
    parser.add_argument("--thr", default=0.4, type=float)
    parser.add_argument("--alpha", default=2.0, type=float)

    args = parser.parse_args()

    args.type = ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
                "snow", "frost", "fog", "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression"]

    args.severity = [5]

    args.e_margin = math.log(10)*0.40 if args.dataset == "cifar10" else math.log(100)*0.40

    return args

def evaluate(args):
    num_class = 10 if args.dataset == "cifar10" else 100
    g_pathmgr.mkdirs(args.save_dir)

    set_random_seed(args.rng_seed)

    base_model = load_model(args.arch, args.ckpt_dir, args.dataset, ThreatModel.corruptions).cuda()

    if args.adaptation == "source":
        base_model.eval()
        model = base_model

    elif args.adaptation in "cotta" :
        base_model = cotta.configure_model(base_model)
        params, _ = cotta.collect_params(base_model)
        optimizer = setup_optimizer(params, args.lr)
        ap = .92 if args.dataset == 'cifar10' else .72
        model = cotta.CoTTA(base_model, optimizer, steps=args.steps, episodic=args.episodic, mt_alpha=args.mt, rst_m=args.rst, ap=ap, criterion=args.criterion)
        
    elif args.adaptation == "tent":
        base_model = tent.configure_model(base_model)
        params, _ = tent.collect_params(base_model)
        optimizer = setup_optimizer(params, args.lr)
        if args.criterion == 'ent_unf': alpha = [1., 1.] if args.dataset == 'cifar10' else [.2, .2]
        elif args.criterion == 'ent_ind_ood': alpha = [1., .5] if args.dataset == 'cifar10' else [.2, .2]
        else: alpha = [0.0, 0.0]
        model = tent.Tent(base_model, optimizer, steps=args.steps, episodic=args.episodic, alpha=alpha, criterion=args.criterion)
        
    elif args.adaptation == "eata":
        if args.dataset == 'cifar10':
            fisher_dataset = eval("datasets." + f"{args.dataset}".upper())(args.data_dir, transform=transforms.ToTensor(), download=True)
        sampled_indices = torch.randperm(len(fisher_dataset))[:args.fisher_size]
        sampler = SubsetRandomSampler(sampled_indices)
        fisher_loader = DataLoader(fisher_dataset, batch_size=args.batch_size, sampler=sampler)

        base_model = eata.configure_model(base_model)
        params, _ = eata.collect_params(base_model)
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

        optimizer = setup_optimizer(params, args.lr)
        if args.criterion == 'ent_unf': alpha = [.1, 1.] if args.dataset == 'cifar10' else [.5, .5]
        elif args.criterion == 'ent_ind_ood': alpha = [1., 1.] if args.dataset == 'cifar10' else [.2, .5]
        else: alpha = [0.0, 0.0]
        args.d_margin = 0.2
        args.e_margin = math.log(10) * 0.6
        model = eata.EATA(base_model, optimizer, fishers, args.fisher_alpha, e_margin=args.e_margin, d_margin=args.d_margin, alpha=alpha, criterion=args.criterion)

    elif args.adaptation == "ostta":
        base_model = ostta.configure_model(base_model)
        params, _ = ostta.collect_params(base_model)
        optimizer = setup_optimizer(params, args.lr)
        if args.criterion == 'ent_unf': alpha = [1., .2] if args.dataset == 'cifar10' else [.5, .1]
        elif args.criterion == 'ent_ind_ood': alpha = [.5, .2] if args.dataset == 'cifar10' else [.5, .1]
        else: alpha = [1.0, 0.0] if args.dataset == 'cifar10' else [0.5, 0.0]
        model = ostta.OSTTA(base_model, optimizer, steps=args.steps, episodic=args.episodic, alpha=alpha, criterion=args.criterion)
    
    elif args.adaptation == "sotta":
        base_model = sotta.configure_model(base_model)
        params, _ = sotta.collect_params(base_model)
        base_optimizer = optim.Adam
        optimizer = SAM(params, base_optimizer, lr=args.lr, rho=0.5)
        threshold = 0.99 if args.dataset == 'cifar10' else 0.66 
        model = sotta.SoTTA(base_model, optimizer, memory_size=200, threshold=threshold, num_class=num_class, steps=args.steps, episodic=args.episodic)

    elif args.adaptation == "stamp":
        base_model = stamp.configure_model(base_model)
        params, _ = stamp.collect_params(base_model)
        args.method == 'SAM'
        if args.method == 'SAM':
            args.lr = .1 if args.dataset == 'cifar10' else .05

        optimizer = setup_optimizer(params, args.lr)
        if args.dataset == 'cifar10': alpha = .25
        elif args.dataset == 'cifar100': alpha = .9
        model = stamp.STAMP(base_model, optimizer, num_class, alpha=[alpha], n_aug = args.n_aug)

    elif args.adaptation == "rotta":
        base_model = rotta.configure_model(base_model, ALPHA=.05)
        params, param_names = rotta.collect_params(base_model)
        optimizer = setup_optimizer(params, args.lr)
        model = rotta.RoTTA(base_model, optimizer, num_class)

    elif args.adaptation == "ours":
        base_model = ours.configure_model(base_model)
        params, _ = ours.collect_params(base_model)
        optimizer = setup_optimizer(params, args.lr)
        model = ours.PAFKIP(base_model, optimizer, steps=args.steps, episodic=args.episodic, n_aug=args.n_aug, 
                                teacher_ema=args.mt, ent_thr_ratio=args.thr, alpha=args.alpha)

    for i in range(args.rounds):
        t = PrettyTable(["corruption", "acc", "auroc", "fpr95tpr", "auknkroc", "hscore"])
        top1, auroc, fpr95tpr, auknkroc, hscore = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

        for severity in args.severity:
            for corruption_type in args.type:
                if args.continual == "True": print("not resetting model")
                else:
                    if args.adaptation != 'source':
                        print('resetting model')
                        model.reset()

                if args.open_set_ratio <= 1.:
                    x_ind, y_ind = eval(f"load_{args.dataset}c")(args.num_ex, severity, args.data_dir, False, [corruption_type])
                    num_ood_ex = int(args.num_ex * args.open_set_ratio)
                else:
                    x_ind, y_ind = eval(f"load_{args.dataset}c")(int(args.num_ex * (1 / args.open_set_ratio)), severity, args.data_dir, False, [corruption_type])
                    num_ood_ex = args.num_ex
                if args.ood_dataset == 'svhn': x_ood, _ = load_svhn_c(num_ood_ex, severity, args.data_dir, False, [corruption_type])
                elif args.ood_dataset == 'tiny_imagenet': x_ood, _ = load_tiny_imagenet_c(num_ood_ex, severity, args.data_dir, True, [corruption_type])
                elif args.ood_dataset == 'places365': x_ood, _ = load_places365_c(num_ood_ex, severity, args.data_dir, True, [corruption_type])
                elif args.ood_dataset == 'textures': x_ood, _ = load_textures_c(num_ood_ex, severity, args.data_dir, True, [corruption_type])
                if args.open_set_tta == "True":
                    x_ind, y_ind, x_ood = x_ind.cuda(), y_ind.cuda(), x_ood.cuda()
                    y_ood = torch.ones(num_ood_ex).long().cuda() * -1
                    ind_dataset = TensorDataset(x_ind, y_ind)
                    ood_dataset = TensorDataset(x_ood, y_ood)
                    combined_dataset = ConcatDataset([ind_dataset, ood_dataset])
                    dataloader = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
                else:
                    x_ind, y_ind = x_ind.cuda(), y_ind.cuda()
                    ind_dataset = TensorDataset(x_ind, y_ind)
                    dataloader = DataLoader(ind_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
                
                acc, (auc, fpr), auknkroc_ = get_results(model, dataloader)

                hscore_ = (2 * acc * auc) / (acc + auc)

                print(f"[{corruption_type}{severity}] accuracy: {acc:.2%}, auroc: {auc:.2%}, auknkroc: {auknkroc_:.2%}")
                t.add_row([f"{severity}/{corruption_type}", f"{acc:.2%}", f"{auc:.2%}", f"{fpr:.2%}", f"{auknkroc_:.2%}", f"{hscore_:.2%}"])
                top1.update(acc)
                auroc.update(auc)
                fpr95tpr.update(fpr)
                auknkroc.update(auknkroc_)
                hscore.update(hscore_)

        t.add_row(["mean", f"{top1.avg:.2%}", f"{auroc.avg:.2%}", f"{fpr95tpr.avg:.2%}", f"{auknkroc.avg:.2%}", f"{hscore.avg:.2%}"])
        print(f"results of round {i}:\n{t}")

    return top1.avg, auroc.avg, hscore.avg

def setup_optimizer(params, lr):
    if args.method == "Adam": return optim.Adam(params, lr=lr)
    elif args.method == "SGD": return optim.SGD(params, lr, momentum=args.momentum)
    elif args.method == "SAM": return SAM(params, optim.SGD, lr=lr, rho=0.05)
    else: raise NotImplementedError

def get_results(model, dataloader):
    acc = 0.
    n_closed_set_data = 0
    y_true_auroc, y_true_knk_auroc, y_score, y_label = torch.zeros((0)), torch.zeros((0)), torch.zeros((0)), torch.zeros((20000))
    total_inference_time = 0
    with torch.no_grad():
        for _, (x, y) in enumerate(tqdm(dataloader)):
            output = model(x)

            if isinstance(output, tuple): 
                (output, energy) = output
            else: 
                energy = output.logsumexp(1)

            max_logit, pred_ = output.max(1)
            prob = output.softmax(1) 
            max_prob, pred_ = prob.max(1)
            correct = pred_ == y
            acc += correct.float().sum()
            n_closed_set_data += sum(y.to('cpu') != -1).item()
            
            if args.open_set_tta == "True":
                y_label = torch.cat((y_label, y.cpu().reshape(-1)))
                y_true_auroc = torch.cat(( y_true_auroc , torch.where(y.to('cpu') == -1, torch.tensor(0), torch.tensor(1)) ), dim=0)
                _condition = (correct.to('cpu') == False) | (y.to('cpu') == -1)
                y_true_knk_auroc = torch.cat((y_true_knk_auroc, torch.where(_condition, torch.tensor(0), torch.tensor(1))), dim=0)
                y_score = torch.cat((y_score, energy.cpu()), dim=0)
  
    if args.open_set_tta == "True":
        return acc.item() / n_closed_set_data, get_ood_metrics(y_true_auroc.numpy(), y_score.numpy()), get_know_notknow_auroc(y_true_knk_auroc.numpy(), y_score.numpy())
    else:
        return acc.item() / n_closed_set_data, (0, 0), 0


def get_ood_metrics(y_true, y_score):
    auroc = roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    
    return auroc, float(np.interp(0.95, tpr, fpr)) 

def get_know_notknow_auroc(y_true, y_score):
    auroc = roc_auc_score(y_true, y_score)
    return auroc

def get_oscr(score_ind, score_ood, pred, y_ind):
    score = np.concatenate((score_ind, score_ood), axis=0)
    def get_fpr(t): return (score_ood >= t).sum() / len(score_ood)
    def get_ccr(t): return ((score_ind > t) & (pred == y_ind)).sum() / len(score_ind)
    fpr, ccr = [0.0], [0.0]
    for s in -np.sort(-score):
        fpr.append(get_fpr(s))
        ccr.append(get_ccr(s))
    fpr.append(1.0)
    ccr.append(1.0)
    roc = sorted(zip(fpr, ccr), reverse=True)
    oscr = 0.0
    for i in range(len(score)): 
        oscr += (roc[i][0] - roc[i + 1][0]) * (roc[i][1] + roc[i + 1][1]) / 2.0
    return oscr

if __name__ == "__main__":
    args = load_args()
    evaluate(args)
