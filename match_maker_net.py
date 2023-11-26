import os
import argparse
import numpy as np
import sklearn.metrics as metrics

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torchvision.ops import sigmoid_focal_loss
from torchmetrics.classification import BinaryF1Score, BinaryConfusionMatrix


from src.config import get_cfg_defaults
from src.model import load_model
from src.dataloader import BreakingBad
from src.model.mmn import MatchMakerNet
from src.utils.checkpoints import IOStream, SaveBestModel, EarlyStopper, checkpoint_init



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(BASE_DIR, 'data/geometry')



def train_step(model, train_loader, device, opt, criterion, epoch, io):
    train_loss = 0.0
    count = 0.0
    model.train()
    train_pred = []
    train_true = []
    for batch_idx, (obj1, obj2, targets) in enumerate(train_loader):
        obj1, obj2, targets = obj1.to(device, dtype=torch.float), obj2.to(device, dtype=torch.float), targets.to(device).unsqueeze(1)
        obj1, obj2 = obj1.permute(0, 2, 1), obj2.permute(0, 2, 1)
        batch_size = obj1.size()[0]
        opt.zero_grad()
        logits = model(obj1, obj2)
        loss = criterion(logits, targets)
        loss.backward()
        opt.step()
        preds = logits.max(dim=1)[1]
        count += batch_size
        train_loss += loss.item() * batch_size
        train_true.append(targets.cpu().numpy())
        train_pred.append(preds.detach().cpu().numpy())
        if batch_idx % cfg.exp.log_interval == 0:
            io.cprint('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(obj1), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    train_loss = train_loss*1.0/count
    train_acc = metrics.accuracy_score(train_true, train_pred)
    train_f1 = BinaryF1Score()(train_true, train_pred)
    outstr = f'Train {epoch}, loss: {train_loss:.6f}, train acc: {train_acc:.6f}, train f1: {train_f1:.6f}'
    io.cprint(outstr)
    return train_loss, train_acc, train_f1


def test_step(model, test_loader, device, criterion, epoch, io, name='train'):
    test_loss, f1_score, accuracy = 0, 0, 0
    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
    model.eval()
    test_pred = []
    test_true = []
    
    bf1 = BinaryF1Score().to(device)
    bcm = BinaryConfusionMatrix(normalize='all').to(device)
    for _, (obj1, obj2, targets) in enumerate(test_loader):
        obj1, obj2, targets = obj1.to(device, dtype=torch.float), obj2.to(device, dtype=torch.float), targets.to(device).unsqueeze(1)
        obj1, obj2 = obj1.permute(0, 2, 1), obj2.permute(0, 2, 1)
        logits = model(obj1, obj2)
        test_loss += criterion(logits, targets).mean().item() 
        pred = torch.where(logits > 0.5, 1, 0) 
        tn, fp, fn, tp = bcm(targets, pred).ravel()
        true_neg += tn.item()
        false_pos += fp.item()
        false_neg += fn.item()
        true_pos += tp.item()
        accuracy += pred.eq(targets.squeeze(0).view_as(pred.squeeze(0))).mean().item()
        f1_score += bf1(targets.squeeze(0), pred.squeeze(0)).item()
        test_true.append(targets.cpu().numpy())
        test_pred.append(pred.detach().cpu().numpy())

    false_pos /= len(test_loader)
    false_neg /= len(test_loader)
    true_pos /= len(test_loader)
    true_neg /= len(test_loader)
    
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    f1_score /= len(test_loader)
    test_loss /= len(test_loader)
    accuracy = metrics.accuracy_score(test_true, test_pred)

    io.cprint('{} {}: Average loss: {:.4f}, F1 score: {} ({:.0f}%), Accuracy: {} ({:.0f}%)'.format(
        name, epoch, test_loss, np.round(f1_score, 4), 100 * f1_score, np.round(accuracy, 4), 100*accuracy))
    io.cprint(f'Value of true negative: {np.round(true_neg, 2)} items')
    io.cprint(f'Value of false positive: {np.round(false_pos, 2)} items')
    io.cprint(f'Value of false negative: {np.round(false_neg, 2)} items')
    io.cprint(f'Value of true positive: {np.round(true_pos, 2)} items')
    
    return test_loss, accuracy, f1_score


def test_dist(test_partition, device, criterion, epoch, io):
    model = MatchMakerNet(args.model).to(device)
    model = nn.DataParallel(model)
    model = load_model(args, model)
    
    io.cprint(f'\n\n\nWith same distribution\n')
    test_loader = DataLoader(BreakingBad(partition=test_partition, mode='test', args=args.data, data_dir=data_dir),
                            batch_size=args.exp.test_batch_size, shuffle=True, drop_last=False)
    loss_same, acc_same, f1_same = test_step(model, test_loader, device, criterion, epoch, io, name='Test')
    
    io.cprint(f'\n\n\nWith 50 50 distribution\n')
    args.data.prop_pn = 0.5
    test_loader = DataLoader(BreakingBad(partition=test_partition, mode='test', args=args.data, data_dir=data_dir),
                            batch_size=args.exp.test_batch_size, shuffle=True, drop_last=False)
    loss_half, acc_half, f1_half = test_step(model, test_loader, device, criterion, epoch, io, name='Test')
    
    io.cprint(f'\n\n\nWith complete all to all distribution\n')
    args.data.prop_pn = 0.01
    test_loader = DataLoader(BreakingBad(partition=test_partition, mode='test', args=args.data, data_dir=data_dir),
                            batch_size=args.exp.test_batch_size, shuffle=True, drop_last=False)
    loss_cmp, acc_cmp, f1_cmp = test_step(model, test_loader, device, criterion, epoch, io, name='Test')
    
    return [loss_same, loss_half, loss_cmp], [acc_same, acc_half, acc_cmp], [f1_same, f1_half, f1_cmp]


def train(args, io, train_partition, val_partition, test_partition):
    train_loader = DataLoader(BreakingBad(partition=train_partition, mode='train', args=args.data, data_dir=data_dir), 
                                               batch_size=args.exp.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(BreakingBad(partition=val_partition, mode='test', args=args.data, data_dir=data_dir),
                                             batch_size=args.exp.test_batch_size, shuffle=True, drop_last=False)
    device = torch.device("cuda" if args.model.cuda else "cpu")
    
    model = MatchMakerNet(args.model).to(device)
    model = nn.DataParallel(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    io.cprint(str(model))
    io.cprint(f'NUMBER OF TRAINABLE PARAMS: {total_params}')
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    
    if args.opt.name == 'SGD':
        opt = optim.SGD(model.parameters(), lr=args.opt.lr, momentum=args.opt.gamma, weight_decay=args.opt.weight_decay)
    elif args.opt.name == 'Adam':
        opt = optim.Adam(model.parameters(), lr=args.opt.lr, weight_decay=args.opt.weight_decay)
    elif args.opt.name == 'Adadelta':
        opt = optim.Adadelta(model.parameters(), lr=args.opt.lr, weight_decay=args.opt.weight_decay)
    else:
        raise Exception('Optmizer not implemented')
    
    if args.opt.scheduler == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(opt, patience=1, verbose=True, factor=args.opt.gamma)
    elif args.opt.scheduler == 'StepLR':
        scheduler = StepLR(opt, step_size=1, gamma=args.opt.gamma, verbose=True)
    elif args.opt.scheduler == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(opt, args.exp.epochs, eta_min=args.opt.lr_min, verbose=True)
    else:
        raise Exception('Scheduler not implemented')
    io.cprint(str(opt))
    
    if args.model.loss == 'focal_loss':
        criterion = sigmoid_focal_loss
    elif args.model.loss == 'bce_loss':
        lst = torch.FloatTensor([cfg.data.prop_pn]).to(device, dtype=torch.float)
        criterion = nn.BCEWithLogitsLoss(pos_weight=lst)
    else:
        raise Exception('Loss criterion not implemented')
    
    save_best_model = SaveBestModel(dir=BASE_DIR, id=cfg.exp.id, idx=i)
    early_stopper = EarlyStopper(patience=20)
    
    loss_train_lt, loss_val_lt = [], []
    acc_train_lt, acc_val_lt = [], []
    f1_train_lt, f1_val_lt = [], []
    
    for epoch in range(args.exp.epochs):
        if epoch:
            scheduler.step()
            train_step(model, train_loader, device, opt, criterion, epoch, io)
        train_loss, train_acc, train_f1 = test_step(model, train_loader, device, opt, criterion, epoch, io, name='Train')
        val_loss, val_acc, val_f1 = test_step(model, val_loader, device, criterion, epoch, io, name='Val')
        
        loss_train_lt.append(train_loss), loss_val_lt.append(val_loss)
        acc_train_lt.append(train_acc), acc_val_lt.append(val_acc)
        f1_train_lt.append(train_f1), f1_val_lt.append(val_f1)
        save_best_model(val_loss, epoch, model, opt, criterion)
            
        if early_stopper.early_stop(val_loss): 
            break
        
    io.cprint(f'loss_train_lt\n{loss_train_lt}\nloss_val_lt\n{loss_val_lt}\n\n')
    io.cprint(f'acc_train_lt\n{acc_train_lt}\acc_val_lt\n{acc_val_lt}\n\n')
    io.cprint(f'f1_train_lt\n{f1_train_lt}\f1_val_lt\n{f1_val_lt}\n\n')
    
    loss, acc, f1 = test_dist(test_partition, device, criterion, epoch, io)
    
    return loss, acc, f1


def train_simple(args, io):
    list_dir = np.asarray(os.listdir(os.path.join(data_dir, f'labels_matching_{args.data.subset}')))
    size = len(list_dir)
    train_idx = np.random.choice(size, round(args.data.prop_test*size), replace=False)
    
    sel_idx = np.arange(0, args.data.prop_val*len(train_idx))
    test_idx = np.delete(np.arange(size), train_idx)
    val_idx = train_idx[sel_idx]
    train_idx = np.delete(train_idx, sel_idx)
    
    train_partition =  list_dir[train_idx]
    val_partition = list_dir[val_idx]
    test_partition = list_dir[test_idx]
    
    train(args, io, train_partition, val_partition, test_partition)


def train_kfold(args, io):
    list_dir = np.asarray(os.listdir(os.path.join(data_dir, f'labels_matching_{args.data.subset}')))
    size = len(list_dir)
    train_idx = np.random.choice(size, round(args.data.prop_test*size), replace=False)
    test_idx = np.delete(np.arange(size), train_idx)
    
    test_partition = list_dir[test_idx]
    split_size = len(train_idx) // args.kfolds
    
    loss_same, loss_half, loss_cmp = [], [], []
    acc_same, acc_half, acc_cmp = [], [], []
    f1_same, f1_half, f1_cmp = [], [], []
    for i in range(args.kfolds):
        sel_idx = np.arange(i*split_size, (i+1)*split_size)
        val_idx = train_idx[sel_idx]
        train_idx_ = np.delete(train_idx, sel_idx)
        
        val_partition = list_dir[val_idx]
        train_partition = list_dir[train_idx_]
        
        loss, acc, f1 = train(args, io, train_partition, val_partition, test_partition)
        loss_same.append(loss[0]), loss_half.append(loss[1]), loss_cmp.append(loss[2])
        acc_same.append(acc[0]), acc_half.append(acc[1]), acc_cmp.append(acc[2])
        f1_same.append(f1[0]), f1_half.append(f1[1]), f1_cmp.append(f1[2])
    
    io.cprint(f'\n\n\nFOLD SAME\nLoss {np.mean(loss_same)} \n{loss_same}\nAcc {np.mean(acc_same)} \n{acc_same}\nF1 {np.mean(f1_same)} \n{f1_same}')
    io.cprint(f'\n\n\nFOLD HALF\nLoss {np.mean(loss_half)} \n{loss_half}\nAcc {np.mean(acc_half)} \n{acc_half}\nF1 {np.mean(f1_half)} \n{f1_half}')
    io.cprint(f'\n\n\nFOLD CMP\nLoss {np.mean(loss_cmp)} \n{loss_cmp}\nAcc {np.mean(acc_cmp)} \n{acc_cmp}\nF1 {np.mean(f1_cmp)} \n{f1_cmp}')


def test(args, io):
    device = torch.device("cuda" if args.model.cuda else "cpu")
    test_partition = np.asarray(os.listdir(os.path.join(data_dir, f'labels_matching_{args.data.subset}')))
    np.random.shuffle(test_partition)
    
    if args.model.loss == 'focal_loss':
        criterion = sigmoid_focal_loss
    elif args.model.loss == 'bce_loss':
        lst = torch.FloatTensor([cfg.data.prop_pn]).to(device, dtype=torch.float)
        criterion = nn.BCEWithLogitsLoss(pos_weight=lst)
    else:
        raise Exception('Loss criterion not implemented')
    
    loss, acc, f1 = test_dist(test_partition, device, criterion, -1, io)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Feature extractor')
    parser.add_argument('--cfg_file', default='config/mmn/mmn.yml', type=str)
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--kfold', type=int,  default=-1,
                        help='number of folds')
    parser.add_argument('--model_path', type=str, default=None, metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--model_path_extractor', type=str, default='checkpoints/ours_2048_modelnet40/train/best_model_0.pth', 
                        metavar='N', help='Pretrained model path extractor')
    
    args = parser.parse_args()
    print(args)

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg_file)
    
    if args.model_path is None:
        cfg.model.path = None
    else:
        cfg.model.path = os.path.join(BASE_DIR, args.model_path)
    print(cfg)

    exp_name = f'{cfg.exp.name}_{cfg.data.name}'
    cfg.exp.name = exp_name
    checkpoint_init(exp_name)

    path = f"/{'test' if args.eval else 'train'}/run.log"
    io = IOStream('checkpoints/' + exp_name + path)
    io.cprint(str(args))

    cfg.kfold = args.kfold
    cfg.model.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(cfg.exp.seed)
    if cfg.model.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(cfg.exp.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval and args.kfold == -1:
        train_simple(cfg, io)
    elif not args.eval and args.kfold > -1:
        train_kfold(cfg, io)
    elif args.eval and args.kfold == -1:
        test(cfg, io)
    else:
        raise Exception("Not valid parameters")