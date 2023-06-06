import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
import numpy as np
from torch.utils.data import DataLoader
import sklearn.metrics as metrics


from src.config import get_cfg_defaults
from src.dataloader import ModelNet40
from src.model.extractor import PointNet, DGCNN, DGCNN_ORIG, VN_DGCNN, VN_DGCNN_ORIG, cal_loss
from src.utils.checkpoints import IOStream, SaveBestModel, checkpoint_init


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(BASE_DIR, 'data/extractor')



def get_model(model_args):
    if model_args.extractor == 'pointnet':
        model = PointNet(model_args)
    elif model_args.extractor == 'dgcnn':
        model = DGCNN(model_args)
    elif model_args.extractor == 'dgcnn_orig':
        model = DGCNN_ORIG(model_args)
    elif model_args.extractor == 'vn_dgcnn':
        model = VN_DGCNN(model_args)
    elif model_args.extractor == 'vn_dgcnn_orig':
        model = VN_DGCNN_ORIG(model_args)
    else:
        raise Exception("Not implemented")
    return model


def train_step(model, train_loader, device, opt, criterion, epoch, io):
    train_loss = 0.0
    count = 0.0
    model.train()
    train_pred = []
    train_true = []
    for data, label in train_loader:
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        opt.zero_grad()
        logits = model(data)
        loss = criterion(logits, label)
        loss.backward()
        opt.step()
        preds = logits.max(dim=1)[1]
        count += batch_size
        train_loss += loss.item() * batch_size
        train_true.append(label.cpu().numpy())
        train_pred.append(preds.detach().cpu().numpy())
    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    train_loss = train_loss*1.0/count
    train_acc = metrics.accuracy_score(train_true, train_pred)
    train_acc_avg = metrics.balanced_accuracy_score(train_true, train_pred)
    outstr = f'Train {epoch}, loss: {train_loss:.6f}, train acc: {train_acc:.6f}, train avg acc: {train_acc_avg:.6f}'
    io.cprint(outstr)
    return train_loss, train_acc, train_acc_avg
    
    
def test_step(model, test_loader, device, criterion, epoch, io, log_loss=True):
    test_loss = 0.0
    count = 0.0
    model.eval()
    test_pred = []
    test_true = []
    for data, label in test_loader:
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits = model(data)
        if log_loss:
            loss = criterion(logits, label)
            test_loss += loss.item() * batch_size
        preds = logits.max(dim=1)[1]
        count += batch_size
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_loss = test_loss*1.0/count
    test_acc = metrics.accuracy_score(test_true, test_pred)
    test_acc_avg = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = f'Test {epoch}, loss: {test_loss:.6f}, test acc: {test_acc:.6f}, test avg acc: {test_acc_avg:.6f}'
    io.cprint(outstr)
    return test_loss, test_acc, test_acc_avg



def train(args, io):
    train_loader = DataLoader(ModelNet40(partition='train', data_dir=data_dir, num_points=args.data.num_points), num_workers=8,
                              batch_size=args.exp.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', data_dir=data_dir, num_points=args.data.num_points), num_workers=8,
                             batch_size=args.exp.test_batch_size, shuffle=True, drop_last=False)
    device = torch.device("cuda" if args.cuda else "cpu")

    # if args.model.extractor == 'pointnet':
    #     model = PointNet(args.model).to(device)
    # elif args.model.extractor == 'dgcnn':
    #     model = DGCNN(args.model).to(device)
    # elif args.model.extractor == 'dgcnn_orig':
    #     model = DGCNN_ORIG(args.model).to(device)
    # else:
    #     raise Exception("Not implemented")
    
    model = get_model(args.model).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    io.cprint(str(model))
    io.cprint(f'NUMBER OF TRAINABLE PARAMS: {total_params}')

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    
    if args.model_path:
        best_model_cp = torch.load(args.model_path)
        best_model_epoch = best_model_cp['epoch']
        model.load_state_dict(best_model_cp['model_state_dict'])
        print(f"Best model loaded, its was saved at {best_model_epoch} epochs\n")
    
    if cfg.opt.name == 'SGD':
        opt = optim.SGD(model.parameters(), lr=cfg.opt.lr, momentum=args.opt.gamma, weight_decay=cfg.opt.weight_decay)
    elif cfg.opt.name == 'Adam':
        opt = optim.Adam(model.parameters(), lr=cfg.opt.lr, weight_decay=cfg.opt.weight_decay)
    elif cfg.opt.name == 'Adadelta':
        opt = optim.Adadelta(model.parameters(), lr=cfg.opt.lr, weight_decay=cfg.opt.weight_decay)
    else:
        raise Exception('Optmizer not implemented')
    
    if cfg.opt.scheduler == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(opt, patience=1, verbose=True, factor=cfg.opt.gamma)
    elif cfg.opt.scheduler == 'StepLR':
        scheduler = StepLR(opt, step_size=1, gamma=cfg.opt.gamma, verbose=True)
    elif cfg.opt.scheduler == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(opt, args.exp.epochs, eta_min=args.opt.lr_min, verbose=True)
    io.cprint(str(opt))
    
    if cfg.model.loss == 'cal_loss':
        criterion = cal_loss
    elif cfg.model.loss == 'cross_entropy_loss':
        criterion = nn.CrossEntropyLoss()
    else:
        raise Exception('Loss criterion not implemented')

    save_best_model = SaveBestModel(dir=BASE_DIR, name=cfg.exp.name, asc=True)
    for epoch in range(args.exp.epochs):
        scheduler.step()
        
        train_loss, train_acc, train_acc_avg = train_step(model, train_loader, device, opt, criterion, epoch, io)
        val_loss, val_acc, val_acc_avg = test_step(model, test_loader, device, criterion, epoch, io)
        
        save_best_model(val_acc_avg, epoch, model, opt, criterion)


def test(args, io):
    test_loader = DataLoader(ModelNet40(partition='test', data_dir=data_dir, num_points=args.data.num_points),
                             batch_size=args.exp.test_batch_size, shuffle=True, drop_last=False)
    device = torch.device("cuda" if args.cuda else "cpu")

    # if args.model.extractor == 'pointnet':
    #     model = PointNet(args.model).to(device)
    # elif args.model.extractor == 'dgcnn':
    #     model = DGCNN(args.model).to(device)
    # elif args.model.extractor == 'dgcnn_orig':
    #     model = DGCNN_ORIG(args.model).to(device)
    # else:
    #     raise Exception("Not implemented")
    
    model = get_model(args.model).to(device)
    model = nn.DataParallel(model)
    best_model_cp = torch.load(args.model_path)
    best_model_epoch = best_model_cp['epoch']
    model.load_state_dict(best_model_cp['model_state_dict'])
    print(f"Best model loaded, its was saved at {best_model_epoch} epochs\n")
    
    model = model.eval()
    test_loss, test_acc, test_acc_avg = test_step(model, test_loader, device, None, 'accum', io, log_loss=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Feature extractor')
    parser.add_argument('--cfg_file', default='config/train_extractor.yml', type=str)
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--model_path', type=str, default=None, metavar='N',
                        help='Pretrained model path')
    
    args = parser.parse_args()
    print(args)

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg_file)
    
    if args.model_path is None:
        cfg.model_path = None
    else:
        cfg.model_path = os.path.join(BASE_DIR, args.model_path)
    
    print(cfg)

    checkpoint_init(cfg.exp.name)

    path = f"/{'test' if args.eval else 'train'}/run.log"
    io = IOStream('checkpoints/' + cfg.exp.name + path)
    io.cprint(str(args))

    cfg.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(cfg.exp.seed)
    if cfg.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(cfg.exp.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(cfg, io)
    else:
        test(cfg, io)