from __future__ import print_function
import argparse
import numpy as np

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torchvision.ops import sigmoid_focal_loss

from torchmetrics.classification import BinaryF1Score, BinaryConfusionMatrix, BinaryAUROC
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import KFold

from src_pt.dataloader import BreakingBad, ModelNet40
from src_pt.model import SiameseNet

from src_pt.config import get_cfg_defaults



class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            print(f'Patience {self.counter}')
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class SaveBestModel:
    def __init__(
        self, dir, best_valid_loss=float('inf'), id=0, idx=0
    ):
        self.dir = dir
        self.best_valid_loss = best_valid_loss
        self.id = id
        self.idx = idx
        
    def __call__(
        self, current_valid_loss, 
        epoch, model, optimizer, criterion
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, f'{self.dir}/outputs/best_model_{self.id}_{self.idx}.pth')


def train(cfg, model, criterion, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (images_1, images_2, targets) in enumerate(train_loader):
        images_1, images_2, targets = images_1.to(device, dtype=torch.float), images_2.to(device, dtype=torch.float), targets.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(images_1, images_2)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if batch_idx % cfg.exp.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(images_1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            

def test(cfg, model, criterion, metric, device, test_loader, name='train'):
    model.eval()
    
    test_loss = 0
    f1_score = 0
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    accuracy = 0
    
    bcm = BinaryConfusionMatrix(normalize='all').to(device)

    with torch.no_grad():
        for (images_1, images_2, targets) in test_loader:
            images_1, images_2, targets = images_1.to(device, dtype=torch.float), images_2.to(device, dtype=torch.float), targets.to(device).unsqueeze(1)
            outputs = model(images_1, images_2)
            test_loss += criterion(outputs, targets).sum().item()  # sum up batch loss
            pred = torch.where(outputs > 0.5, 1, 0)  # get the index of the max log-probability
            tn, fp, fn, tp = bcm(targets, pred).ravel()
            true_neg += tn.item()
            false_pos += fp.item()
            false_neg += fn.item()
            true_pos += tp.item()
            accuracy += pred.eq(targets.squeeze(0).view_as(pred.squeeze(0))).sum().item()
            f1_score += metric(targets.squeeze(0), pred.squeeze(0)).item()

    test_loss /= len(test_loader)
    f1_score /= len(test_loader)
    false_pos /= len(test_loader)
    false_neg /= len(test_loader)
    true_pos /= len(test_loader)
    true_neg /= len(test_loader)
    accuracy /= len(test_loader.dataset)

    print('{} set: Average loss: {:.4f}, F1 score: {} ({:.0f}%), Accuracy: {} ({:.0f}%)'.format(
        name, test_loss, np.round(f1_score, 4), 100 * f1_score, np.round(accuracy, 4), 100*accuracy))
    print(f'Value of true negative: {np.round(true_neg, 2)} items')
    print(f'Value of false positive: {np.round(false_pos, 2)} items')
    print(f'Value of false negative: {np.round(false_neg, 2)} items')
    print(f'Value of true positive: {np.round(true_pos, 2)} items')
    
    return test_loss, accuracy, f1_score


def main():
    parser = argparse.ArgumentParser(description='Siamese Breaking Bad')
    parser.add_argument('--cfg_file', default='config/train.yml', type=str)
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    
    args = parser.parse_args()
    print(args)
    
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg_file)
    print(cfg)
    
    writer = SummaryWriter(f'runs/experiment_{cfg.exp.id}')
    layout = {
        "Summary": {
            "loss": ["Multiline", ["loss/train", "loss/val"]],
            "accuracy": ["Multiline", ["accuracy/train", "accuracy/val"]],
            "f1": ["Multiline", ["f1/train", "f1/val"]]
        },
    }
    writer.add_custom_scalars(layout)
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(cfg.exp.seed)

    if use_cuda:
        device = torch.device("cuda")
        torch.cuda.empty_cache()
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': cfg.exp.batch_size, 'drop_last': True}
    val_kwargs = {'batch_size': cfg.exp.test_batch_size, 'drop_last': True}
    test_kwargs = {'batch_size': cfg.exp.test_batch_size, 'drop_last': True}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        val_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # READ DATASET
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    list_dir = np.asarray(os.listdir(os.path.join(DATA_DIR, 'labels')))
    np.random.shuffle(list_dir)
    size = len(list_dir)
    train_idx = np.random.choice(size, round(cfg.data.prop_test*size), replace=False)
    # [train_idx, val_idx] = np.split(train_idx, [round(len(train_idx)*cfg.data.prop_val)])
    test_idx = np.delete(np.arange(size), train_idx)
    
    # train_partition = list_dir[train_idx]
    # val_partition = list_dir[val_idx]
    test_partition = list_dir[test_idx]
    
    # K FOLD
    k = 5
    fold_acc = []
    fold_acc_50 = []
    fold_acc_all = []
    split_size = len(train_idx) // k 
    train_prop = cfg.data.prop_pn
    
    for i in range(k):
        sel_idx = np.arange(i*split_size, (i+1)*split_size)
        val_idx = train_idx[sel_idx]
        train_idx_ = np.delete(train_idx, sel_idx)
        val_partition = list_dir[val_idx]
        train_partition = list_dir[train_idx_]
        
        cfg.data.prop_pn = train_prop
        train_dataset = BreakingBad(partition=train_partition, mode='train', args=cfg.data, data_dir=DATA_DIR)
        train_size = train_dataset.get_num_points()
        val_dataset = BreakingBad(partition=val_partition, mode='test', args=cfg.data, data_dir=DATA_DIR)
        val_size = val_dataset.get_num_points()
        # For test keep the same proportion
        
        assert(train_size == val_size)
        cfg.model.emb_dims = 1024 # train_size
        print(cfg.model.emb_dims)
        
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        val_loader = torch.utils.data.DataLoader(val_dataset, **val_kwargs)

        model = SiameseNet(cfg.model).to(device)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(model)
        print(f'NUMBER OF TRAINABLE PARAMS: {total_params}')
        
        if cfg.opt.name == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=cfg.opt.lr, momentum=0.9, weight_decay=cfg.opt.weight_decay)
        elif cfg.opt.name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=cfg.opt.lr, weight_decay=cfg.opt.weight_decay)
        elif cfg.opt.name == 'Adadelta':
            optimizer = optim.Adadelta(model.parameters(), lr=cfg.opt.lr, weight_decay=cfg.opt.weight_decay)
        else:
            raise Exception('Optmizer not implemented')
        
        
        if cfg.opt.scheduler == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, patience=1, verbose=True)
        elif cfg.opt.scheduler == 'StepLR':
            scheduler = StepLR(optimizer, step_size=1, gamma=cfg.opt.gamma, verbose=True)
        print(optimizer)
        
        if cfg.model.loss == 'focal_loss':
            criterion = sigmoid_focal_loss
        elif cfg.model.loss == 'bce_loss':
            lst = torch.FloatTensor([cfg.data.prop_pn]).to(device, dtype=torch.float)
            criterion = nn.BCEWithLogitsLoss(pos_weight=lst)
        else:
            raise Exception('Loss criterion not implemented')
        
        
        global_epoch = 0
        loss_train_lt = []
        loss_val_lt = []
        f1_train_lt = []
        f1_val_lt = []
        acc_train_lt = []
        acc_val_lt = []
        early_stopper = EarlyStopper(patience=5)
        # metric = BinaryF1Score().to(device)
        metric = BinaryAUROC().to(device)
        save_best_model = SaveBestModel(dir=BASE_DIR, id=cfg.exp.id, idx=i)
        
        # training process
        for epoch in range(1, cfg.exp.epochs + 1):
            cnt = False
            for batch_idx, (images_1, images_2, targets) in enumerate(train_loader):
                model.train()
                images_1, images_2, targets = images_1.to(device, dtype=torch.float), images_2.to(device, dtype=torch.float), targets.to(device).unsqueeze(1)
                optimizer.zero_grad()
                outputs = model(images_1, images_2)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                if batch_idx % cfg.exp.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_epoch, batch_idx * len(images_1), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
            
                    print()
                    loss_train, acc_train, f1_train = test(cfg, model, criterion, metric, device, train_loader, 'train')
                    loss_val, acc_val, f1_val = test(cfg, model, criterion, metric, device, val_loader, 'val')
                    print()
                    writer.add_scalar('loss/train', loss_train, global_epoch)
                    writer.add_scalar('accuracy/train', acc_train, global_epoch)
                    writer.add_scalar('f1/train', f1_train, global_epoch)
                    writer.add_scalar('loss/val', loss_val, global_epoch)
                    writer.add_scalar('accuracy/val', acc_val, global_epoch)
                    writer.add_scalar('f1/val', f1_val, global_epoch)
                    
                    loss_train_lt.append(loss_train)
                    loss_val_lt.append(loss_val)
                    acc_train_lt.append(acc_train)
                    acc_val_lt.append(acc_val)
                    f1_train_lt.append(f1_train)
                    f1_val_lt.append(f1_val)
                    
                    save_best_model(loss_val, global_epoch, model, optimizer, criterion)
                    global_epoch += 1
                    
                    if early_stopper.early_stop(loss_val): 
                        cnt = True
                        break
                    
                    if cfg.opt.scheduler == 'ReduceLROnPlateau':
                        scheduler.step(loss_val)
                    elif cfg.opt.scheduler == 'StepLR':
                        scheduler.step()
            
            if cnt:
                break
            
            # train(cfg, model, criterion, device, train_loader, optimizer, epoch)

        print('loss_train_lt\n', loss_train_lt)
        print('loss_val_lt\n', loss_val_lt)
        print('acc_train_lt\n', acc_train_lt)
        print('acc_val_lt\n', acc_val_lt)
        print('f1_train_lt\n', f1_train_lt)
        print('f1_val_lt\n', f1_val_lt)
        
        
        # load the best model checkpoint
        model = SiameseNet(cfg.model).to(device)
        best_model_cp = torch.load(f'outputs/best_model_{cfg.exp.id}_{i}.pth')
        best_model_epoch = best_model_cp['epoch']
        model.load_state_dict(best_model_cp['model_state_dict'])
        print(f"Best model loaded, its was saved at {best_model_epoch} epochs\n")
        
        
        print('\nWith same distribution\n')
        test_dataset = BreakingBad(partition=test_partition, mode='test', args=cfg.data, data_dir=DATA_DIR)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        loss_test, acc_test, f1_test = test(cfg, model, criterion, metric, device, test_loader, 'test')
        print('\n\n')
        fold_acc.append(acc_test)
        
        print('\nWith 50 50 distribution\n')
        cfg.data.prop_pn = 0.5
        test_dataset = BreakingBad(partition=test_partition, mode='test', args=cfg.data, data_dir=DATA_DIR)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        loss_test, acc_test, f1_test = test(cfg, model, criterion, metric, device, test_loader, 'test')
        print('\n\n')
        fold_acc_50.append(acc_test)
        
        print('\nWith complete all to all search\n')
        cfg.data.prop_pn = 0.01
        test_dataset = BreakingBad(partition=test_partition, mode='test', args=cfg.data, data_dir=DATA_DIR)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        loss_test, acc_test, f1_test = test(cfg, model, criterion, metric, device, test_loader, 'test')
        print('\n\n')
        fold_acc_all.append(acc_test)
        
        # if args.save_model:
        #     torch.save(model.state_dict(), "siamese_network.pt")
    
    print(f'\n\nFOLD ACCURACY ORIGINAL {np.mean(fold_acc)}\n{fold_acc}\n\n')
    print(f'\n\nFOLD ACCURACY 50 50 {np.mean(fold_acc_50)}\n{fold_acc_50}\n\n')
    print(f'\n\nFOLD ACCURACY ALL {np.mean(fold_acc_all)}\n{fold_acc_all}\n\n')


if __name__ == '__main__':
    main()