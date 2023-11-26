import torch
import torch.nn as nn


from src.model.extractor import PointNet, PointNet2, DGCNN, DGCNN_ORIG, VN_PointNet, VN_DGCNN, VN_DGCNN_ORIG



def load_model(args, model):
    best_model_cp = torch.load(args.path)
    best_model_epoch = best_model_cp['epoch']
    model.load_state_dict(best_model_cp['model_state_dict'])
    print(f"Best model loaded, its was saved at {best_model_epoch} epochs\n")
    return model


def get_model(args):
    if args.extractor == 'pointnet':
        model = PointNet(args)
    elif args.extractor == 'pointnet2':
        model = PointNet2(args)
    elif args.extractor == 'ours':
        model = DGCNN(args)
    elif args.extractor == 'dgcnn':
        model = DGCNN_ORIG(args)
    elif args.extractor == 'vn_pointnet':
        model = VN_PointNet(args)
    elif args.extractor == 'vn_ours':
        model = VN_DGCNN(args)
    elif args.extractor == 'vn_dgcnn':
        model = VN_DGCNN_ORIG(args)
    else:
        raise Exception("Not implemented")
    
    device = torch.device("cuda" if args.cuda else "cpu")
    model = model.to(device)
    model = nn.DataParallel(model)
    if args.path is not None:
        model = load_model(args, model)

    return model