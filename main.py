import torch.utils
import torch.utils.data
import numpy as np

import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18
from torch.optim.lr_scheduler import CosineAnnealingLR

import utils
import train
import os
import model as md
def main():
    args = utils.make_arg().parse_args()
    
    train_transform = transforms.Compose(
        [#transforms.Resize((32, 32)),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
        ])
    
    test_transform = transforms.Compose(
        [#transforms.Resize((32, 32)),  
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]) 
        ])
    
    if args.mode == 'train':
        trainset = utils.CutMix_CIFAR(args.dataset, load_type='train', transform=train_transform)
        validset = utils.CutMix_CIFAR(args.dataset, load_type='valid', transform=test_transform)
        
        train_loader = torch.utils.data.DataLoader(
            trainset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.workers)
        
        valid_loader = torch.utils.data.DataLoader(
            validset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers)
        
        loaders = (train_loader, valid_loader)
        
        model = None
        if args.model == 'ViT':
            # model = md.ViT()
            model = md.SimpleViT(image_size=32, patch_size=4, num_classes=100, dim=256, depth=4, heads=12, mlp_dim=256)
            weight_name = 'ViT_weights.pth'
        elif args.model == 'cnn':
            model = resnet18(num_classes=100)
            weight_name = 'cnn_weights.pth'
        
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=args.lr, 
            momentum=args.momentum,
            weight_decay=args.weight_decay)
        
        criterion = utils.CutMixCrossEntropyLoss().to(args.device)
        scheduler = CosineAnnealingLR(
            optimizer, 
            int(args.epochs), 
            eta_min=0)
        
        train.train(model, loaders, optimizer, criterion, scheduler, args)
        torch.save(model.state_dict(), os.path.join(args.out_dir, weight_name))
        
    elif args.mode == 'test':
        testset = utils.CutMix_CIFAR(args.dataset, load_type='test', transform=test_transform)
        test_loader = torch.utils.data.DataLoader(
            testset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.workers)
        
        model = None
        if args.model == 'ViT':
            model = md.SimpleViT(image_size=32, patch_size=4, num_classes=100, dim=256, depth=16, heads=8, mlp_dim=256)
            weights = torch.load(args.weight)
            model.load_state_dict(weights)
        elif args.model == 'cnn':
            pass
        
        acc = train.test(model, test_loader, args)
        print(f'Acc is {acc*100: .2f}%')
        
if __name__ == '__main__':
    main()
        