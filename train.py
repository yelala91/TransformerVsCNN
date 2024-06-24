import torch
import torch.nn as nn  
from torch.utils.tensorboard import SummaryWriter
from os.path import join
from tqdm import tqdm

def accuracy(output, target):
    correct = 0
    with torch.no_grad():
        _, pred = torch.max(output.data, 1)
        correct += (pred == target).sum().item()
        acc = correct/target.size(0)
    
        return acc
    
def train_one_step(model, loaders, optimizer, curr_epoch,
    criterion, scheduler, args, writer=None):

    train_loader, valid_loader = loaders
    print(f'Current epoch:{curr_epoch+1} ')
    total_loss = 0.0; pre_step = len(train_loader)*curr_epoch
    curr_step = pre_step + 1
    for imgs, labels in tqdm(train_loader):
        if type(imgs) is not torch.cuda.FloatTensor:
            imgs = imgs.float()
        imgs = imgs.to(args.device)
        labels = labels.to(args.device)
        output = model(imgs)
        
        loss = criterion(output, labels)
        total_loss += loss.item()
        
        if writer is not None:
            writer.add_scalar('Loss', loss.item(), curr_step)
        curr_step += 1
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    scheduler.step()
    print(f'Epoch {curr_epoch+1}/{args.epochs}, Loss: {total_loss/len(train_loader)}')

    if (curr_epoch+1) % args.val_freq == 0:
        with torch.no_grad():
            print('start valid...')
            correct = 0; total = 0
            for imgs, labels in tqdm(valid_loader):
                if type(imgs) is not torch.cuda.FloatTensor:
                    imgs = imgs.float()
                imgs = imgs.to(args.device)
                labels = labels.to(args.device)
                output = model(imgs)
                
                _, pred = torch.max(output.data, 1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        
        acc = correct/total
        if writer is not None:
            writer.add_scalar('Acc', acc)
        
        print(f'Acc: {acc*100: .2f}%')  
    
def train(model, loaders, optimizer, criterion, scheduler, args):
    
    writer = SummaryWriter(join(args.out_dir, 'run'))
    model = model.to(args.device)
    model.train()
    
    for epoch in range(args.epochs):
        train_one_step(model, loaders, optimizer, epoch, criterion, 
            scheduler, args, writer)

def test(model, test_loader, args):
    model = model.to(args.device)
    
    print('testing......')
    correct = 0
    total = 0
    for imgs, lbs in tqdm(test_loader):
        if type(imgs) is not torch.cuda.FloatTensor:
            imgs = imgs.float()
        imgs = imgs.to(args.device)
        lbs = lbs.to(args.device)
        
        with torch.no_grad():
            output = model(imgs)
            total += lbs.size(0)
            
            _, pred = torch.max(output.data, 1)
            correct += (pred == lbs).sum().item()
    
    acc = correct/total
    # print(f'The Accuracy in test set is {acc*100: .2f}%')
    return acc
    
# def LCP_test(model, train_loader, optimizer, criterion, scheduler, args):
#     writer = SummaryWriter(join(args.out_dir, 'run'))
#     model = model.to(args.device)
#     model.train()
    
#     for epoch in args.epochs:
#         train_one_step(model, train_loader, optimizer, epoch, criterion,
#             scheduler, args, w)
        
    