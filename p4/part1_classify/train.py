import torch
import os
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import argparse
from os.path import join, split
import shutil
from data_fashion import test_set, val_set, train_set, train_sampler, val_sampler
from model import Class_Net, Class_Net96
# import sys


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def mkdir(path):
	folder = os.path.exists(path)
	if not folder:                  
		os.makedirs(path)

def parse_args():
    parser = argparse.ArgumentParser(description='cs194 proj4')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--freq', default=10, type=int)
    parser.add_argument('--cp_dir', default='./')
    parser.add_argument('--log', default='./') 

    args = parser.parse_args()
    return args

def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    _, pred = output.max(1)
    pred = pred.view(1, -1)
    target = target.view(1, -1)
    correct = pred.eq(target)

    correct = correct.view(-1)
    score = correct.float().sum(0).mul(100.0 / correct.size(0))
    return score.item()

def validate(model, val_loader, criterion, eval_score):
    args = parse_args()
    print_freq = args.freq
    losses = AverageMeter()
    score = AverageMeter()

    model.eval()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda().float()
        target = target.cuda().long()
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)
        
        losses.update(loss.item(), input.size(0))
        score.update(eval_score(output, target), input.size(0))
    
    print('Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        'Score {top1.val:.3f} ({top1.avg:.3f})'.format(loss=losses, top1=score), flush=True)
    
    return score.avg


def train(model, train_loader, val_loader, optimizer, criterion, writer, epoch, eval_score):
    args = parse_args() 
    checkpoint_dir = args.cp_dir
    print_freq = args.freq
    for k, v in args.__dict__.items():
        print(k, ':', v)
    
    criterion.cuda()
    best_prec1 = 0
    start_epoch = 0
    global_step = 0
    
    for epoch in range(start_epoch, args.epochs):
        losses = AverageMeter()
        scores = AverageMeter()
        model.train()
        for i, (input, target) in enumerate(train_loader):
            optimizer.zero_grad()
            input = input.cuda().float()
            target = target.cuda().long()
            output = model(input)
            loss = criterion(output, target)

            writer.add_scalar('Loss/train', loss.item(), global_step)
            losses.update(loss.item(), input.size(0))
            scores.update(eval_score(output, target), input.size(0))
            writer.add_scalar('accuracy/train', scores.avg, global_step)

            loss.backward()
            optimizer.step()    
            global_step += 1

            
        print('Epoch: [{0}]\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Score {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, loss=losses, top1=scores))

        prec1 = validate(model, val_loader, criterion, eval_score)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        writer.add_scalar('accuracy/val', prec1, epoch+1)

        checkpoint_path = join(checkpoint_dir,'checkpoint_{}.pth.tar'.format(epoch))
        cp_state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'prec1': prec1,
        }
        torch.save(cp_state, checkpoint_path)
        if is_best:
            torch.save(cp_state, join(checkpoint_dir, 'best_checkpoint.pth.tar'))

    writer.close()

if __name__=='__main__':
    args = parse_args()
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, 
                                sampler=train_sampler, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, 
                                sampler=val_sampler, pin_memory=True)
    writer = SummaryWriter(comment= args.log)
    criterion = nn.CrossEntropyLoss()
    mkdir(args.cp_dir)

    # model = Class_Net()
    model = Class_Net()

    model = torch.nn.DataParallel(model).cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    train(model, train_loader, val_loader, optimizer, criterion, writer, args.epochs, accuracy) 

        


