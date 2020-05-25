import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import png
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from colormap.colors import Color, hex2rgb
from sklearn.metrics import average_precision_score as ap_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

from dataset import FacadeDataset, process_img
from model_part import SingleConv, Up, Down
from torch.utils.tensorboard import SummaryWriter
import argparse
from utils import AverageMeter, mkdir
from os.path import join, split
import collections

N_CLASS=5

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.n_class = N_CLASS
        self.conv_in = SingleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 128)
        self.up1 = Up(256, 64)
        self.up2 = Up(128, 64)
        self.conv_out = nn.Conv2d(64, self.n_class, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv_in(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.up1(x3, x2)
        x5 = self.up2(x4, x1)
        x = self.conv_out(x5)
        # x = self.relu(x)
        return x

def m_to_s(m_dict):
    '''
    change multi-GPU model checkpoint to single-GPU
    '''
    s_dict = collections.OrderedDict()
    for key, value in m_dict.items():
        if 'module' in key:
            new_key = key[7:]
            s_dict[new_key] = value
        else:
            s_dict[key] = value

    return s_dict

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
    parser.add_argument('--test', default=False, type=bool)

    args = parser.parse_args()
    return args

def val_loss(net, val_loader_loss, criterion, device):
    cnt = 0
    loss_total = 0
    with torch.no_grad():
        net = net.eval()
        for images, labels in tqdm(val_loader_loss):
            images = images.to(device)
            labels = labels.to(device)
            output = net(images)
            loss = criterion(output, labels)
            cnt += 1
            loss_total += loss.item()
    
    return loss_total/cnt

def train(model, train_loader, val_loader, val_loader_loss, optimizer, criterion, writer, epoch):
    args = parse_args() 
    checkpoint_dir = args.cp_dir
    print_freq = args.freq
    for k, v in args.__dict__.items():
        print(k, ':', v)
    
    # criterion.cuda() #TODO

    best_prec1 = 0
    start_epoch = 0
    global_step = 0
    
    for epoch in range(start_epoch, args.epochs):
        losses = AverageMeter()
        # scores = AverageMeter()
        model.train()
        for i, (input, target) in enumerate(train_loader):
            optimizer.zero_grad()
            input = input.cuda().float()
            target = target.cuda().long()
            output = model(input)
            loss = criterion(output, target)

            writer.add_scalar('Loss/train', loss.item(), global_step)
            losses.update(loss.item(), input.size(0))
            # scores.update(eval_score(output, target), input.size(0))
            # writer.add_scalar('accuracy/train', scores.avg, global_step)

            loss.backward()
            optimizer.step()    
            global_step += 1
            if i % print_freq==0:
                print('Epoch: [{0}][{index}/{length}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                    epoch, index=i, length=len(train_loader), loss=losses))

            
        print('Epoch: [{0}]\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch, loss=losses))

        # prec1 = validate(model, val_loader, criterion, eval_score)
        prec1 = cal_AP(val_loader, model, criterion, torch.device('cpu')) #TODO
        loss_val = val_loss(model, val_loader_loss, criterion, torch.device('cpu')) #TODO
        print('mAP: %.4f'%prec1)
        print('val loss: %.4f'%loss_val)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        writer.add_scalar('accuracy/val', prec1, epoch+1)
        writer.add_scalar('loss/val', loss_val, epoch+1)

        checkpoint_path = join(checkpoint_dir,'checkpoint_{}.pth.tar'.format(epoch))
        cp_state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'prec': prec1
        }
        torch.save(cp_state, checkpoint_path)
        if is_best:
            torch.save(cp_state, join(checkpoint_dir, 'best_checkpoint.pth.tar'))

    writer.close()

def save_label(label, path):
    '''
    Function for ploting labels.
    '''
    colormap = [
        '#000000',
        '#0080FF',
        '#80FF80',
        '#FF8000',
        '#FF0000',
    ]
    assert(np.max(label)<len(colormap))
    colors = [hex2rgb(color, normalise=False) for color in colormap]
    w = png.Writer(label.shape[1], label.shape[0], palette=colors, bitdepth=4)
    with open(path, 'wb') as f:
        w.write(f, label)

def cal_AP(testloader, net, criterion, device):
    '''
    Calculate Average Precision
    '''
    losses = 0.
    cnt = 0
    with torch.no_grad():
        net = net.eval()
        preds = [[] for _ in range(5)]
        heatmaps = [[] for _ in range(5)]
        for images, labels in tqdm(testloader):
            images = images.to(device)
            labels = labels.to(device)
            output = net(images)
            # loss = criterion(output, labels)
            output = output.cpu().numpy()
            for c in range(5):
                preds[c].append(output[:, c].reshape(-1))
                heatmaps[c].append(labels[:, c].cpu().numpy().reshape(-1))

        aps = []
        for c in range(5):
            preds[c] = np.concatenate(preds[c])
            heatmaps[c] = np.concatenate(heatmaps[c])
            if heatmaps[c].max() == 0:
                ap = float('nan')
            else:
                ap = ap_score(heatmaps[c], preds[c])
                aps.append(ap)
            print("AP = {}".format(ap))
    
    aps = np.array(aps)
    return aps.sum()/len(aps)

    # return None
def test_img(model, fn):
    img = process_img(fn)
    with torch.no_grad():
        model = model.eval()
        output = model(img)[0].cpu().numpy()
        c, h, w = output.shape
        assert(c == N_CLASS)
        y = np.zeros((h,w)).astype('uint8')
        # output = output[1:,:,:]
        y = np.argmax(output, axis=0)
        y = y.astype('uint8')
        # print(y)
        # # print(output)
        # for i in range(N_CLASS):
        #     mask = output[i]>0.5
        #     y[mask] = i
        # print(y, y.shape)
        # gt = labels.cpu().data.numpy().squeeze(0).astype('uint8')
        save_label(y, 'result.png')
        # save_label(gt, './{}/gt{}.png'.format(folder, cnt))
        # plt.imsave(
        #     './{}/x{}.png'.format(folder, cnt),
        #     ((images[0].cpu().data.numpy()+1)*128).astype(np.uint8).transpose(1,2,0))


def get_result(testloader, net, device, folder='output_train'):
    result = []
    cnt = 1
    with torch.no_grad():
        net = net.eval()
        cnt = 0
        for images, labels in tqdm(testloader):
            images = images.to(device)
            labels = labels.to(device)
            output = net(images)[0].cpu().numpy()
            c, h, w = output.shape
            assert(c == N_CLASS)
            y = np.zeros((h,w)).astype('uint8')
            # output = output[1:,:,:]
            y = np.argmax(output, axis=0)
            y = y.astype('uint8')
            # print(y)
            # # print(output)
            # for i in range(N_CLASS):
            #     mask = output[i]>0.5
            #     y[mask] = i
            # print(y, y.shape)
            gt = labels.cpu().data.numpy().squeeze(0).astype('uint8')
            save_label(y, './{}/y{}.png'.format(folder, cnt))
            save_label(gt, './{}/gt{}.png'.format(folder, cnt))
            plt.imsave(
                './{}/x{}.png'.format(folder, cnt),
                ((images[0].cpu().data.numpy()+1)*128).astype(np.uint8).transpose(1,2,0))

            cnt += 1

def main():
    args = parse_args()
    device = torch.device("cpu") #TODO
    net = Net()
    # net = torch.nn.DataParallel(net).cuda() #TODO
    # net = torch.nn.DataParallel(net) #TODO
    criterion = nn.CrossEntropyLoss()

    if not args.test:
        mkdir(args.cp_dir)
        train_data = FacadeDataset(flag='train', data_range=(0,800), onehot=False)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True)
        val_data = FacadeDataset(flag='train', data_range=(801,906), onehot=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)
        val_data_loss = FacadeDataset(flag='train', data_range=(801,906), onehot=False)
        val_loader_loss = DataLoader(val_data_loss, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)

        name = 'starter_net'
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        writer = SummaryWriter(comment='_'+args.log)
        print('\nStart training')
        train(net, train_loader, val_loader, val_loader_loss, optimizer, criterion, writer, args.epochs) 
    else:
        pass
    test_data = FacadeDataset(flag='test_dev', data_range=(0,114), onehot=False)
    test_loader = DataLoader(test_data, batch_size=1, num_workers=args.workers, shuffle=False, pin_memory=True)
    ap_data = FacadeDataset(flag='test_dev', data_range=(0,114), onehot=True)
    ap_loader = DataLoader(ap_data, batch_size=1, pin_memory=True, num_workers=args.workers)

    folder = args.log + '_test_result'
    mkdir(folder)

    fn = 'best_checkpoint.pth.tar'
    checkpoint = torch.load(join(args.cp_dir, fn))
    # print(type(checkpoint['state_dict']))
    net.load_state_dict(m_to_s(checkpoint['state_dict']))
    print("model loaded, mAP: %.4f"%checkpoint['prec'])
    test_img(net, 'p.jpg')
    # get_result(test_loader, net, device, folder=folder)

    # torch.save(net.state_dict(), './models/model_{}.pth'.format(name))

    # mAP = cal_AP(ap_loader, net, criterion, device)
    # print("test mAP: %.4f"%mAP)

if __name__ == "__main__":
    main()
