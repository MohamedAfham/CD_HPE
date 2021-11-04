import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import yaml
import cv2
import json
import os
import time
import random
from PIL import Image
from shutil import copy
import argparse

import torch
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import optim
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR

from utils.inference import get_max_preds
from utils.loss import JointsMSELoss, JointsKLLoss
from models.stacked_hg import PoseNet
from models.simple_baseline import PoseResNet, get_pose_net
from datasets.slp_dataset import SLP, SLPWeak
from utils.evaluate import accuracy
from utils.avg_metrics import AverageMeter
from cyclegan_transform.cyclegan_transform import cyclegan_transform, get_cyclegan_opt
from  extreme_transform.extreme_transform import extreme_transform
import wandb

FILELIST_DIR = "filelists"

def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=200, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--device', type=str, default='cpu', help='device')

    # optimization
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,80', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--use_target_weight', action='store_true', help='use target_weight for loss computation')

    # dataset
    parser.add_argument('--model', type=str, default='stacked_hg', choices = ['stacked_hg', 'simple_baseline'])
    parser.add_argument('--sigma', type=int, default=2, help='sigma value for generating heatmaps')
    parser.add_argument('--heatmap', type=str, default='64,64', help='Size of the heatmap')
    parser.add_argument('--n_stack', type=int, default=8, help='Number of stacks in Hourglass model')
    parser.add_argument("--wandb_run", default=None, help="Name of the WandB run")
    
    # simple baseline
    parser.add_argument('--init_weights', action='store_true', help='initialize pretrained backbone weights')
    parser.add_argument('--pretrained', type=str, default='', help='path for pretrained backbone')
    parser.add_argument('--num_layers', type=int, default=18, choices=[18, 34, 50, 101, 152], help='number of layers in backbone')
    parser.add_argument('--num_deconv_layers', type=int, default=3, help='Deconv layers in simple baseline')
    parser.add_argument('--num_deconv_filters', type=list, default=[256, 256, 256], help='list of deconv filters')
    parser.add_argument('--num_deconv_kernels', type=list, default=[4, 4, 4], help='list of deconv kernels')
    parser.add_argument('--num_joints', type=int, default=14, help='number of joints to predict')
    parser.add_argument('--final_conv_kernel', type=int, default=1, help='number of conv kernels after decoder')
    parser.add_argument('--deconv_with_bias', type=bool, default=True, help='Biased used during deconvolution')
    
    # cosine annealing
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')

    # specify folder
    parser.add_argument('--model_path', type=str, default='', help='path to save model')

    opt = parser.parse_args()

    # set the path according to the environment
    if not opt.model_path:
        opt.model_path = './results'

    iterations = opt.lr_decay_epochs.split(',')
    opt.heatmap_size = list(map(int,opt.heatmap.split(',')))
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_lr_{}'.format(opt.model, opt.lr)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    if opt.adam:
        opt.model_name = '{}_useAdam'.format(opt.model_name)

    opt.save_folder = os.path.join(opt.model_path, opt.wandb_run)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
        
    #opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return opt

def main():

    args = parse_option()
    
    wandb.init(project='gpt3.123', name=args.wandb_run)

    transform = transforms.Compose([transforms.Resize((256, 256)),transforms.ToTensor()])
    #cover1_transform= transforms.Compose([transforms.ToTensor(), cyclegan_transform(cyclegan_opt= get_cyclegan_opt(name = 'InbedPose_CyleGAN_cover1')), cyclegan_transform(cyclegan_opt= get_cyclegan_opt(name = 'InbedPose_CyleGAN_cover1'))])
    #cover2_transform= transforms.Compose([transforms.ToTensor(), cyclegan_transform(cyclegan_opt= get_cyclegan_opt(name = 'InbedPose_CyleGAN_cover2')), cyclegan_transform(cyclegan_opt= get_cyclegan_opt(name = 'InbedPose_CyleGAN_cover2'))])
    cycaug_cover1_transform= transforms.Compose([transforms.ToTensor(), cyclegan_transform(cyclegan_opt= get_cyclegan_opt(name = 'InbedPose_CyleGAN_cover1'))])
    cycaug_cover2_transform= transforms.Compose([transforms.ToTensor(), cyclegan_transform(cyclegan_opt= get_cyclegan_opt(name = 'InbedPose_CyleGAN_cover2'))])

    extremeaug_cover1_transform= transforms.Compose([transforms.ToTensor(), cyclegan_transform(cyclegan_opt= get_cyclegan_opt(name = 'InbedPose_CyleGAN_cover1')), extreme_transform()])
    extremeaug_cover2_transform= transforms.Compose([transforms.ToTensor(), cyclegan_transform(cyclegan_opt= get_cyclegan_opt(name = 'InbedPose_CyleGAN_cover2')), extreme_transform()])
    
    train_uncover_file = os.path.join(FILELIST_DIR, 'train_uncover.json')
    train_cover1_file = os.path.join(FILELIST_DIR, 'train_cover1.json')
    train_cover2_file = os.path.join(FILELIST_DIR, 'train_cover2.json')
    val_cover1_file = os.path.join(FILELIST_DIR, 'valid_cover1.json')
    val_cover2_file = os.path.join(FILELIST_DIR, 'valid_cover2.json')

    train_uncover_loader = DataLoader(SLP(train_uncover_file, (cycaug_cover1_transform, cycaug_cover2_transform, extremeaug_cover1_transform, extremeaug_cover2_transform, transform), args, isTrain = True), batch_size=args.batch_size, shuffle=True)
    train_cover1_loader = DataLoader(SLPWeak(train_cover1_file, transform), batch_size=args.batch_size, shuffle=True)
    train_cover1_loader = DataLoader(SLPWeak(train_cover2_file, transform), batch_size=args.batch_size, shuffle=True)

    val_cover1_loader = DataLoader(SLP(val_cover1_file, transform, args, isTrain = False), batch_size=args.batch_size, shuffle=True)
    val_cover2_loader = DataLoader(SLP(val_cover2_file, transform, args, isTrain = False), batch_size=args.batch_size, shuffle=True)
    
    if args.model == 'stacked_hg':
        model = PoseNet(nstack=args.n_stack, inp_dim=256, oup_dim=14).to(args.device)
    else:
        model = get_pose_net(args, is_train = True)
        model.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model = model.to(args.device)
    wandb.watch(model)
    
    criterion = JointsMSELoss(use_target_weight=args.use_target_weight).to(args.device)
#     criterion = JointsKLLoss().to(args.device)
    
    if args.adam:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr) #, weight_decay=0.0005)
    else:
        optimizer = optim.SGD(model.parameters(),
                              lr=args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    
    
    max_val5_acc = 0
    max_val2_acc = 0 
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        print("==> Training=====================>")

        train_acc_5, train_acc_2, train_loss = train(train_uncover_loader, model, criterion, optimizer, epoch, args)
        
        print("Validation Cover1=====================>")
        val_acc_5_cover1, val_acc_2_cover1 = validate(val_cover1_loader, model, epoch, args)
        
        print("Validation Cover2=====================>")
        val_acc_5_cover2, val_acc_2_cover2 = validate(val_cover2_loader, model, epoch, args)
        
        val_acc_5 = (val_acc_5_cover1 + val_acc_5_cover2)/2
        val_acc_2 = (val_acc_2_cover1 + val_acc_2_cover2)/2        

        wandb.log({"Train Accuracy@0.5": train_acc_5, "Train Accuracy@0.2": train_acc_2,
                   "Train Loss": train_loss, "Val Cover1 Accuracy@0.5": val_acc_5_cover1, "Val Cover1 Accuracy@0.2": val_acc_2_cover1,
                   "Val Cover2 Accuracy@0.5": val_acc_5_cover2, "Val Cover2 Accuracy@0.2": val_acc_2_cover2,
                   "Val Accuracy@0.5": val_acc_5, "Val Accuracy@0.2": val_acc_2})
        
        if val_acc_5 > max_val5_acc:
            best_epoch = epoch
            print('==> Saving Best Model......(according to PCK0.5)')
            max_val5_acc = val_acc_5
            state = {
                'epoch': best_epoch,
                'model': model.state_dict()
            }
            save_file = os.path.join(args.save_folder, 'best_model.pth')
            torch.save(state, save_file)

        elif val_acc_2 > max_val2_acc:
            best_epoch = epoch
            print('==> Saving Best Model......(according to PCK0.2)')
            max_val2_acc = val_acc_2
            state = {
                'epoch': best_epoch,
                'model': model.state_dict()
            }
            save_file = os.path.join(args.save_folder, 'best_model.pth')
            torch.save(state, save_file)


        # regular saving
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model.state_dict()
            }
            save_file = os.path.join(args.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)
    
    
def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy_5 = AverageMeter()
    accuracy_2 = AverageMeter()

    # switch to train mode

    model.train()

    end = time.time()
    for i, ((image, image_cover1, image_cover2, extr_cover1, extr_cover2), target, target_weight) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        image = image.to(args.device)
        image_cover1 = image_cover1.to(args.device)
        image_cover2 = image_cover2.to(args.device)
        extr_cover1 = extr_cover1.to(args.device)
        extr_cover2 = extr_cover2.to(args.device)

        input = torch.cat((image, image_cover1, image_cover2, extr_cover1, extr_cover2))
        #[image, image_cover1, image_cover2]
        target = target.to(args.device)
        target_weight = target_weight.to(args.device)
        target, target_weight = torch.cat((target, target, target, target, target)), torch.cat((target_weight, target_weight, target_weight, target_weight, target_weight)) 
        batch_size = input.shape[0]
        
        if args.model == 'stacked_hg':
            pose = model(input)[0][:, -1]
        else:
            pose, pose_feats = model(input)

        loss = criterion(pose, target, target_weight)

        output_acc = pose.detach().cpu().numpy()
        target_acc = target.detach().cpu().numpy()
        acc_5, avg_acc_5, cnt, pred = accuracy(output_acc, target_acc, thr=0.5)
        acc_2, avg_acc_2, cnt, pred = accuracy(output_acc, target_acc, thr=0.2)

        accuracy_5.update(avg_acc_5 * 100, cnt)
        accuracy_2.update(avg_acc_2 * 100, cnt)
        

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Total Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Train Accuracy@0.5 {acc_5.avg:.5f}\t' \
                  'Train Accuracy@0.2 {acc_2.avg:.5f}\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, acc_5=accuracy_5, acc_2 = accuracy_2)
            
            print(msg)
            
    return accuracy_5.avg, accuracy_2.avg, losses.avg

def validate(val_loader, model, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    accuracy_5 = AverageMeter()
    accuracy_2 = AverageMeter()

    # switch to eval mode
    model.eval()

  
    end = time.time()
    with torch.no_grad():
        for i, (input, target, target_weight) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            # compute output
            input = input.to(args.device)
            target = target.to(args.device)
            target_weight = target_weight.to(args.device)

            batch_size = input.shape[0]
            if args.model == 'stacked_hg':
                pose = model(input)[0]
                output = pose.detach().cpu().numpy()[:, -1]
            else:
                pose, pose_feats = model(input)
                output = pose.detach().cpu().numpy()
           
            target = target.detach().cpu().numpy()
            acc_5, avg_acc_5, cnt, pred = accuracy(output, target, thr=0.5)
            acc_2, avg_acc_2, cnt, pred = accuracy(output, target, thr=0.2)

            accuracy_5.update(avg_acc_5 * 100, cnt)
            accuracy_2.update(avg_acc_2 * 100, cnt)
            
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        epoch_msg = 'Epoch: [{0}]\t' \
                    'Accuracy@0.5 {acc_5.avg:.5f}\t' \
                    'Accuracy@0.2 {acc_2.avg:.5f}\t' .format(
                        epoch, acc_5=accuracy_5, acc_2 = accuracy_2)
        
        print(epoch_msg)
                    
        return accuracy_5.avg, accuracy_2.avg
    
    
if __name__ == '__main__':
    main()


