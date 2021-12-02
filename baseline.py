# ========== Thanks https://github.com/Eric-mingjie/rethinking-network-pruning ============
# ========== we adopt the code from the above link and did modifications ============
# ========== the comments as #=== === were added by us, while the comments as # were the original one ============

from __future__ import print_function

import argparse
import os
import random
import shutil
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import models as models
from models.layers import SoftMaskedConv2d

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/CIFAR100/TinyImagenet Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=160, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=50, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[80, 120],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--save_dir', default='results/', type=str)
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

# ========== mask =============
parser.add_argument('--mask-initial-value', type=float, default=0., help='initial value for mask parameters')
parser.add_argument('--s_lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--b_lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--final-temp', type=float, default=200, help='temperature at the end of each round (default: 200)')
parser.add_argument('--s_init', type=float, default=-8.0, help='initial value for mask parameters')
parser.add_argument('--b_init', type=float, default=1., help='initial value for mask parameters')
parser.add_argument('--s_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--b_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

# ========== imagenet =============
parser.add_argument('--datapath', default='', type=str)
parser.add_argument('--distributed', type=int, default=0, help='distributed or not')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'tinyimagenet' or args.dataset == 'imagenet', 'Dataset can only be cifar10 or cifar100 or tinyimagenet.'


gpu_id = args.gpu_id
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    os.makedirs(args.save_dir, exist_ok=True)

    # ========== Data
    # ========== The following preprocessing procedure is adopted from https://github.com/alecwangcq/GraSP ============
    print('==> Preparing dataset %s' % args.dataset)
    if args.dataset == 'cifar10':
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
        dataloader = datasets.CIFAR10
        num_classes = 10
    elif args.dataset == 'cifar100':
        dataloader = datasets.CIFAR100
        num_classes = 100
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
    elif args.dataset == 'tinyimagenet':
        args.schedule = [150,225]
        num_classes = 200
        tiny_mean = [0.48024578664982126, 0.44807218089384643, 0.3975477478649648]
        tiny_std = [0.2769864069088257, 0.26906448510256, 0.282081906210584]
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(tiny_mean, tiny_std)])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(tiny_mean, tiny_std)])
        args.workers = 16
        args.epochs = 300
    elif args.dataset == 'imagenet':
        args.epochs = 90
        args.schedule = [30,60]
        args.train_batch = 256
        args.test_batch = 256
        args.workers = 8
        num_classes = 1000

        traindir = os.path.join(args.datapath, 'train')
        valdir = os.path.join(args.datapath, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        trainloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.train_batch, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)


        val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
        testloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.test_batch, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
        trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
        testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
        testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    elif args.dataset == 'tinyimagenet':
        trainset = datasets.ImageFolder('./data' + '/tiny_imagenet/train', transform=transform_train)
        trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
        testset = datasets.ImageFolder('./data' + '/tiny_imagenet/valset', transform=transform_test)
        testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    
    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    mask_initial_value=args.mask_initial_value
                )
    else:
        model = models.__dict__[args.arch](mask_initial_value=args.mask_initial_value,num_classes=num_classes)

    model.cuda()

    cudnn.benchmark = True
    total_weights = sum(p.weight.data.numel() for p in model.modules() if isinstance(p,nn.Linear) or isinstance(p,nn.Conv2d) or isinstance(p,SoftMaskedConv2d))/1000000.0
    print('    Total Conv and Linear Params: %.2fM' % (total_weights))

    criterion = nn.CrossEntropyLoss()

    weight_params = map(lambda a: a[1], filter(lambda p: p[1].requires_grad and 'sparseThreshold' not in p[0] and 'betaThreshold' not in p[0], model.named_parameters()))
    s_params = map(lambda a: a[1], filter(lambda p: p[1].requires_grad and 'sparseThreshold' in p[0], model.named_parameters()))
    b_params = map(lambda a: a[1], filter(lambda p: p[1].requires_grad and 'betaThreshold' in p[0], model.named_parameters()))
    weight_optim = optim.SGD(weight_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay) 
    s_optim = optim.SGD(s_params, lr=args.s_lr, momentum=args.momentum, weight_decay=args.s_decay)
    b_optim = optim.SGD(b_params, lr=args.b_lr, momentum=args.momentum, weight_decay=args.b_decay)
    optimizers = [weight_optim, s_optim, b_optim]

    for m in model.mask_modules:
        m.sparseThreshold.data = torch.tensor(args.s_init).cuda()
        m.betaThreshold.data = torch.tensor(1e6*args.b_init).cuda()


    # Resume
    if args.dataset == 'cifar10':
        title = 'cifar-10-' + args.arch
    elif args.dataset == 'cifar100':
        title = 'cifar-100-' + args.arch
    elif args.dataset == 'tinyimagenet':
        title = 'tinyimagenet' + args.arch
    elif args.dataset == 'imagenet':
        title = 'imagenet' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.save_dir = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizers[0].load_state_dict(checkpoint['optimizer_w'])
        optimizers[1].load_state_dict(checkpoint['optimizer_s'])
        optimizers[2].load_state_dict(checkpoint['optimizer_b'])
        logger = Logger(os.path.join(args.save_dir, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.save_dir, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.', 'sparsity'])

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    #save_checkpoint({'state_dict': model.state_dict()}, False, checkpoint=args.save_dir, filename='init.pth.tar')
    
    # Train and val
    for epoch in range(start_epoch, args.epochs):
        for m in model.mask_modules:
            m.ticket = False
        if len(optimizers) == 3:
            adjust_learning_rate(optimizers[0], epoch, 'weight')
            adjust_learning_rate(optimizers[1], epoch, 's')
            adjust_learning_rate(optimizers[2], epoch, 'b')

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(trainloader, model, criterion, optimizers, epoch, use_cuda)

        test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)
        
        for m in model.mask_modules:
            m.ticket = True
        
        test_loss_r, test_acc_r = test(testloader, model, criterion, epoch, use_cuda)

        count = 0
        sum_sparse = 0.0

        for m in model.mask_modules:
            sparsity, total_params, thresh, beta, bb = m.getSparsity(ticket=True)
            sum_sparse += int(((100 - sparsity) / 100) * total_params)
            count += total_params
        #total_sparsity = 100 - (100 * sum_sparse / count)
        total_sparsity = 100 - (100 * sum_sparse / (total_weights*1000000))
        print('\tBeta: {}\tb: {}\tS: {}\tTest acc: {}\tRTest acc: {}\tSparsity: {}'.format(beta, bb, thresh, test_acc, test_acc_r, total_sparsity))


        #remaining_weights,Remaining = 0,0#compute_remaining_weights(tkmasks, total_weights)
        #mm = [round((torch.abs(m.tp)).item(), 4) for m in model.mask_modules]
        #print(mm)
        #print('Remaining weights: {:.4f}\tRemaining weights: {:.4f}'.format(remaining_weights,Remaining))

        
        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc, total_sparsity])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer_w' : optimizers[0].state_dict(),
                'optimizer_m' : optimizers[1].state_dict(),
                'optimizer_t' : optimizers[2].state_dict(),
            }, is_best, checkpoint=args.save_dir)
        

    logger.close()
    
    print('Best acc:')
    print(best_acc)


def train(trainloader, model, criterion, optimizers, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    print(args)
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)

        loss = criterion(outputs, targets)


        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        for optimizer in optimizers: optimizer.zero_grad()
        loss.backward()
        for optimizer in optimizers: optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)

        loss = criterion(outputs, targets)


        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch, str=None):
    global state
    if str == 'weight':
        if epoch in args.schedule:
            state['lr'] *= args.gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = state['lr']
    elif str == 's':
        if epoch in args.schedule:
            state['s_lr'] *= args.gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = state['s_lr']
    elif str == 'b':
        if epoch in args.schedule:
            state['b_lr'] *= args.gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = state['b_lr']


def compute_remaining_weights(masks, total_weights):
    return 1 - sum(float((m == 0).sum()) for m in masks) / sum(m.numel() for m in masks), 1 - sum(float((m == 0).sum()) for m in masks) / (total_weights*1000000.0)

def compute_val(masks):
    mean = sum(float(m.sum()) for m in masks) / sum(m.numel() for m in masks)
    negative = sum(float((m < 0).sum()) for m in masks) / sum(m.numel() for m in masks)
    zerotive = sum(float((m == 0).sum()) for m in masks) / sum(m.numel() for m in masks)
    positive = sum(float((m > 0).sum()) for m in masks) / sum(m.numel() for m in masks)

    return mean, negative, zerotive, positive

def ticket_set(ticket):
    for m in model.mask_modules:
        m.ticket = ticket

if __name__ == '__main__':
    main()
