import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from trainer import Trainer
import utils
import time
trainer = None

import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms, models
import os
import math
from tqdm import tqdm

# Training settings
parser = argparse.ArgumentParser(description='PyTorch ImageNet Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train-dir', default=os.path.expanduser('~/imagenet/train'),
                    help='path to training data')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing allreduce across workers; it multiplies '
                         'total batch size.')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')
parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                    help='apply gradient predivide factor in optimizer (default: 1.0)')

# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=90,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.0125,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=0,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.00005,
                    help='weight decay')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--model', type=str, default='resnet50',
                    help='model to benchmark')
parser.add_argument('--iterations', type=int, default=1000000,
                    help='number of iterations to train')
parser.add_argument('--scheduler_ip', type=str, required=True)
parser.add_argument('--scheduler_port', type=int, default=6889)
parser.add_argument('--trainer_port', type=int)
parser.add_argument('--job_id', type=int, default=-1)


def train(epoch):
    model.train()

    last_timestamp = time.time()

    with tqdm(total=min(len(train_loader), args.iterations),
              desc='Train Epoch     #{}'.format(epoch + 1),
              disable=not verbose) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.iterations <= 0:
                torch.cuda.synchronize()
                break
            else:
                args.iterations -= 1
            
            adjust_learning_rate(epoch, batch_idx)

            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            data_batch = data
            target_batch = target
            output = model(data_batch)
            loss = F.cross_entropy(output, target_batch)
            loss.backward()
            optimizer.step()
            t.update(1)

            timestamp = time.time()
            trainer.record(timestamp - last_timestamp)
            last_timestamp = timestamp



# Horovod: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
# accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
# the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
# After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
def adjust_learning_rate(epoch, batch_idx):
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / len(train_loader)
        lr_adj = 1. / 1 * (epoch * (1 - 1) / args.warmup_epochs + 1)
    elif epoch < 30:
        lr_adj = 1.
    elif epoch < 60:
        lr_adj = 1e-1
    elif epoch < 80:
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.base_lr * 1 * args.batches_per_allreduce * lr_adj


def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += val.detach().cpu()
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n


if __name__ == '__main__':
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    allreduce_batch_size = args.batch_size * args.batches_per_allreduce

    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    cudnn.benchmark = True

    resume_from_epoch = 0

    verbose = 1

    log_writer = None

    kwargs = {'num_workers': 4, 'pin_memory': True, 'drop_last': True} if args.cuda else {}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    train_dataset = \
        datasets.ImageFolder(args.train_dir,
                             transform=transforms.Compose([
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])
                             ]))
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=allreduce_batch_size,
        sampler=train_sampler, **kwargs)

    torch.set_num_threads(4)

    model = getattr(models, args.model)()

    # # By default, Adasum doesn't need scaling up learning rate.
    # # For sum/average with gradient Accumulation: scale learning rate by batches_per_allreduce
    # lr_scaler = args.batches_per_allreduce * hvd.size() if not args.use_adasum else 1
    lr_scaler = 1

    if args.cuda:
        # Move model to GPU.
        model.cuda()

    optimizer = optim.SGD(model.parameters(),
                          lr=(args.base_lr *
                              lr_scaler),
                          momentum=args.momentum, weight_decay=args.wd)


    trainer = Trainer(args.scheduler_ip, args.scheduler_port, utils.get_host_ip(), args.trainer_port, args.job_id, args.batch_size)
    args.epochs = args.iterations // len(train_loader) + 1
    for epoch in range(resume_from_epoch, args.epochs):
        train(epoch)
    
    trainer.close()