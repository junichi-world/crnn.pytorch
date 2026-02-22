from __future__ import division
from __future__ import print_function

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

import dataset
import models.crnn as crnn
import utils


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TRAIN_ROOT = os.path.join(BASE_DIR, 'data', 'train_lmdb')
DEFAULT_VAL_ROOT = os.path.join(BASE_DIR, 'data', 'val_lmdb')

parser = argparse.ArgumentParser()
parser.add_argument('--trainRoot', default=DEFAULT_TRAIN_ROOT if os.path.isdir(DEFAULT_TRAIN_ROOT) else None,
                    help='path to train lmdb dataset')
parser.add_argument('--valRoot', default=DEFAULT_VAL_ROOT if os.path.isdir(DEFAULT_VAL_ROOT) else None,
                    help='path to val lmdb dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--cpu', action='store_true', help='force CPU even if CUDA is available')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--pretrained', default='', help='path to pretrained model (to continue training)')
parser.add_argument('--alphabet', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz/')
parser.add_argument('--expr_dir', default='expr', help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate for Critic, not used by adadealta')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--manualSeed', type=int, default=1234, help='reproduce experiment')
parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def build_optimizer(model, opt):
    if opt.adam:
        return optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    if opt.adadelta or not opt.adam:
        return optim.Adadelta(model.parameters())
    return optim.RMSprop(model.parameters(), lr=opt.lr)


def ctc_loss(criterion, preds, text, length):
    batch_size = preds.size(1)
    preds = preds.log_softmax(2)
    preds_size = torch.full((batch_size,), preds.size(0), dtype=torch.long, device=preds.device)
    return criterion(preds, text, preds_size, length) / batch_size, preds_size


def val(net, val_dataset, criterion, converter, device, opt, max_iter=100):
    print('Start val')

    for p in net.parameters():
        p.requires_grad_(False)

    net.eval()
    data_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=True,
        batch_size=opt.batchSize,
        num_workers=int(opt.workers))
    val_iter = iter(data_loader)

    n_correct = 0
    n_samples = 0
    loss_avg = utils.averager()
    max_iter = min(max_iter, len(data_loader))

    last_cpu_texts = []
    last_sim_preds = []
    last_raw_preds = []

    with torch.no_grad():
        for _ in range(max_iter):
            cpu_images, cpu_texts = next(val_iter)
            batch_size = cpu_images.size(0)
            n_samples += batch_size

            images = cpu_images.to(device, non_blocking=device.type == 'cuda')
            t, l = converter.encode(cpu_texts)
            text = t.to(device)
            length = l.to(device)

            preds = net(images)
            cost, preds_size = ctc_loss(criterion, preds, text, length)
            loss_avg.add(cost.detach())

            _, preds_index = preds.max(2)
            preds_index = preds_index.transpose(1, 0).contiguous().view(-1)

            sim_preds = converter.decode(preds_index.cpu(), preds_size.cpu(), raw=False)
            raw_preds = converter.decode(preds_index.cpu(), preds_size.cpu(), raw=True)
            for pred, target in zip(sim_preds, cpu_texts):
                if pred == target.lower():
                    n_correct += 1

            last_cpu_texts = cpu_texts
            last_sim_preds = sim_preds
            last_raw_preds = raw_preds

    for raw_pred, pred, gt in zip(last_raw_preds[:opt.n_test_disp], last_sim_preds[:opt.n_test_disp], last_cpu_texts[:opt.n_test_disp]):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = (n_correct / float(n_samples)) if n_samples else 0.0
    print('Test loss: %f, accuracy: %f' % (loss_avg.val(), accuracy))


def train_batch(net, criterion, optimizer, train_iter, converter, device):
    cpu_images, cpu_texts = next(train_iter)
    batch_size = cpu_images.size(0)
    images = cpu_images.to(device, non_blocking=device.type == 'cuda')
    t, l = converter.encode(cpu_texts)
    text = t.to(device)
    length = l.to(device)

    preds = net(images)
    cost, _ = ctc_loss(criterion, preds, text, length)

    optimizer.zero_grad(set_to_none=True)
    cost.backward()
    optimizer.step()
    return cost.detach(), batch_size


def is_lmdb_dir(path):
    return os.path.isdir(path) and os.path.exists(os.path.join(path, 'data.mdb'))


def resolve_dataset_roots(opt):
    if opt.trainRoot and opt.valRoot:
        return True

    base_dir = BASE_DIR
    candidates = [
        ('data/train', 'data/val'),
        ('data/train', 'data/test'),
        ('data/train_lmdb', 'data/val_lmdb'),
        ('data/train_lmdb', 'data/test_lmdb'),
    ]

    for train_rel, val_rel in candidates:
        train_path = os.path.join(base_dir, train_rel)
        val_path = os.path.join(base_dir, val_rel)
        if is_lmdb_dir(train_path) and is_lmdb_dir(val_path):
            opt.trainRoot = opt.trainRoot or train_path
            opt.valRoot = opt.valRoot or val_path
            break

    if not (opt.trainRoot and opt.valRoot):
        lmdb_dirs = []
        for root, dirs, _ in os.walk(base_dir):
            for d in dirs:
                full = os.path.join(root, d)
                if is_lmdb_dir(full):
                    lmdb_dirs.append(full)

        lmdb_dirs.sort()
        if not opt.trainRoot:
            for path in lmdb_dirs:
                name = os.path.basename(path).lower()
                if 'train' in name:
                    opt.trainRoot = path
                    break
        if not opt.valRoot:
            for path in lmdb_dirs:
                name = os.path.basename(path).lower()
                if 'val' in name or 'test' in name:
                    opt.valRoot = path
                    break

    missing = []
    if not opt.trainRoot:
        missing.append('--trainRoot')
    if not opt.valRoot:
        missing.append('--valRoot')
    if missing:
        print('Missing dataset path(s): {0}'.format(', '.join(missing)))
        print('Set them in Run Configuration or CLI.')
        print('Example: python train.py --trainRoot path/to/train_lmdb --valRoot path/to/val_lmdb')
        print('Auto-detection checks common folders (e.g. data/train_lmdb, data/val_lmdb) and scans for LMDB directories containing data.mdb.')
        return False

    return True


def main():
    opt = parser.parse_args()
    if opt.cuda and opt.cpu:
        parser.error('--cuda and --cpu cannot be used together')
    if not resolve_dataset_roots(opt):
        return
    print(opt)

    if not os.path.exists(opt.expr_dir):
        os.makedirs(opt.expr_dir)

    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if os.name == 'nt' and opt.workers > 0:
        print('WARNING: Windows + lmdb dataset may fail with num_workers > 0 (cannot pickle Environment).')
        print('Setting workers=0 automatically.')
        opt.workers = 0

    cuda_available = torch.cuda.is_available()
    use_cuda = bool((not opt.cpu) and cuda_available)
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('Runtime device: {0}'.format(device))
    if use_cuda:
        print('CUDA device: {0}'.format(torch.cuda.get_device_name(device.index or 0)))
        print('Torch CUDA runtime: {0}'.format(torch.version.cuda))
        if not opt.cuda:
            print('CUDA auto-enabled (use --cpu to force CPU).')
    elif opt.cuda:
        print('CUDA requested but unavailable. Check your PyTorch build (CPU-only) and drivers.')
    elif cuda_available and opt.cpu:
        print('CUDA available, but CPU forced by --cpu.')

    if cuda_available and opt.cpu:
        print('WARNING: CUDA device is available but disabled by --cpu')

    cudnn.benchmark = use_cuda

    train_dataset = dataset.lmdbDataset(root=opt.trainRoot)
    assert train_dataset

    if not opt.random_sample:
        sampler = dataset.randomSequentialSampler(train_dataset, opt.batchSize)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batchSize,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(opt.workers),
        collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio))

    test_dataset = dataset.lmdbDataset(
        root=opt.valRoot,
        transform=dataset.resizeNormalize((opt.imgW, opt.imgH)))

    nclass = len(opt.alphabet) + 1
    nc = 1

    converter = utils.strLabelConverter(opt.alphabet)
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True).to(device)

    net = crnn.CRNN(opt.imgH, nc, nclass, opt.nh)
    net.apply(weights_init)

    if opt.pretrained:
        print('loading pretrained model from %s' % opt.pretrained)
        # Legacy checkpoints may require full unpickling on PyTorch >= 2.6.
        state_dict = torch.load(opt.pretrained, map_location=device, weights_only=False)
        net.load_state_dict(state_dict)

    net = net.to(device)
    if use_cuda and opt.ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(opt.ngpu)))
    print(net)

    loss_avg = utils.averager()
    optimizer = build_optimizer(net, opt)

    for epoch in range(opt.nepoch):
        train_iter = iter(train_loader)
        i = 0
        steps_since_log = 0
        samples_since_log = 0
        if use_cuda:
            torch.cuda.synchronize(device)
        log_start_time = time.perf_counter()
        while i < len(train_loader):
            for p in net.parameters():
                p.requires_grad_(True)
            net.train()

            cost, batch_size = train_batch(net, criterion, optimizer, train_iter, converter, device)
            loss_avg.add(cost)
            i += 1
            steps_since_log += 1
            samples_since_log += batch_size

            if i % opt.displayInterval == 0:
                if use_cuda:
                    torch.cuda.synchronize(device)
                elapsed = max(time.perf_counter() - log_start_time, 1e-9)
                iter_per_sec = steps_since_log / elapsed
                samples_per_sec = samples_since_log / elapsed
                ms_per_iter = (elapsed / max(steps_since_log, 1)) * 1000.0
                print('[%d/%d][%d/%d] Loss: %f | %.2f it/s | %.1f samples/s | %.1f ms/it' %
                      (epoch, opt.nepoch, i, len(train_loader), loss_avg.val(),
                       iter_per_sec, samples_per_sec, ms_per_iter))
                loss_avg.reset()
                steps_since_log = 0
                samples_since_log = 0
                if use_cuda:
                    torch.cuda.synchronize(device)
                log_start_time = time.perf_counter()

            if i % opt.valInterval == 0:
                val(net, test_dataset, criterion, converter, device, opt)

            if i % opt.saveInterval == 0:
                state_dict = net.module.state_dict() if isinstance(net, torch.nn.DataParallel) else net.state_dict()
                torch.save(state_dict, '{0}/netCRNN_{1}_{2}.pth'.format(opt.expr_dir, epoch, i))


if __name__ == '__main__':
    main()
