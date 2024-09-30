import argparse
import numpy as np
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.ScanObjectNN import ScanObjectNN
from models.pointnet2_cls import pointnet2_cls_ssg, pointnet2_cls_msg, cls_loss
from data.ModelNet40 import ModelNet40
from clip import clip
import torch.nn.functional as F
import logging
import random

from utils.prompts_feature import prompts_feature
from utils.clip_adapter_model import clip_adapter_model
from utils.my_utils.SUEM_model import SUEM_model


seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  #
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



def cosine_loss(A, B, t=1):
    return torch.clamp(t - F.cosine_similarity(A, B, dim=-1), min=0).mean()



def train_one_epoch(train_loader, SUEM_model, optimizer, device):
    losses, total_seen, total_correct = [], 0, 0

    for data, labels in train_loader:
        optimizer.zero_grad()  # Important
        labels = labels.to(device)
        data = data.to(device)
        xyz = data
        points = data
        # xyz, points = data[:, :, :3], data[:, :, 3:]

        pred, loss = SUEM_model(xyz, points, labels)

        loss.backward()
        optimizer.step()
        pred = torch.max(pred, dim=-1)[1]
        total_correct += torch.sum(pred == labels)
        total_seen += xyz.shape[0]
        losses.append(loss.item())
    return np.mean(losses), total_correct, total_seen, total_correct / float(total_seen)


def test_one_epoch(test_loader, SUEM_model, device):
    losses, total_seen, total_correct = [], 0, 0

    for data, labels in test_loader:
        labels = labels.to(device)
        data = data.to(device)
        xyz = data
        points = data
        # xyz, points = data[:, :, :3], data[:, :, 3:]
        with torch.no_grad():

            pred, loss = SUEM_model(xyz, points, labels)

            pred = torch.max(pred, dim=-1)[1]
            total_correct += torch.sum(pred == labels)
            total_seen += xyz.shape[0]
            losses.append(loss.item())
    return np.mean(losses), total_correct, total_seen, total_correct / float(total_seen)


def train(train_loader, test_loader, optimizer, scheduler, device, ngpus, nepoches, log_interval, log_dir, checkpoint_interval, SUEM_model):  # ---------


    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    tensorboard_dir = os.path.join(log_dir, 'tensorboard')
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    writer = SummaryWriter(tensorboard_dir)
    for epoch in range(nepoches):
        if epoch % checkpoint_interval == 0:
            print('='*40)
            # if epoch % checkpoint_interval*2 == 0 and epoch > 150:
            #     if ngpus > 1:
            #         torch.save(SUEM_model.state_dict(), os.path.join(checkpoint_dir, "SUEM_model_cls_%d.pth" % epoch))
            #     else:
            #         torch.save(SUEM_model.state_dict(), os.path.join(checkpoint_dir, "SUEM_model_cls_%d.pth" % epoch))
            if ngpus > 1:
                torch.save(SUEM_model.state_dict(), os.path.join(checkpoint_dir, "SUEM_model_cls_%d.pth" % epoch))
            else:
                torch.save(SUEM_model.state_dict(), os.path.join(checkpoint_dir, "SUEM_model_cls_%d.pth" % epoch))
            SUEM_model.eval()
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            loss, total_correct, total_seen, acc = test_one_epoch(test_loader, SUEM_model, device)   #----------------
            print('Test  Epoch: {} / {}, lr: {:.6f}, Loss: {:.2f}, Corr: {}, Total: {}, Acc: {:.4f}'.format(epoch, nepoches, lr, loss, total_correct, total_seen, acc))
            logging.info('=========================================================================================================')
            logging.info('Test Epoch: {} / {}, lr: {:.6f}, Loss: {:.2f}, Corr: {}, Total: {}, Acc: {:.4f}'.format(epoch,
                                                                                                                 nepoches,
                                                                                                                 lr,
                                                                                                                 loss,
                                                                                                                 total_correct,
                                                                                                                 total_seen,
                                                                                                                 acc))

            writer.add_scalar('test loss', loss, epoch)
            writer.add_scalar('test acc', acc, epoch)
        SUEM_model.train()
        loss, total_correct, total_seen, acc = train_one_epoch(train_loader, SUEM_model, optimizer, device)  #-----------------

        writer.add_scalar('train loss', loss, epoch)
        writer.add_scalar('train acc', acc, epoch)
        if epoch % log_interval == 0:  # 每隔一定间隔打印训练结果
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            print('Train Epoch: {} / {}, lr: {:.6f}, Loss: {:.2f}, Corr: {}, Total: {}, Acc: {:.4f}'.format(epoch, nepoches, lr, loss, total_correct, total_seen, acc))
            logging.info('Train Epoch: {} / {}, lr: {:.6f}, Loss: {:.2f}, Corr: {}, Total: {}, Acc: {:.4f}'.format(epoch,
                                                                                                                  nepoches,
                                                                                                                  lr,
                                                                                                                  loss,
                                                                                                                  total_correct,
                                                                                                                  total_seen,
                                                                                                                  acc))
        scheduler.step()


if __name__ == '__main__':
    Models = {
        'pointnet2_cls_ssg': pointnet2_cls_ssg,
        'pointnet2_cls_msg': pointnet2_cls_msg
    }
    parser = argparse.ArgumentParser()

    #parser.add_argument('--data_root', type=str, default='', help='to the ScanObject chnage the url')

    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--npoints', type=int, default=1024, help='Number of the training points')
    parser.add_argument('--nclasses', type=int, default=40, help='Number of classes')
    parser.add_argument('--augment', type=bool, default=False, help='Augment the train data')
    parser.add_argument('--dp', type=bool, default=False, help='Random input dropout during training')
    parser.add_argument('--model', type=str, default='pointnet2_cls_ssg', help='Model name')
    parser.add_argument('--gpus', type=str, default='0', help='Cuda ids')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learing rate')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='Initial learing rate')
    parser.add_argument('--nepoches', type=int, default=251, help='Number of traing epoches')
    parser.add_argument('--step_size', type=int, default=20, help='StepLR step size')
    parser.add_argument('--gamma', type=float, default=0.7, help='StepLR gamma')
    parser.add_argument('--log_interval', type=int, default=1, help='Print iterval')
    parser.add_argument('--log_dir', type=str, default='my_log_cls_test_6_23', help='Train/val loss and accuracy logs')
    parser.add_argument('--checkpoint_interval', type=int, default=1, help='Checkpoint saved interval')
    args = parser.parse_args()
    print(args)


    logging.basicConfig(filename='', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info(args)



    device_ids = list(map(int, args.gpus.strip().split(','))) if ',' in args.gpus else [int(args.gpus)]
    ngpus = len(device_ids)


    clip_model, preprocess = clip.load("ViT-B/32", device='cuda',jit=False)  #---------------------导入clip模型获取文本特征，下面训练集和测试集都加了


    scanobjetcnn_train = ScanObjectNN(partition='training', num_points=args.npoints)
    scanobjetcnn_test = ScanObjectNN(partition='test', num_points=args.npoints)
    train_loader = DataLoader(scanobjetcnn_train, num_workers=4,
               batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(scanobjetcnn_test, num_workers=4,
                             batch_size=args.batch_size, shuffle=True, drop_last=False)



    print('Train set: {}'.format(len(scanobjetcnn_train)))
    print('Test set: {}'.format(len(scanobjetcnn_test)))

    Model = Models[args.model]
    model = Model(6, args.nclasses)
    # Mutli-gpus
    device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")
    if ngpus > 1 and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device)

    clip_adapter_model = clip_adapter_model(clip_model, device)
    clip_adapter_model = clip_adapter_model.to(device)
    clip_adapter_model = clip_adapter_model.float()

    loss_func = cls_loss().to(device)
    SUEM_model = SUEM_model(clip_adapter_model, model, loss_func, device)  #总的模型



    ######################################################################



    parameters = SUEM_model.parameters()
    optimizer = torch.optim.Adam(
        parameters,
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.7)

    tic = time.time()

    train(train_loader=train_loader,
          test_loader=test_loader,
          optimizer=optimizer,
          scheduler=scheduler,
          device=device,
          ngpus=ngpus,
          nepoches=args.nepoches,
          log_interval=args.log_interval,
          log_dir=args.log_dir,
          checkpoint_interval=args.checkpoint_interval,
          SUEM_model=SUEM_model
          )
    toc = time.time()
    print('Training completed, {:.2f} minutes'.format((toc - tic) / 60))
    logging.info('Training completed, {:.2f} minutes'.format((toc - tic) / 60))