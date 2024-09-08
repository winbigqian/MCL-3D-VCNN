import argparse
import numpy as np
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.pointnet2_seg import pointnet2_seg_ssg, seg_loss
from data.ShapeNet import ShapeNet
from utils.IoU import cal_accuracy_iou

from clip import clip
from utils.seg_util.SUEM_model_seg import SUEM_model_seg
from utils.seg_util.clip_adapter_model_seg import clip_adapter_model_seg

import matplotlib.pyplot as plt
import logging

def train_one_epoch(train_loader, seg_classes, SUEM_model_seg, optimizer, device, pt):
    losses, preds, labels = [], [], []
    for data, label in train_loader:
        # print("labels-----------", labels)
        labels.append(label)
        optimizer.zero_grad()  # Important
        label = label.long().to(device)  #torch.Size([32, 2500])
        xyz, points = data[:, :, :3], data[:, :, 3:]
        pred, loss = SUEM_model_seg(xyz, points, label)

        loss.backward()
        optimizer.step()
        pred = torch.max(pred, dim=1)[1]
        preds.append(pred.cpu().detach().numpy())
        losses.append(loss.item())
    iou, acc = cal_accuracy_iou(np.concatenate(preds, axis=0), np.concatenate(labels, axis=0), seg_classes, pt)
    return np.mean(losses), iou, acc


def test_one_epoch(test_loader, seg_classes, SUEM_model_seg, device):
    losses, preds, labels = [], [], []
    for data, label in test_loader:
        labels.append(label)
        label = label.long().to(device)
        # print("labels-----------", labels)
        xyz, points = data[:, :, :3], data[:, :, 3:]
        with torch.no_grad():
            pred, loss = SUEM_model_seg(xyz.to(device), points.to(device), label.to(device))
            pred = torch.max(pred, dim=1)[1]
            preds.append(pred.cpu().detach().numpy())
            losses.append(loss.item())
    iou, acc = cal_accuracy_iou(np.concatenate(preds, axis=0), np.concatenate(labels, axis=0), seg_classes)
    return np.mean(losses), iou, acc




def train(train_loader, test_loader, seg_classes, optimizer, scheduler, device, ngpus, nepoches, log_interval, log_dir, checkpoint_interval, SUEM_model_seg):
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
            if epoch % checkpoint_interval * 2 == 0 and epoch > 150:
                if ngpus > 1: # Mutli-gpus
                    torch.save(SUEM_model_seg.state_dict(), os.path.join(checkpoint_dir, "SUEM_model_seg_%d.pth" % epoch))
                else:
                    torch.save(SUEM_model_seg.state_dict(), os.path.join(checkpoint_dir, "SUEM_model_seg_%d.pth" % epoch))
            model.eval()
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            loss, iou, acc = test_one_epoch(test_loader, seg_classes, SUEM_model_seg, device)
            print('Test  Epoch: {} / {}, lr: {:.6f}, Loss: {:.2f}, IoU: {:.4f}, Acc: {:.4f}'.format(epoch, nepoches, lr, loss, iou, acc))
            logging.info('=========================================================================================================')
            logging.info('Test  Epoch: {} / {}, lr: {:.6f}, Loss: {:.2f}, IoU: {:.4f}, Acc: {:.4f}'.format(epoch, nepoches, lr, loss, iou, acc))
            writer.add_scalar('test loss', loss, epoch)
            writer.add_scalar('test iou', iou, epoch)
            writer.add_scalar('test acc', acc, epoch)

            # add into logging.info
            logging.info('test loss: {:.2f}, epoch: {}'.format(loss, epoch))
            logging.info('test iou: {:.4f}, epoch: {}'.format(iou, epoch))
            logging.info('test acc: {:.4f}, epoch: {}'.format(acc, epoch))


        model.train()
        pt = False
        if epoch % log_interval == 0:
            pt = True
        loss, iou, acc = train_one_epoch(train_loader, seg_classes, SUEM_model_seg, optimizer, device, pt)
        writer.add_scalar('train loss', loss, epoch)
        writer.add_scalar('train iou', iou, epoch)
        writer.add_scalar('train acc', acc, epoch)

        # add into logging.info
        logging.info('train loss: {:.2f}, epoch: {}'.format(loss, epoch))
        logging.info('trian iou: {:.4f}, epoch: {}'.format(iou, epoch))
        logging.info('train acc: {:.4f}, epoch: {}'.format(acc, epoch))
        
        if epoch % log_interval == 0:
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            print('Train Epoch: {} / {}, lr: {:.6f}, Loss: {:.2f}, IoU: {:.4f}, Acc: {:.4f}'.format(epoch, nepoches, lr, loss, iou, acc))
            logging.info('Train Epoch: {} / {}, lr: {:.6f}, Loss: {:.2f}, IoU: {:.4f}, Acc: {:.4f}'.format(epoch, nepoches, lr, loss, iou, acc))
        scheduler.step()


if __name__ == '__main__':
    Models = {
        'pointnet2_seg_ssg': pointnet2_seg_ssg,
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='', help='Root to the dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--npoints', type=int, default=2500, help='Number of the training points')
    parser.add_argument('--nclasses', type=int, default=50, help='Number of classes')
    parser.add_argument('--augment', type=bool, default=False, help='Augment the train data')
    parser.add_argument('--dp', type=bool, default=False, help='Random input dropout during training')
    parser.add_argument('--model', type=str, default='pointnet2_seg_ssg', help='Model name')
    parser.add_argument('--gpus', type=str, default='0', help='Cuda ids')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learing rate')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='Initial learing rate')
    parser.add_argument('--nepoches', type=int, default=251, help='Number of traing epoches')
    parser.add_argument('--step_size', type=int, default=20, help='StepLR step size')
    parser.add_argument('--gamma', type=float, default=0.7, help='StepLR gamma')
    parser.add_argument('--log_interval', type=int, default=1, help='Print iterval')
    parser.add_argument('--log_dir', type=str, default='', help='Train/val loss and accuracy logs')
    parser.add_argument('--checkpoint_interval', type=int, default=5, help='Checkpoint saved interval')
    args = parser.parse_args()
    print(args)

    # output log
    logging.basicConfig(filename='', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info(args)


    device_ids = list(map(int, args.gpus.strip().split(','))) if ',' in args.gpus else [int(args.gpus)]
    ngpus = len(device_ids)

    clip_model, preprocess = clip.load("ViT-B/32", device='cuda', jit=False)

    shapenet_train = ShapeNet(data_root=args.data_root, split='trainval', npoints=args.npoints, augment=args.augment, dp=args.dp)
    shapenet_test = ShapeNet(data_root=args.data_root, split='test', npoints=args.npoints)
    train_loader = DataLoader(dataset=shapenet_train, batch_size=args.batch_size // ngpus, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=shapenet_test, batch_size=args.batch_size // ngpus, shuffle=False, num_workers=4)
    print('Train set: {}'.format(len(shapenet_train)))
    print('Test set: {}'.format(len(shapenet_test)))

    Model = Models[args.model]
    model = Model(6, args.nclasses)
    # Mutli-gpus
    device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")
    if ngpus > 1 and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device)

    clip_adapter_model_seg = clip_adapter_model_seg(clip_model, device)
    clip_adapter_model_seg = clip_adapter_model_seg.to(device)
    clip_adapter_model_seg = clip_adapter_model_seg.float()

    loss = seg_loss().to(device)

    SUEM_model_seg = SUEM_model_seg(clip_adapter_model_seg, model, loss, device)  # 总的模型

    # try:
    #
    #     SUEM_model_seg.load_state_dict(torch.load(''))
    #     print("导入模型成功")
    # except Exception as e:
    #     raise RuntimeError(f"Failed to load model parameters: {e}")



    ######################################################################


    #optimizer = torch.optim.SGD(model.parameters(), lr=args.init_lr, momentum=args.momentum)
    optimizer = torch.optim.Adam(
        SUEM_model_seg.parameters(),
        #model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.7)

    tic = time.time()
    train(train_loader=train_loader,
          test_loader=test_loader,
          seg_classes=shapenet_train.seg_classes,
          optimizer=optimizer,
          scheduler=scheduler,
          device=device,
          ngpus=ngpus,
          nepoches=args.nepoches,
          log_interval=args.log_interval,
          log_dir=args.log_dir,
          checkpoint_interval=args.checkpoint_interval,
          SUEM_model_seg=SUEM_model_seg
          )
    toc = time.time()
    print('Training completed, {:.2f} minutes'.format((toc - tic) / 60))
    logging.info('Training completed, {:.2f} minutes'.format((toc - tic) / 60))