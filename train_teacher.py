import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.optim.lr_scheduler import MultiStepLR
from dataset import TeacherDataSet
from model import ViTHead
from utils import vision_transformer
from PIL import Image
from tqdm import tqdm
import argparse


def vis(imgs, segs, epoch, vis_dir, vis_num_rc=[3, 1], mode='train'):
    """
    Args:
       imgs (N, C, H, W)
       segs (N, C, H, W)
    """
    N, C = imgs.shape[:2]
    row_inds = np.random.choice(np.arange(N), min(N, vis_num_rc[0]), False)
    canvas = []
    for row_ind in row_inds:
        col_inds = np.random.choice(np.arange(C), min(C, vis_num_rc[1]), False)
        row_img = []
        for col_ind in col_inds:
            img = imgs[row_ind, col_ind].detach().cpu().numpy() * 255
            img = np.tile(np.expand_dims(img, 2), (1, 1, 3))
            seg = segs[row_ind, col_ind].detach().cpu().numpy() * 255
            seg = np.expand_dims(seg, 2)
            seg = np.concatenate([seg, np.zeros_like(seg), np.zeros_like(seg)], axis=2)
            img_show = 0.5 * img + 0.5 * seg
            row_img.append(img_show)
        row_img = np.concatenate(row_img, axis=1)
        canvas.append(row_img)
    canvas = np.concatenate(canvas, axis=0)
    canvas = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    canvas.save(os.path.join(vis_dir, '{}_{}.png'.format(mode, str(epoch).zfill(6))))


def compute_iou(seg_pred, seg_gt, conf_thres):
    seg_pred_bin = torch.greater_equal(seg_pred, conf_thres)
    seg_gt_bin = torch.greater_equal(seg_gt, conf_thres)
    i = torch.logical_and(seg_pred_bin, seg_gt_bin).float().sum()
    u = torch.logical_or(seg_pred_bin, seg_gt_bin).float().sum()
    intsc = i.detach().cpu().numpy()
    union = u.detach().cpu().numpy()
    return intsc, union


def sample(t, coords):
    return F.grid_sample(t, coords.permute(0, 2, 1, 3), padding_mode='border', align_corners=True)


def norm(t):
    return F.normalize(t, dim=1, eps=1e-10)


def tensor_correlation(a, b):
    return torch.einsum("nchw,ncij->nhwij", a, b)


def correlation_loss(f1, f2, c1, c2, shift=0.1):
    with torch.no_grad():
        fd = tensor_correlation(norm(f1), norm(f2))
        old_mean = fd.mean()
        fd -= fd.mean([3, 4], keepdim=True)
        fd = fd - fd.mean() + old_mean

    cd = tensor_correlation(norm(c1), norm(c2))
    loss = -cd.clamp(0, 0.8) * (fd - shift)
    return loss


def knowledge_distillation_loss(feat_teacher, feat_vit):
    coord_shape = [feat_teacher.shape[0], 16, 16, 2]
    coords = torch.rand(coord_shape, device=feat_teacher.device) * 2 - 1
    feat_vit_s = sample(feat_vit, coords)
    feat_teacher_s = sample(feat_teacher, coords)
    perm = torch.randperm(feat_vit.shape[0])

    loss_self = correlation_loss(feat_vit_s, feat_vit_s, feat_teacher_s, feat_teacher_s)
    loss_perm = correlation_loss(feat_vit_s, feat_vit_s[perm], feat_teacher_s, feat_teacher_s[perm])

    return loss_self.mean(), loss_perm.mean()


def train(seg_head, vit, dataloader, optimizer, device, epoch, vis_dir):
    seg_head.train()
    loss_mean = []
    normalize_imgs = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    
    for imgs, seg_gt, gt_valid in tqdm(dataloader):
        imgs = imgs.to(device)
        seg_gt = seg_gt.to(device)
        gt_valid = gt_valid.to(device)

        feat_vit = vit(   # (B, 384, h_featmap,  w_featmap) 
            normalize_imgs(torch.tile(imgs, (1, 3, 1, 1)))).detach()
        seg_pred = seg_head(feat_vit)

        loss_seg = torch.mean(F.binary_cross_entropy_with_logits(
            seg_pred, seg_gt, reduction='none'), dim=(1, 2, 3))
        loss_seg = torch.sum(loss_seg * gt_valid) / (1e-9 + torch.sum(gt_valid))
        loss = loss_seg 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_mean.append(loss_seg.detach().cpu().numpy())

    loss_mean = np.mean(loss_mean)

    print('Epoch {}, LR: {}, Loss: seg {:.3f}'.format(
        epoch, optimizer.param_groups[0]['lr'], loss_mean))
    
    vis(imgs, torch.sigmoid(seg_pred), epoch, vis_dir, mode='train')


def val(seg_head, vit, dataloader, device, epoch, vis_dir, conf_thres=[0.3, 0.5, 0.7]):
    seg_head.eval()
    intsc_list = []
    union_list = []
    normalize_imgs = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    for imgs, seg_gt, gt_valid in tqdm(dataloader):
        imgs = imgs.to(device)
        seg_gt = seg_gt.to(device)
        gt_valid = gt_valid.to(device)

        feat_vit = vit(   # (B, 384, h_featmap,  w_featmap) 
            normalize_imgs(torch.tile(imgs, (1, 3, 1, 1)))).detach()
        seg_pred = seg_head(feat_vit)
        
        intsc_list.append([])
        union_list.append([])
        for c in conf_thres:
            intsc, union = compute_iou(
                torch.flatten(torch.sigmoid(seg_pred), 1) * gt_valid.reshape(-1, 1), 
                torch.flatten(seg_gt, 1) * gt_valid.reshape(-1, 1), c)
            intsc_list[-1].append(intsc)
            union_list[-1].append(union)

    iou = np.sum(intsc_list, axis=0) / (1e-9 + np.sum(union_list, axis=0))
    print('Validation IoU')
    for j in range(len(conf_thres)):
        print('Conf thres: {:.1f}, IoU: {:.3f}'.format(conf_thres[j], iou[j]))
    vis(imgs, torch.sigmoid(seg_pred), epoch, vis_dir, mode='val')


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg_head = ViTHead()
    seg_head.to(device)
    seg_head = torch.nn.DataParallel(seg_head)
    optimizer = torch.optim.Adam(seg_head.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = MultiStepLR(
        optimizer, milestones=[int(args.epochs * 0.7), int(args.epochs * 0.9)], gamma=0.1)

    vit = vision_transformer.vit_small(patch_size=8)
    for p in vit.parameters():
        p.requires_grad = False
    vit.eval()
    vit.to(device)
    state_dict = torch.load(args.vit_pretrained, map_location="cpu")['teacher']
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = vit.load_state_dict(state_dict, strict=False)
    vit = torch.nn.DataParallel(vit)
    print('ViT pretrained weights found and loaded with msg: {}'.format(msg))

    save_dir = args.ckp
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    vis_dir = args.vis
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    train_dataset = TeacherDataSet(args.data, 'train', anno_ratio=args.anno_ratio, random_state=args.random_state)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=args.workers)
    
    val_dataset = TeacherDataSet(args.data, 'val', anno_ratio=1.0, random_state=args.random_state)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=True, num_workers=1)

    for i in range(args.epochs):
        train(seg_head, vit, train_loader, optimizer, device, i, vis_dir)
        val(seg_head, vit, val_loader, device, i, vis_dir)
        scheduler.step()
        torch.save(seg_head.state_dict(), os.path.join(save_dir, 'seg_head_{}.pth'.format(str(i).zfill(3))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--workers', type=int, default=32, help='number of workers in dataloader')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs in training')
    parser.add_argument('--data', default='./data/cetus', help='dataset root')
    parser.add_argument('--anno_ratio', type=float, default=1.0, help='ratio of samples with annotation')
    parser.add_argument('--random_state', type=int, default=0, help='random state to choose which samples have gt')
    parser.add_argument('--vit_pretrained', type=str, default='./data/pretrained/dino_vit_small.pth', help='path to pretrained vit')

    parser.add_argument('--ckp', default='./outputs/checkpoint', help='path to save checkpoints')
    parser.add_argument('--vis', default='./outputs/vis_teacher', help='path to save visualizations')
    
    args = parser.parse_args()

    main(args)
