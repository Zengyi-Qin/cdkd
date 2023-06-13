import os
import numpy as np 
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from model import ViTHead
from utils import vision_transformer
from tqdm import tqdm
import argparse
import cv2


def inverse_affine(angle, translate):
    r = np.array([[np.cos(np.pi*angle/180), -np.sin(np.pi*angle/180)],
                  [np.sin(np.pi*angle/180),  np.cos(np.pi*angle/180)]])
    t = np.array(translate).reshape(2, 1)
    mat = np.concatenate([r, t], axis=1)
    mat = np.concatenate([mat, np.array([[0, 0, 1]])], axis=0)
    inv_mat = np.linalg.inv(mat)
    inv_angle = np.arctan2(inv_mat[1, 0], inv_mat[0, 0]) / np.pi * 180
    inv_translate = inv_mat[:2, 2].flatten()
    return inv_angle, inv_translate


def inference_random_augmentation(head, vit, imgs_dir, save_dir, num_aug=20, device='cuda'):
    normalize_imgs = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    for img_name in tqdm(sorted(os.listdir(imgs_dir))):
        img = cv2.imread(os.path.join(imgs_dir, img_name))[..., 0]
        img_h, img_w = img.shape

        input_img = torch.tensor(np.expand_dims(img, (0, 1)) / 255.0).float().to(device)

        seg_sum = np.zeros((1, 1, img_h, img_w))
        seg_cnt = np.zeros((1, 1, img_h, img_w))
        seg_sqr_sum = np.zeros((1, 1, img_h, img_w))

        angle_range = 45
        trans_range = max(img_h, img_w) / 10
        for i in range(num_aug):
            angle = np.random.uniform(-angle_range, angle_range)
            translate = np.random.uniform(-trans_range, trans_range, size=(2,))
            inv_angle, inv_translate = inverse_affine(angle, translate)

            img_t = T.functional.affine(input_img, angle, list(translate), 1.0, 0)
            img_r = F.interpolate(img_t, (256, 256), mode='bicubic')

            feat_vit = vit(   # (B, 384, h_featmap,  w_featmap) 
                normalize_imgs(torch.tile(img_r.to(device), (1, 3, 1, 1))))
            seg = head(feat_vit)

            seg = torch.sigmoid(seg)
            seg = F.interpolate(seg, (img_h, img_w), mode='bicubic')
            seg = T.functional.affine(seg, inv_angle, list(inv_translate), 1.0, 0)
            valid = T.functional.affine(
                torch.ones_like(seg), inv_angle, list(inv_translate), 1.0, 0)

            seg_sum = seg_sum + seg.detach().cpu().numpy()
            seg_cnt = seg_cnt + valid.detach().cpu().numpy()
            seg_sqr_sum = seg_sqr_sum + seg.detach().cpu().numpy() ** 2
        
        seg_mean = np.squeeze(seg_sum / (1e-9 + seg_cnt))
        seg_var = np.squeeze(seg_sqr_sum / (1e-9 + seg_cnt)) - seg_mean ** 2
        seg_std = np.sqrt(np.maximum(0, seg_var))

        npz_name = img_name[:-3] + 'npz'
        np.savez(open(os.path.join(save_dir, npz_name), 'wb'), 
            seg_mean=seg_mean.astype(np.float32), seg_std=seg_std.astype(np.float32))


def inference_baseline(head, vit, imgs_dir, save_dir, device='cuda'):
    normalize_imgs = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    for img_name in tqdm(sorted(os.listdir(imgs_dir))):
        img = cv2.imread(os.path.join(imgs_dir, img_name))[..., 0]
        img_h, img_w = img.shape

        input_img = torch.tensor(np.expand_dims(img, (0, 1)) / 255.0).float().to(device)
        input_img = F.interpolate(input_img, (256, 256), mode='bicubic')

        feat_vit = vit(   # (B, 384, h_featmap,  w_featmap) 
            normalize_imgs(torch.tile(input_img, (1, 3, 1, 1))))
        seg = head(feat_vit)

        seg = torch.sigmoid(seg)
        seg = F.interpolate(seg, (img_h, img_w), mode='bicubic')
        seg = np.squeeze(seg.detach().cpu().numpy())

        seg[seg < 0.1] = 0
        seg_std = np.zeros_like(seg)

        npz_name = img_name[:-3] + 'npz'
        np.savez_compressed(open(os.path.join(save_dir, npz_name), 'wb'), 
            seg_mean=seg.astype(np.float16), seg_std=seg_std.astype(np.float16))


def compute_spatial_std(temp_dir, save_dir):
    npz_names = sorted(os.listdir(temp_dir))
    for i in tqdm(range(len(npz_names))):
        frame = np.load(open(os.path.join(temp_dir, npz_names[i]), 'rb'))
        i_prev = i - 1 if i > 0 else i
        i_next = i + 1 if i < len(npz_names) - 1 else i

        frame_prev = np.load(open(os.path.join(temp_dir, npz_names[i_prev]), 'rb'))
        frame_next = np.load(open(os.path.join(temp_dir, npz_names[i_next]), 'rb'))

        var = (frame['seg_mean'] - frame_prev['seg_mean'])**2 + \
            (frame['seg_mean'] - frame_next['seg_mean'])**2
        var = var * 0.5 if i != i_prev and i != i_next else var
        std = np.sqrt(var).astype(np.float32)

        seg_std = frame['seg_std']
        seg_std[seg_std < 0.01] = 0

        seg_std_spatial = std 
        seg_std_spatial[seg_std_spatial < 0.01] = 0

        seg_mean = frame['seg_mean']
        seg_mean[seg_mean < 0.1] = 0

        np.savez_compressed(open(os.path.join(save_dir, npz_names[i]), 'wb'), 
            seg_mean=seg_mean.astype(np.float16), 
            seg_std=seg_std.astype(np.float16),
            seg_std_spatial=seg_std_spatial.astype(np.float16))


def optimize(input_dir, output_dir, seq_len=64, opt_loops=60):
    npz_names = sorted(os.listdir(input_dir))
    start_ind = 0
    while start_ind < len(npz_names):
        end_ind = min(len(npz_names), start_ind + seq_len)
        
        if end_ind - start_ind == 1:
            npz_name = npz_names[start_ind]
            frame = np.load(open(os.path.join(input_dir, npz_name), 'rb'))
            seg_std = frame['seg_std']
            seg_std_spatial = frame['seg_std_spatial']
            seg_mean = frame['seg_mean']
            seg_mean[seg_mean < 0.1] = 0

            np.savez_compressed(open(os.path.join(output_dir, npz_name), 'wb'), 
                seg_mean=seg_mean.astype(np.float16), 
                seg_std=seg_std.astype(np.float16),
                seg_std_spatial=seg_std_spatial.astype(np.float16))
            return

        seg_means = []
        seg_uncertainties = []
        for npz_name in npz_names[start_ind:end_ind]:
            frame = np.load(open(os.path.join(input_dir, npz_name), 'rb'))
            seg_means.append(frame['seg_mean'])
            seg_uncertainties.append(frame['seg_std'])
        seg_means = np.stack(seg_means, axis=0)
        seg_uncertainties = np.stack(seg_uncertainties, axis=0)

        seg_prob = torch.tensor(seg_means, dtype=torch.float32, device='cuda', requires_grad=True)
        seg_prior = torch.tensor(seg_means, dtype=torch.float32, device='cuda', requires_grad=False)
        seg_uncertainties = torch.tensor(seg_uncertainties, dtype=torch.float32, device='cuda', requires_grad=False)
        prior_weight = torch.clip(1 - seg_uncertainties, 0, 1)

        optim = torch.optim.Adam([seg_prob], lr=0.01)
        for i in range(opt_loops):
            diff_z = torch.mean(torch.square(seg_prob[1:] - seg_prob[:-1]))
            diff_y = torch.mean(torch.square(seg_prob[:, 1:] - seg_prob[:, :-1]))
            diff_x = torch.mean(torch.square(seg_prob[:, :, 1:] - seg_prob[:, :, :-1]))
            loss_smoothness = diff_x + diff_y + diff_z
            loss_prior = torch.mean(torch.square(seg_prior - seg_prob) * prior_weight)

            loss = loss_prior * 1000.0 + loss_smoothness * 100.0
            print('Start ind: {}, Step: {}, Loss: {:.3f}'.format(start_ind, i, loss.detach().cpu().numpy()))

            optim.zero_grad()
            loss.backward()
            optim.step()
        seg_prob = np.clip(seg_prob.detach().cpu().numpy(), 0, 1)

        for i, npz_name in enumerate(npz_names[start_ind:end_ind]):
            frame = np.load(open(os.path.join(input_dir, npz_name), 'rb'))
            seg_std = frame['seg_std']
            seg_std_spatial = frame['seg_std_spatial']

            seg_mean = seg_prob[i]
            seg_mean[seg_mean < 0.1] = 0

            np.savez_compressed(open(os.path.join(output_dir, npz_name), 'wb'), 
                seg_mean=seg_mean.astype(np.float16), 
                seg_std=seg_std.astype(np.float16),
                seg_std_spatial=seg_std_spatial.astype(np.float16))

        start_ind = end_ind


def visualize(imgs_dir, save_dir, vis_dir):
    for img_name in tqdm(sorted(os.listdir(imgs_dir))):
        img = cv2.imread(os.path.join(imgs_dir, img_name))[..., 0]
        npz_name = img_name[:-3] + 'npz'
        seg = np.load(open(os.path.join(save_dir, npz_name), 'rb'))
        
        seg_mean = seg['seg_mean']
        seg_mean[seg_mean < 0.1] = 0
        
        seg_std = seg['seg_std']
        seg_std_spatial = seg['seg_std_spatial']

        seg_mean = np.stack([np.zeros_like(seg_mean), np.zeros_like(seg_mean), seg_mean], axis=2)
        seg_std = np.stack([np.zeros_like(seg_std), np.zeros_like(seg_std), seg_std], axis=2)
        seg_std_spatial = np.stack([np.zeros_like(seg_std_spatial), np.zeros_like(seg_std_spatial), seg_std_spatial], axis=2)
        
        img = np.tile(np.expand_dims(img, 2), (1, 1, 3))
        img_show = np.concatenate([
            img,
            img * 0.5 + 255 * seg_mean * 0.5,
            img * 0.5 + 255 * seg_std * 0.5,
            img * 0.5 + 255 * seg_std_spatial * 0.5,
        ], axis=1)
        img_show = np.clip(img_show, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(vis_dir, img_name), img_show)
        

def main(args):
    seg_head = ViTHead()
    state_dict = torch.load(args.head)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    seg_head.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg_head.to(device)
    seg_head.eval()

    vit = vision_transformer.vit_small(patch_size=8)
    state_dict = torch.load(args.vit, map_location="cpu")['teacher']
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    vit.load_state_dict(state_dict, strict=False)
    vit.to(device)
    vit.eval()

    imgs_dir = os.path.join(args.sequence, 'us')
    temp_dir = os.path.join(args.sequence, 'tmp')
    save_dir = os.path.join(args.sequence, 'anno_pseudo_{}'.format(args.method))
    vis_dir  = os.path.join(args.sequence, 'vis')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    if args.method == 'baseline':
        print('Running inference with method baseline')
        inference_baseline(seg_head, vit, imgs_dir, temp_dir, device=device)
        print('Computing spatial std')
        compute_spatial_std(temp_dir, save_dir)
        
    elif args.method == 'randaug':
        print('Running inference with method randaug')
        inference_random_augmentation(seg_head, vit, imgs_dir, temp_dir, num_aug=20, device=device)
        print('Computing spatial std')
        compute_spatial_std(temp_dir, save_dir)

    elif args.method == 'randaugopt':
        print('Running inference with method randaug')
        inference_random_augmentation(seg_head, vit, imgs_dir, temp_dir, num_aug=20, device=device)
        print('Computing spatial std')
        compute_spatial_std(temp_dir, temp_dir)
        print('Optimizing results')
        optimize(temp_dir, save_dir)

    else:
        raise NotImplementedError    
    print('Running visualization')
    visualize(imgs_dir, save_dir, vis_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', type=str, default='./data/liver_1130_pseudo/01')
    parser.add_argument('--head', type=str, default='./outputs/cetus/teacher_vit_270213/models/seg_head_098.pth')
    parser.add_argument('--vit', type=str, default='./data/pretrained/dino_vit_small.pth')
    parser.add_argument('--num_aug', type=int, default=30)
    parser.add_argument('--method', type=str, default='baseline', 
        help='method of pseudo label generation')
    args = parser.parse_args()
    print('Processing {}'.format(args.sequence))
    main(args)
