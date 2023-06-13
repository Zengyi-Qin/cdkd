import os 
import numpy as np
from numpy.random import RandomState
import torch
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
import json


def binarize(x):
    x[x < 0.5] = 0
    x[x >= 0.5] = 1
    return x


class USDataSet(object):

    def __init__(self, data_root, mode='train', xy_size=256, z_size=16, anno_ratio=1.0, random_state=0):
        assert mode in ['train', 'val']
        self.data_root = data_root
        self.mode = mode

        volumes = sorted(os.listdir(data_root))
        split = json.load(open(os.path.join(data_root, 'trainval_split.json')))[mode]
        self.volumes = [v for v in volumes if v in split]
        
        self.img_names = []
        self.seg_names = []
        for v in self.volumes:
            self.img_names.append(sorted(os.listdir(os.path.join(data_root, v, 'us'))))
            self.seg_names.append(sorted(os.listdir(os.path.join(data_root, v, 'anno_opt'))))
        self.num_imgs_cumsum = np.cumsum([len(i) for i in self.img_names])

        self.xy_size = xy_size
        self.z_size = z_size
        
        rs = RandomState(random_state)
        anno_inds = rs.choice(np.arange(self.num_imgs_cumsum[-1]), 
                              size=int(self.num_imgs_cumsum[-1]*anno_ratio), replace=False)
        self.has_anno = list(anno_inds.astype(np.int32))
        
        print('Mode: {}'.format(mode))
        print('Using volumes: {}'.format(self.volumes))
        print('Dataset length: {}'.format(self.__len__()))
        print('Number of annotations: {}'.format(len(self.has_anno)))
        print('Annotation indices sum: {}'.format(np.sum(anno_inds)))
    
    def __len__(self):
        num_imgs = self.num_imgs_cumsum[-1]
        dataset_len = num_imgs // self.z_size if self.mode == 'val' else num_imgs
        return dataset_len

    def __getitem__(self, i):
        if self.mode == 'train':
            if np.random.uniform() < 0.5:
                # Guarantee that this volume has at least one labeled slice
                i = max(0, np.random.choice(self.has_anno) - np.random.randint(self.z_size))
        else:
            # In validation mode, the step of indices is z_size
            i = i * self.z_size

        v_ind = np.searchsorted(self.num_imgs_cumsum, i, side='right')
        z_ind = i if v_ind == 0 else i - self.num_imgs_cumsum[v_ind-1]
        img_seq = []
        seg_seq = []
        gt_valid = []
        for j in range(self.z_size):
            ind = min(z_ind + j, len(self.img_names[v_ind]) - 1)
            img_name = self.img_names[v_ind][ind]
            img_path = os.path.join(self.data_root, self.volumes[v_ind], 'us', img_name)
            img = np.array(Image.open(img_path))
            img = img[..., 0] if len(img.shape) == 3 else img
            img_seq.append(img)

            if i + j in self.has_anno:
                seg_name = self.seg_names[v_ind][ind]
                seg_path = os.path.join(self.data_root, self.volumes[v_ind], 'anno_opt', seg_name)
                seg = np.load(open(seg_path, 'rb'))
                seg_seq.append(seg['seg_mean'])
                gt_valid.append(1)
            else:
                seg_seq.append(np.zeros_like(img))
                gt_valid.append(0)

        img_seq = np.array(img_seq) / 255.0
        seg_seq = np.array(seg_seq)
        gt_valid = np.array(gt_valid).astype(np.float32)

        comb = torch.tensor(np.concatenate([img_seq, seg_seq], axis=0).astype(np.float32))
        if self.mode == 'train':
            comb = T.RandomAffine(degrees=45, scale=(0.8, 1.2), translate=(0.2, 0.2))(comb)
            comb = T.RandomHorizontalFlip(0.5)(comb)
            # if np.random.uniform() < 0.5:
            #     crop_size = int(max(comb.shape[1], comb.shape[2]) / np.random.uniform(2, 4))
            #     comb = T.RandomCrop(crop_size, pad_if_needed=True)(comb)

        comb = F.interpolate(comb.unsqueeze(0), 
            (self.xy_size, self.xy_size),
            mode='bicubic', align_corners=True).squeeze(0)

        img, seg_gt = torch.split(comb, (len(img_seq), len(seg_seq)), dim=0)
        seg_gt = binarize(seg_gt)

        return img, seg_gt, gt_valid


class StudentMixDataSet(object):

    def __init__(self, data_root, mode='train', xy_size=256, z_size=16, anno_ratio=1.0, pseudo_method='baseline', random_state=0):
        assert mode in ['train', 'val']
        self.data_root = data_root
        self.mode = mode
        self.pseudo_method = pseudo_method

        volumes = sorted(os.listdir(data_root))
        split = json.load(open(os.path.join(data_root, 'trainval_split.json')))[mode]
        self.volumes = [v for v in volumes if v in split]
        
        self.img_names = []
        self.seg_names = []
        self.seg_pseudo_names = []
        for v in self.volumes:
            self.img_names.append(sorted(os.listdir(os.path.join(data_root, v, 'us'))))
            self.seg_names.append(sorted(os.listdir(os.path.join(data_root, v, 'anno_opt'))))
            self.seg_pseudo_names.append(sorted(os.listdir(os.path.join(data_root, v, 'anno_pseudo_'+pseudo_method))))
        self.num_imgs_cumsum = np.cumsum([len(i) for i in self.img_names])

        self.xy_size = xy_size
        self.z_size = z_size

        rs = RandomState(random_state)
        anno_inds = rs.choice(np.arange(self.num_imgs_cumsum[-1]), 
                              size=int(self.num_imgs_cumsum[-1]*anno_ratio), replace=False)
        
        self.has_anno = list(anno_inds.astype(np.int32))
        
        print('Mode: {}'.format(mode))
        print('Using volumes: {}'.format(self.volumes))
        print('Dataset length: {}'.format(self.__len__()))
        print('Number of annotations: {}'.format(len(self.has_anno)))
        print('Annotation indices sum: {}'.format(np.sum(anno_inds)))
        print('Using pseudo label generation method {}'.format(pseudo_method))
    
    def __len__(self):
        num_imgs = self.num_imgs_cumsum[-1]
        dataset_len = num_imgs // self.z_size if self.mode == 'val' else num_imgs
        return dataset_len

    def __getitem__(self, i):
        i = i * self.z_size if self.mode == 'val' else i
        v_ind = np.searchsorted(self.num_imgs_cumsum, i, side='right')
        z_ind = i if v_ind == 0 else i - self.num_imgs_cumsum[v_ind-1]
        img_seq = []
        seg_seq = []
        gt_valid = []
        for j in range(self.z_size):
            ind = min(z_ind + j, len(self.img_names[v_ind]) - 1)
            img_name = self.img_names[v_ind][ind]
            img_path = os.path.join(self.data_root, self.volumes[v_ind], 'us', img_name)
            img = np.array(Image.open(img_path))
            img = img[..., 0] if len(img.shape) == 3 else img
            img_seq.append(img)

            if i + j in self.has_anno:
                seg_name = self.seg_names[v_ind][ind]
                seg_path = os.path.join(self.data_root, self.volumes[v_ind], 'anno_opt', seg_name)
                seg = np.load(open(seg_path, 'rb'))
                seg_seq.append(seg['seg_mean'])
                gt_valid.append(1)
            else:
                seg_name = self.seg_names[v_ind][ind]
                seg_path = os.path.join(self.data_root, self.volumes[v_ind], 'anno_pseudo_'+self.pseudo_method, seg_name)
                seg = np.load(open(seg_path, 'rb'))
                seg_seq.append(seg['seg_mean'])
                gt_valid.append(1)

        img_seq = np.array(img_seq) / 255.0
        seg_seq = np.array(seg_seq)
        gt_valid = np.array(gt_valid).astype(np.float32)

        comb = torch.tensor(np.concatenate([img_seq, seg_seq], axis=0).astype(np.float32))
        if self.mode == 'train':
            comb = T.RandomAffine(degrees=45, scale=(0.8, 1.2), translate=(0.2, 0.2))(comb)
            comb = T.RandomHorizontalFlip(0.5)(comb)
            # if np.random.uniform() < 0.5:
            #     crop_size = int(max(comb.shape[1], comb.shape[2]) / np.random.uniform(2, 4))
            #     comb = T.RandomCrop(crop_size, pad_if_needed=True)(comb)

        comb = F.interpolate(comb.unsqueeze(0), 
            (self.xy_size, self.xy_size),
            mode='bicubic', align_corners=True).squeeze(0)

        img, seg_gt = torch.split(comb, (len(img_seq), len(seg_seq)), dim=0)
        seg_gt = binarize(seg_gt)

        return img, seg_gt, gt_valid

   
class TeacherDataSet(object):

    def __init__(self, data_root, mode='train', output_size=256, anno_ratio=1.0, random_state=0):
        assert mode in ['train', 'val']
        self.data_root = data_root
        self.mode = mode

        volumes = sorted(os.listdir(data_root))
        split = json.load(open(os.path.join(data_root, 'trainval_split.json')))[mode]
        volumes = [v for v in volumes if v in split]
        self.img_names = []
        self.anno_names = []
        for v in volumes:
            for img_name in sorted(os.listdir(os.path.join(data_root, v, 'us'))):
                self.img_names.append(v + '/us/' + img_name)
            for anno_name in sorted(os.listdir(os.path.join(data_root, v, 'anno_opt'))):
                self.anno_names.append(v + '/anno_opt/' + anno_name)

        rs = RandomState(random_state)
        anno_inds = rs.choice(np.arange(len(self.img_names)), 
                              size=int(len(self.img_names)*anno_ratio), replace=False)
        
        self.has_anno = list(anno_inds.astype(np.int32))
        self.output_size = output_size
        
        print('Mode: {}'.format(mode))
        print('Using volumes: {}'.format(volumes))
        print('Dataset length: {}'.format(self.__len__()))
        print('Number of annotations: {}'.format(len(anno_inds)))
        print('Annotation indices sum: {}'.format(np.sum(anno_inds)))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, i):
        if self.mode == 'train' and np.random.uniform() < 0.5:
            i = np.random.choice(self.has_anno)
        img_path = os.path.join(self.data_root, self.img_names[i])
        img = np.array(Image.open(img_path))
        img = img[..., 0] if len(img.shape) == 3 else img
        img = img / 255.0

        if i in self.has_anno:
            anno_path = os.path.join(self.data_root, self.anno_names[i])
            seg_gt = np.load(open(anno_path, 'rb'))['seg_mean']
            gt_valid = 1
        else:
            seg_gt = np.zeros_like(img)
            gt_valid = 0

        comb = torch.tensor(np.stack([img, seg_gt], axis=0).astype(np.float32))
        if self.mode == 'train':
            comb = T.RandomAffine(degrees=45, scale=(0.8, 1.2), translate=(0.2, 0.2))(comb)
            comb = T.RandomHorizontalFlip(0.5)(comb)
            # if np.random.uniform() < 0.5:
            #     crop_size = int(max(comb.shape[1], comb.shape[2]) / np.random.uniform(2, 4))
            #     comb = T.RandomCrop(crop_size, pad_if_needed=True)(comb)

        comb = F.interpolate(
            comb.unsqueeze(0), 
            (self.output_size, self.output_size),
            mode='bicubic', align_corners=True).squeeze(0)

        img, seg_gt = torch.split(comb, (1, 1), dim=0)
        seg_gt = binarize(seg_gt)

        return img, seg_gt, gt_valid


class UATEDataSet(object):

    def __init__(self, data_root, pseudo_root, mode='train', xy_size=256, z_size=16, anno_ratio=1.0, random_state=0):
        assert mode in ['train', 'val', 'ensemble']
        self.data_root = data_root
        self.pseudo_root = pseudo_root
        self.mode = mode

        volumes = sorted(os.listdir(data_root))
        if mode == 'train' or mode == 'ensemble':
            split = json.load(open(os.path.join(data_root, 'trainval_split.json')))['train']
        else:
            split = json.load(open(os.path.join(data_root, 'trainval_split.json')))['val']
        self.volumes = [v for v in volumes if v in split]
        
        self.img_names = []
        self.seg_names = []
        for v in self.volumes:
            self.img_names.append(sorted(os.listdir(os.path.join(data_root, v, 'us'))))
            self.seg_names.append(sorted(os.listdir(os.path.join(data_root, v, 'anno_opt'))))
        self.pseudo_gt_names = self.seg_names
        self.num_imgs_cumsum = np.cumsum([len(i) for i in self.img_names])

        self.xy_size = xy_size
        self.z_size = z_size

        rs = RandomState(random_state)
        anno_inds = rs.choice(np.arange(self.num_imgs_cumsum[-1]), 
                              size=int(self.num_imgs_cumsum[-1]*anno_ratio), replace=False)
        
        self.has_anno = list(anno_inds.astype(np.int32))
        
        print('Mode: {}'.format(mode))
        print('Using volumes: {}'.format(self.volumes))
        print('Dataset length: {}'.format(self.__len__()))
        print('Number of annotations: {}'.format(len(self.has_anno)))
        print('Annotation indices sum: {}'.format(np.sum(anno_inds)))
    
    def __len__(self):
        num_imgs = self.num_imgs_cumsum[-1]
        dataset_len = num_imgs // self.z_size if self.mode == 'val' or self.mode == 'ensemble' else num_imgs
        return dataset_len

    def __getitem__(self, i):
        if self.mode == 'train':
            if np.random.uniform() < 0.5:
                # Guarantee that this volume has at least one labeled slice
                i = max(0, np.random.choice(self.has_anno) - np.random.randint(self.z_size))
        else:
            # In validation mode, the step of indices is z_size
            i = i * self.z_size

        v_ind = np.searchsorted(self.num_imgs_cumsum, i, side='right')
        z_ind = i if v_ind == 0 else i - self.num_imgs_cumsum[v_ind-1]
        img_seq = []
        seg_seq = []
        uncertainty_seq = []
        gt_valid = []
        seg_paths = []
        for j in range(self.z_size):
            ind = min(z_ind + j, len(self.img_names[v_ind]) - 1)
            img_name = self.img_names[v_ind][ind]
            img_path = os.path.join(self.data_root, self.volumes[v_ind], 'us', img_name)
            img = np.array(Image.open(img_path))
            img = img[..., 0] if len(img.shape) == 3 else img
            img_seq.append(img)

            if i + j in self.has_anno:
                seg_name = self.seg_names[v_ind][ind]
                seg_path = os.path.join(self.data_root, self.volumes[v_ind], 'anno_opt', seg_name)
                seg = np.load(open(seg_path, 'rb'))
                seg_seq.append(seg['seg_mean'])
                uncertainty_seq.append(np.zeros_like(seg['seg_mean']))
                gt_valid.append(1)
            else:
                seg_name = self.pseudo_gt_names[v_ind][ind]
                seg_path = os.path.join(self.pseudo_root, self.volumes[v_ind], 'anno_opt')
                if not os.path.exists(seg_path):
                    os.makedirs(seg_path)
                seg_path = os.path.join(seg_path, seg_name)
                if not os.path.exists(seg_path):
                    np.savez_compressed(
                        open(seg_path, 'wb'), 
                        seg_mean=np.zeros_like(img).astype(np.float16),
                        seg_uncertainty=np.zeros_like(img).astype(np.float16))
                seg = np.load(open(seg_path, 'rb'))
                seg_seq.append(seg['seg_mean'])
                uncertainty_seq.append(seg['seg_uncertainty'])
                gt_valid.append(0)

            seg_paths.append(seg_path)

        img_seq = np.array(img_seq) / 255.0
        seg_seq = np.array(seg_seq)
        uncertainty_seq = np.array(uncertainty_seq)
        gt_valid = np.array(gt_valid).astype(np.float32)

        comb = torch.tensor(
            np.concatenate([img_seq, seg_seq, uncertainty_seq], axis=0).astype(np.float32))
        if self.mode == 'train':
            comb = T.RandomAffine(degrees=45, scale=(0.8, 1.2), translate=(0.2, 0.2))(comb)
            comb = T.RandomHorizontalFlip(0.5)(comb)
            # if np.random.uniform() < 0.5:
            #     crop_size = int(max(comb.shape[1], comb.shape[2]) / np.random.uniform(2, 4))
            #     comb = T.RandomCrop(crop_size, pad_if_needed=True)(comb)

        comb = F.interpolate(comb.unsqueeze(0), 
            (self.xy_size, self.xy_size),
            mode='bicubic', align_corners=True).squeeze(0)

        img, seg_gt, uncertainty = torch.split(
            comb, (len(img_seq), len(seg_seq), len(uncertainty_seq)), dim=0)
        seg_gt = binarize(seg_gt)

        return img, seg_gt, gt_valid, uncertainty, ':'.join(seg_paths)
