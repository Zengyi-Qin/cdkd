import os 
import numpy as np
import torch 
import torch.nn.functional as F
from model import *
from PIL import Image
import argparse


def inference(model, input_dir, output_dir, z_size=16, xy_size=256):
    img_names = sorted(os.listdir(input_dir))
    start_ind = 0
    while start_ind < len(img_names):
        end_ind = min(len(img_names), start_ind + z_size)
        print('Processing {} / {}'.format(end_ind, len(img_names)))
        img_seq = []
        for img_name in img_names[start_ind:end_ind]:
            img_path = os.path.join(input_dir, img_name)
            img = np.array(Image.open(img_path))
            img = img[..., 0] if len(img.shape) == 3 else img
            img_seq.append(img)
        while len(img_seq) < z_size:
            img_seq.append(img_seq[-1])

        img_seq = np.array(img_seq) / 255.0
        img_h, img_w = img_seq.shape[1:]
        img_seq = torch.tensor(img_seq.astype(np.float32), device='cuda')
        img_seq = F.interpolate(
            img_seq.unsqueeze(0), (xy_size, xy_size), mode='bicubic', align_corners=True)

        with torch.no_grad():
            res = model(img_seq)

        seg_pred = res[0] if isinstance(res, tuple) else res
        seg_pred = F.interpolate(
            torch.sigmoid(seg_pred), (img_h, img_w), mode='bicubic', align_corners=True)
        seg_pred = torch.clip(seg_pred, 0, 1).detach().cpu().numpy()[0]

        for i, img_name in enumerate(img_names[start_ind:end_ind]):
            seg_mean = seg_pred[i]
            seg_mean[seg_mean < 0.1] = 0
            npz_name = img_name[:-4] + '.npz'
            np.savez_compressed(open(os.path.join(output_dir, npz_name), 'wb'), 
                seg_mean=seg_mean.astype(np.float16),
                seg_std=np.zeros_like(seg_mean),
                seg_std_spatial=np.zeros_like(seg_mean))

        start_ind = end_ind


def main(args):

    if args.method == 'stcn':
        model = STCN()
    elif args.method == 'dan':
        model = DANSegNet()
    elif args.method == 'uate':
        model = UATE()
    elif args.method == 'uamt':
        model = UAMT()
    elif args.method == 'student':
        model = Student()
    else:
        raise NotImplementedError

    state_dict = torch.load(args.pretrained)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to('cuda')
    model.eval()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    inference(model, args.input_dir, args.output_dir)


if __name__ == '__main__':
    assert torch.cuda.is_available()

    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True, help='method name')
    parser.add_argument('--pretrained', type=str, required=True, help='path to trained weights')
    parser.add_argument('--input_dir', type=str, required=True, help='path to ultrasound image sequence')
    parser.add_argument('--output_dir', type=str, required=True, help='path to save inference results')
    
    args = parser.parse_args()

    main(args)