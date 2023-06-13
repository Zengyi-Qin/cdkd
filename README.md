# Learning 3D Ultrasound Segmentation under Extreme Label Deficiency

## Environment Setup
Install [pytorch](https://pytorch.org/) following the official guide. Then install other dependencies:
```bash
pip install numpy opencv-python tqdm
```
Download the pretrained DINO model and the sample CETUS dataset from [this folder](https://drive.google.com/drive/folders/1tG0BHuyvPTaWIlSv-sWdJxh9Yh9u-Pic?usp=sharing) and unzip.

## Training and Inference
Train the teacher network using only 0.1% of the ground truth labels:
```bash 
python train_teacher.py --workers 32 --batchsize 32 --anno_ratio 0.001 --epochs 100
```

Use the teacher network to generate high-quality 3D labels:
```bash
python generate_pseudo_anno.py --sequence SEQ_PATH --method randaugopt --head VIT_HEAD_PATH
```

Train the student network on the generated labels:
```bash
python train_student.py --workers 32 --batchsize 16 --data data/cetus --anno_ratio 0.001 --epochs 100 --pseudo_method randaugopt
```

Use the student network to perform 3D segmentation:
```bash
python inference.py --method student_kd --pretrained STUDENT_CKPT --input_dir IMG_DIR --output_dir OUTPUT_DIR
```

