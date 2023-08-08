import torch
from torch import nn
import torch.nn.functional as F


class AnnoGenerator(nn.Module):

    def __init__(self):
        super(AnnoGenerator, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 64, 7, 1, 3)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)

        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)

        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)

        self.conv4_1 = nn.Conv2d(256, 256, 3, 1, 2, 2)
        self.conv4_2 = nn.Conv2d(256, 256, 3, 1, 2, 2)
        self.conv4_3 = nn.Conv2d(256, 256, 3, 1, 2, 2)

        self.conv5_1 = nn.Conv2d(512, 256, 3, 1, 1)
        self.conv5_2 = nn.Conv2d(256, 256, 3, 1, 1)

        self.conv6_1 = nn.Conv2d(384, 128, 3, 1, 1)
        self.conv6_2 = nn.Conv2d(128, 128, 3, 1, 1)

        self.conv7_1 = nn.Conv2d(192, 128, 3, 1, 1)
        self.conv7_2 = nn.Conv2d(128, 1, 3, 1, 1)

        self.relu = nn.ReLU()
        self.max_pooling = nn.MaxPool2d(3, 2, 1)

    def forward(self, x):
        x1_1 = self.relu(self.conv1_1(x))
        x1_2 = self.relu(self.conv1_2(x1_1))

        x2_1 = self.relu(self.conv2_1(self.max_pooling(x1_2)))
        x2_2 = self.relu(self.conv2_2(x2_1))

        x3_1 = self.relu(self.conv3_1(self.max_pooling(x2_2)))
        x3_2 = self.relu(self.conv3_2(x3_1))

        x4_1 = self.relu(self.conv4_1(self.max_pooling(x3_2)))
        x4_2 = self.relu(self.conv4_2(x4_1))
        x4_3 = self.relu(self.conv4_3(x4_2))

        x5_1 = self.relu(self.conv5_1(torch.cat([F.interpolate(x4_3, (64, 64)), x3_2], dim=1)))
        x5_2 = self.relu(self.conv5_2(x5_1))

        x6_1 = self.relu(self.conv6_1(torch.cat([F.interpolate(x5_2, (128, 128)), x2_2], dim=1)))
        x6_2 = self.relu(self.conv6_2(x6_1))

        x7_1 = self.relu(self.conv7_1(torch.cat([F.interpolate(x6_2, (256, 256)), x1_2], dim=1)))
        x7_2 = self.conv7_2(x7_1)

        return x7_2


class Teacher(nn.Module):

    def __init__(self):
        super(Teacher, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 64, 7, 1, 3)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)

        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)

        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)

        self.conv4_1 = nn.Conv2d(256, 256, 3, 1, 2, 2)
        self.conv4_2 = nn.Conv2d(256, 256, 3, 1, 2, 2)
        self.conv4_3 = nn.Conv2d(256, 256, 3, 1, 2, 2)

        self.conv5_1 = nn.Conv2d(512, 256, 3, 1, 1)
        self.conv5_2 = nn.Conv2d(256, 256, 3, 1, 1)

        self.conv6_1 = nn.Conv2d(384, 128, 3, 1, 1)
        self.conv6_2 = nn.Conv2d(128, 128, 3, 1, 1)

        self.conv7_1 = nn.Conv2d(192, 128, 3, 1, 1)
        self.conv7_2 = nn.Conv2d(128, 1, 3, 1, 1)

        self.conv8_1 = nn.Conv2d(256, 256, 1, 1, 0)
        self.conv8_2 = nn.Conv2d(256, 256, 1, 1, 0)

        self.relu = nn.ReLU()
        self.max_pooling = nn.MaxPool2d(3, 2, 1)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x1_1 = self.relu(self.conv1_1(x))
        x1_2 = self.relu(self.conv1_2(x1_1))

        x2_1 = self.relu(self.conv2_1(self.max_pooling(x1_2)))
        x2_2 = self.relu(self.conv2_2(x2_1))

        x3_1 = self.relu(self.conv3_1(self.max_pooling(x2_2)))
        x3_2 = self.relu(self.conv3_2(x3_1))

        x4_1 = self.relu(self.conv4_1(self.max_pooling(x3_2)))
        x4_2 = self.relu(self.conv4_2(x4_1))
        x4_3 = self.relu(self.conv4_3(x4_2))

        x5_1 = self.relu(self.conv5_1(torch.cat([F.interpolate(x4_3, (64, 64)), x3_2], dim=1)))
        x5_2 = self.relu(self.conv5_2(x5_1))

        x6_1 = self.relu(self.conv6_1(torch.cat([F.interpolate(x5_2, (128, 128)), x2_2], dim=1)))
        x6_2 = self.relu(self.conv6_2(x6_1))

        x7_1 = self.relu(self.conv7_1(torch.cat([F.interpolate(x6_2, (256, 256)), x1_2], dim=1)))
        x7_2 = self.conv7_2(x7_1)

        x8_1 = self.relu(self.conv8_1(x4_3))
        x8_2 = self.conv8_2(x8_1)

        return x7_2, x8_2


class Student(nn.Module):
    def __init__(self, z_size=16):
        super(Student, self).__init__()
        self.conv1_1 = nn.Conv3d(1, 32, (1, 7, 7), 1, (0, 3, 3))
        self.conv1_2 = nn.Conv3d(32, 32, (1, 3, 3), 1, (0, 1, 1))

        self.conv2_1 = nn.Conv3d(32, 64, (1, 3, 3), 1, (0, 1, 1))
        self.conv2_2 = nn.Conv3d(64, 64, (1, 3, 3), 1, (0, 1, 1))

        self.conv3_1 = nn.Conv3d(64, 128, (1, 3, 3), 1, (0, 1, 1))
        self.conv3_2 = nn.Conv3d(128, 128, (1, 3, 3), 1, (0, 1, 1))

        self.conv4_1 = nn.Conv3d(128, 128, (3, 3, 3), 1, (1, 2, 2), (1, 2, 2))
        self.conv4_2 = nn.Conv3d(128, 128, (3, 3, 3), 1, (1, 2, 2), (1, 2, 2))
        self.conv4_3 = nn.Conv3d(128, 128, (3, 3, 3), 1, (1, 2, 2), (1, 2, 2))

        self.conv5_1 = nn.Conv3d(256, 128, (1, 3, 3), 1, (0, 1, 1))
        self.conv5_2 = nn.Conv3d(128, 128, (1, 3, 3), 1, (0, 1, 1))

        self.conv6_1 = nn.Conv3d(192, 64, (1, 3, 3), 1, (0, 1, 1))
        self.conv6_2 = nn.Conv3d(64, 64, (1, 3, 3), 1, (0, 1, 1))

        self.conv7_1 = nn.Conv3d(96, 64, (1, 3, 3), 1, (0, 1, 1))
        self.conv7_2 = nn.Conv3d(64, 1, (1, 3, 3), 1, (0, 1, 1))

        self.relu = nn.ReLU()
        self.max_pooling = nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1))

    def forward(self, x):
        x = x.unsqueeze(1)

        x1_1 = self.relu(self.conv1_1(x))       # (32, z_size, 256, 256)
        x1_2 = self.relu(self.conv1_2(x1_1))    # (32, z_size, 256, 256)

        x2_1 = self.relu(self.conv2_1(self.max_pooling(x1_2)))  # (64, z_size, 256, 256)
        x2_2 = self.relu(self.conv2_2(x2_1))                    # (64, z_size, 256, 256)

        x3_1 = self.relu(self.conv3_1(self.max_pooling(x2_2)))  # (128, 16, 64, 64)
        x3_2 = self.relu(self.conv3_2(x3_1))                    # (128, 16, 64, 64)

        x4_1 = self.relu(self.conv4_1(self.max_pooling(x3_2)))  # (128, 16, 32, 32)
        x4_2 = self.relu(self.conv4_2(x4_1))                    # (128, 16, 32, 32)
        x4_3 = self.relu(self.conv4_3(x4_2))                    # (128, 16, 32, 32)

        x5_1 = self.relu(self.conv5_1(torch.cat([F.interpolate(x4_3, (16, 64, 64)), x3_2], dim=1)))
        x5_2 = self.relu(self.conv5_2(x5_1))

        x6_1 = self.relu(self.conv6_1(torch.cat([F.interpolate(x5_2, (16, 128, 128)), x2_2], dim=1)))
        x6_2 = self.relu(self.conv6_2(x6_1))

        x7_1 = self.relu(self.conv7_1(torch.cat([F.interpolate(x6_2, (16, 256, 256)), x1_2], dim=1)))
        x7_2 = self.conv7_2(x7_1)
        
        x_out = x7_2.squeeze(1)
        return x_out


class StudentKD(nn.Module):
    def __init__(self, z_size=16):
        super(StudentKD, self).__init__()
        self.conv1_1 = nn.Conv3d(1, 32, (1, 7, 7), 1, (0, 3, 3))
        self.conv1_2 = nn.Conv3d(32, 32, (1, 3, 3), 1, (0, 1, 1))

        self.conv2_1 = nn.Conv3d(32, 64, (1, 3, 3), 1, (0, 1, 1))
        self.conv2_2 = nn.Conv3d(64, 64, (1, 3, 3), 1, (0, 1, 1))

        self.conv3_1 = nn.Conv3d(64, 128, (1, 3, 3), 1, (0, 1, 1))
        self.conv3_2 = nn.Conv3d(128, 128, (1, 3, 3), 1, (0, 1, 1))

        self.conv4_1 = nn.Conv3d(128, 128, (3, 3, 3), 1, (1, 2, 2), (1, 2, 2))
        self.conv4_2 = nn.Conv3d(128, 128, (3, 3, 3), 1, (1, 2, 2), (1, 2, 2))
        self.conv4_3 = nn.Conv3d(128, 128, (3, 3, 3), 1, (1, 2, 2), (1, 2, 2))

        self.conv5_1 = nn.Conv3d(256, 128, (1, 3, 3), 1, (0, 1, 1))
        self.conv5_2 = nn.Conv3d(128, 128, (1, 3, 3), 1, (0, 1, 1))

        self.conv6_1 = nn.Conv3d(192, 64, (1, 3, 3), 1, (0, 1, 1))
        self.conv6_2 = nn.Conv3d(64, 64, (1, 3, 3), 1, (0, 1, 1))

        self.conv7_1 = nn.Conv3d(96, 64, (1, 3, 3), 1, (0, 1, 1))
        self.conv7_2 = nn.Conv3d(64, 1, (1, 3, 3), 1, (0, 1, 1))

        self.conv8_1 = nn.Conv3d(384, 384, (1, 1, 1))
        self.conv8_2 = nn.Conv3d(384, 384, (1, 1, 1))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5, inplace=False)
        self.max_pooling = nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1))
        self.avg_pooling = nn.AvgPool3d((1, 2, 2), (1, 2, 2))

    def forward(self, x):
        x = x.unsqueeze(1)

        x1_1 = self.relu(self.conv1_1(x))       # (32, z_size, 256, 256)
        x1_2 = self.relu(self.conv1_2(x1_1))    # (32, z_size, 256, 256)

        x2_1 = self.relu(self.conv2_1(self.max_pooling(x1_2)))  # (64, z_size, 256, 256)
        x2_2 = self.relu(self.conv2_2(x2_1))                    # (64, z_size, 256, 256)

        x3_1 = self.relu(self.conv3_1(self.max_pooling(x2_2)))  # (128, 16, 64, 64)
        x3_2 = self.relu(self.conv3_2(x3_1))                    # (128, 16, 64, 64)

        x4_1 = self.relu(self.conv4_1(self.max_pooling(x3_2)))  # (128, 16, 32, 32)
        x4_2 = self.relu(self.conv4_2(x4_1))                    # (128, 16, 32, 32)
        x4_3 = self.relu(self.conv4_3(x4_2))                    # (128, 16, 32, 32)

        x5_1 = self.relu(self.conv5_1(torch.cat([F.interpolate(x4_3, (16, 64, 64)), x3_2], dim=1)))
        x5_2 = self.relu(self.conv5_2(x5_1))

        x6_1 = self.relu(self.conv6_1(torch.cat([F.interpolate(x5_2, (16, 128, 128)), x2_2], dim=1)))
        x6_2 = self.relu(self.conv6_2(x6_1))

        x7_1 = self.relu(self.conv7_1(torch.cat([F.interpolate(x6_2, (16, 256, 256)), x1_2], dim=1)))
        x7_2 = self.conv7_2(x7_1)
        
        x_out = x7_2.squeeze(1)

        feat_concat = torch.cat([
            x4_3, self.avg_pooling(x5_2), self.avg_pooling(self.avg_pooling(x6_2)),
            self.avg_pooling(self.avg_pooling(self.avg_pooling(x7_1)))
        ], dim=1)
        feat_concat = self.dropout(feat_concat)

        x8_1 = self.relu(self.conv8_1(feat_concat))
        x8_2 = self.conv8_2(x8_1)   # (256, 16, 32, 32)
        x_feat = x8_2
        return x_out, x_feat


class StudentKDFeat(nn.Module):
    def __init__(self, z_size=16):
        super(StudentKDFeat, self).__init__()
        self.conv1_1 = nn.Conv3d(1, 32, (1, 7, 7), 1, (0, 3, 3))
        self.conv1_2 = nn.Conv3d(32, 32, (1, 3, 3), 1, (0, 1, 1))

        self.conv2_1 = nn.Conv3d(32, 64, (1, 3, 3), 1, (0, 1, 1))
        self.conv2_2 = nn.Conv3d(64, 64, (1, 3, 3), 1, (0, 1, 1))

        self.conv3_1 = nn.Conv3d(64, 128, (1, 3, 3), 1, (0, 1, 1))
        self.conv3_2 = nn.Conv3d(128, 128, (1, 3, 3), 1, (0, 1, 1))

        self.conv4_1 = nn.Conv3d(128, 128, (3, 3, 3), 1, (1, 2, 2), (1, 2, 2))
        self.conv4_2 = nn.Conv3d(128, 128, (3, 3, 3), 1, (1, 2, 2), (1, 2, 2))
        self.conv4_3 = nn.Conv3d(128, 128, (3, 3, 3), 1, (1, 2, 2), (1, 2, 2))

        self.conv5_1 = nn.Conv3d(256, 128, (1, 3, 3), 1, (0, 1, 1))
        self.conv5_2 = nn.Conv3d(128, 128, (1, 3, 3), 1, (0, 1, 1))

        self.conv6_1 = nn.Conv3d(192, 64, (1, 3, 3), 1, (0, 1, 1))
        self.conv6_2 = nn.Conv3d(64, 64, (1, 3, 3), 1, (0, 1, 1))

        self.conv7_1 = nn.Conv3d(96, 64, (1, 3, 3), 1, (0, 1, 1))
        self.conv7_2 = nn.Conv3d(64, 1, (1, 3, 3), 1, (0, 1, 1))

        self.conv8_1 = nn.Conv3d(256, 256, (1, 1, 1))
        self.conv8_2 = nn.Conv3d(256, 384, (1, 1, 1))

        self.relu = nn.ReLU()
        self.max_pooling = nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1))
        self.avg_pooling = nn.AvgPool3d((1, 2, 2), (1, 2, 2))

    def forward(self, x):
        x = x.unsqueeze(1)

        x1_1 = self.relu(self.conv1_1(x))       # (32, z_size, 256, 256)
        x1_2 = self.relu(self.conv1_2(x1_1))    # (32, z_size, 256, 256)

        x2_1 = self.relu(self.conv2_1(self.max_pooling(x1_2)))  # (64, z_size, 256, 256)
        x2_2 = self.relu(self.conv2_2(x2_1))                    # (64, z_size, 256, 256)

        x3_1 = self.relu(self.conv3_1(self.max_pooling(x2_2)))  # (128, 16, 64, 64)
        x3_2 = self.relu(self.conv3_2(x3_1))                    # (128, 16, 64, 64)

        x4_1 = self.relu(self.conv4_1(self.max_pooling(x3_2)))  # (128, 16, 32, 32)
        x4_2 = self.relu(self.conv4_2(x4_1))                    # (128, 16, 32, 32)
        x4_3 = self.relu(self.conv4_3(x4_2))                    # (128, 16, 32, 32)

        x5_1 = self.relu(self.conv5_1(torch.cat([F.interpolate(x4_3, (16, 64, 64)), x3_2], dim=1)))
        x5_2 = self.relu(self.conv5_2(x5_1))

        x6_1 = self.relu(self.conv6_1(torch.cat([F.interpolate(x5_2, (16, 128, 128)), x2_2], dim=1)))
        x6_2 = self.relu(self.conv6_2(x6_1))

        x7_1 = self.relu(self.conv7_1(torch.cat([F.interpolate(x6_2, (16, 256, 256)), x1_2], dim=1)))
        x7_2 = self.conv7_2(x7_1)
        
        x_out = x7_2.squeeze(1)

        x8_1 = self.relu(self.conv8_1(torch.cat([self.avg_pooling(x5_2), x4_3], dim=1)))
        x8_2 = self.conv8_2(x8_1)   # (384, 16, 32, 32)
        x_feat = x8_2
        return x_out, x_feat

class ViTHead(nn.Module):

    def __init__(self, upsample_scale=8):
        super(ViTHead, self).__init__()
        self.conv1 = nn.Conv2d(384, 64, 1, 1)
        self.conv2 = nn.Conv2d(64, 1, 1, 1)
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=upsample_scale)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.upsample(self.conv2(x))

        return x
