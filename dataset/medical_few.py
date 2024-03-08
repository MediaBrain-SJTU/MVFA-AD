import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
import pandas as pd
import numpy as np

CLASS_NAMES = ['Brain', 'Liver', 'Retina_RESC', 'Retina_OCT2017', 'Chest', 'Histopathology']
CLASS_INDEX = {'Brain':3, 'Liver':2, 'Retina_RESC':1, 'Retina_OCT2017':-1, 'Chest':-2, 'Histopathology':-3}


class MedDataset(Dataset):
    def __init__(self,
                 dataset_path='/data/',
                 class_name='Brain',
                 resize=240,
                 shot = 4,
                 iterate = -1
                 ):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        assert shot>0, 'shot number : {}, should be positive integer'.format(shot)

        self.dataset_path = os.path.join(dataset_path, f'{class_name}_AD')
        self.resize = resize
        self.shot = shot
        self.iterate = iterate
        self.class_name = class_name
        self.seg_flag = CLASS_INDEX[class_name]

        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder(self.seg_flag)


        self.transform_x = transforms.Compose([
            transforms.Resize((resize,resize), Image.BICUBIC),
            transforms.ToTensor(),
            ])


        self.transform_mask = transforms.Compose([
            transforms.Resize((resize,resize), Image.NEAREST),
            transforms.ToTensor()
            ])


        self.fewshot_norm_img = self.get_few_normal()
        self.fewshot_abnorm_img, self.fewshot_abnorm_mask = self.get_few_abnormal()
        
            

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        x = Image.open(x).convert('RGB')
        x_img = self.transform_x(x)

        if self.seg_flag < 0:
            return x_img, y, torch.zeros([1, self.resize, self.resize])

        if mask is None:
            mask = torch.zeros([1, self.resize, self.resize])
            y = 0
        else:
            mask = Image.open(mask).convert('L')
            mask = self.transform_mask(mask)
            y = 1
        return x_img, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self, seg_flag):
        x, y, mask = [], [], []

        normal_img_dir = os.path.join(self.dataset_path, 'test', 'good', 'img')
        img_fpath_list = sorted([os.path.join(normal_img_dir, f) for f in os.listdir(normal_img_dir)])
        x.extend(img_fpath_list)
        y.extend([0] * len(img_fpath_list))
        mask.extend([None] * len(img_fpath_list))


        abnormal_img_dir = os.path.join(self.dataset_path, 'test', 'Ungood', 'img')
        img_fpath_list = sorted([os.path.join(abnormal_img_dir, f) for f in os.listdir(abnormal_img_dir)])
        x.extend(img_fpath_list)
        y.extend([1] * len(img_fpath_list))

        if self.seg_flag > 0:
            gt_fpath_list = [f.replace('img', 'anomaly_mask') for f in img_fpath_list]
            mask.extend(gt_fpath_list)
        else:
            mask.extend([None] * len(img_fpath_list))

        assert len(x) == len(y), 'number of x and y should be same'
        return list(x), list(y), list(mask)


    def get_few_normal(self):
        x = []
        img_dir = os.path.join(self.dataset_path, 'valid', 'good', 'img')
        normal_names = os.listdir(img_dir)

        # select images
        if self.iterate < 0:
            random_choice = random.sample(normal_names, self.shot)
        else:
            random_choice = []
            with open(f'./dataset/fewshot_seed/{self.class_name}/{self.shot}-shot.txt', 'r', encoding='utf-8') as infile:
                for line in infile:
                    data_line = line.strip("\n").split()  # 去除首尾换行符，并按空格划分
                    if data_line[0] == f'n-{self.iterate}:':
                        random_choice = data_line[1:]
                        break

        for f in random_choice:
            if f.endswith('.png') or f.endswith('.jpeg'):
                x.append(os.path.join(img_dir, f))

        fewshot_img = []
        for idx in range(self.shot):
            image = x[idx]
            image = Image.open(image).convert('RGB')
            image = self.transform_x(image)
            fewshot_img.append(image.unsqueeze(0))

        fewshot_img = torch.cat(fewshot_img)
        return fewshot_img


    def get_few_abnormal(self):
        x = []
        y = []
        img_dir = os.path.join(self.dataset_path, 'valid', 'Ungood', 'img')
        mask_dir = os.path.join(self.dataset_path, 'valid', 'Ungood', 'anomaly_mask')

        abnormal_names = os.listdir(img_dir)

        # select images
        if self.iterate < 0:
            random_choice = random.sample(abnormal_names, self.shot)
        else:
            random_choice = []
            with open(f'./dataset/fewshot_seed/{self.class_name}/{self.shot}-shot.txt', 'r', encoding='utf-8') as infile:
                for line in infile:
                    data_line = line.strip("\n").split()  # 去除首尾换行符，并按空格划分
                    if data_line[0] == f'a-{self.iterate}:':
                        random_choice = data_line[1:]
                        break

        for f in random_choice:
            if f.endswith('.png') or f.endswith('.jpeg'):
                x.append(os.path.join(img_dir, f))
                y.append(os.path.join(mask_dir, f))

        fewshot_img = []
        fewshot_mask = []
        for idx in range(self.shot):
            image = x[idx]
            image = Image.open(image).convert('RGB')
            image = self.transform_x(image)
            fewshot_img.append(image.unsqueeze(0))

            if CLASS_INDEX[self.class_name] > 0:
                image = y[idx]
                image = Image.open(image).convert('L')
                image = self.transform_mask(image)
                fewshot_mask.append(image.unsqueeze(0))

        fewshot_img = torch.cat(fewshot_img)

        if len(fewshot_mask) == 0:
            return fewshot_img, None
        else:
            fewshot_mask = torch.cat(fewshot_mask)
            return fewshot_img, fewshot_mask

