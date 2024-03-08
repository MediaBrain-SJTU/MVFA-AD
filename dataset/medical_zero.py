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


class MedTrainDataset(Dataset):
    def __init__(self,
                 dataset_path='/data/',
                 class_name='Brain',
                 resize=240,
                 batch_size = 1
                 ):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)

        self.dataset_path = dataset_path
        self.class_name = class_name
        self.resize = resize
        self.batch_size = batch_size

        # load dataset
        self.data = self.load_dataset_folder()

        # set transforms
        self.transform_x = transforms.Compose([
            transforms.Resize((resize,resize), Image.BICUBIC),
            transforms.ToTensor(),
        ])

        self.transform_mask = transforms.Compose(
            [transforms.Resize((resize,resize), Image.NEAREST),
             transforms.ToTensor()])

            

    def __getitem__(self, idx):
        x, y, mask, seg_idx = self.data[idx]
        y = torch.Tensor(y)

        batch = len(x)
        batch_img = []
        for i in range(batch):
            img = Image.open(x[i]).convert('RGB')
            img = self.transform_x(img)
            batch_img.append(img.unsqueeze(0))
        batch_img = torch.cat(batch_img)

        if seg_idx < 0:
            return batch_img, y, torch.zeros([1, self.resize, self.resize]), seg_idx
        
        batch_mask = []
        for i in range(batch):
            if mask[i] is None:
                batch_mask.append(torch.zeros([1, 1, self.resize, self.resize]))
            else:
                single_mask = Image.open(mask[i]).convert('L')
                single_mask = self.transform_mask(single_mask)
                batch_mask.append(single_mask.unsqueeze(0))
        batch_mask = torch.cat(batch_mask)

        return batch_img, y, batch_mask, seg_idx

    def __len__(self):
        return len(self.data)

    def load_dataset_folder(self):
        data_img = {}
        for class_name_one in CLASS_NAMES:
            if class_name_one != self.class_name:
                data_img[class_name_one] = []
                img_dir = os.path.join(self.dataset_path, f'{class_name_one}_AD/test/good/img')
                for f in os.listdir(img_dir):
                    data_img[class_name_one].append((os.path.join(img_dir, f), 0))

                img_dir = os.path.join(self.dataset_path, f'{class_name_one}_AD/test/Ungood/img')
                for f in os.listdir(img_dir):
                    data_img[class_name_one].append((os.path.join(img_dir, f), 1))
                random.shuffle(data_img[class_name_one])

        data = []

        for class_name_one in data_img.keys():
            if CLASS_INDEX[class_name_one] < 0:
                for image_index in range(0, len(data_img[class_name_one]), self.batch_size):
                    file_path = []
                    img_label = []
                    for batch_count in range(0, self.batch_size):
                        if image_index + batch_count >= len(data_img[class_name_one]):
                            break
                        file_path.append(data_img[class_name_one][image_index + batch_count][0])
                        img_label.append(data_img[class_name_one][image_index + batch_count][1])
                    data.append([file_path, img_label, None, CLASS_INDEX[class_name_one]])
            else:
                for image_index in range(0, len(data_img[class_name_one]), self.batch_size):
                    file_path = []
                    img_label = []
                    gt_path = []
                    for batch_count in range(0, self.batch_size):
                        if image_index + batch_count >= len(data_img[class_name_one]):
                            break
                        single_file_path = data_img[class_name_one][image_index + batch_count][0]
                        single_label = data_img[class_name_one][image_index + batch_count][1]

                        file_path.append(single_file_path)
                        img_label.append(single_label)

                        if single_label == 0:
                            gt_path.append(None)
                        else:
                            gt_path.append(single_file_path.replace('img', 'anomaly_mask'))
                    data.append([file_path, img_label, gt_path, CLASS_INDEX[class_name_one]])
        random.shuffle(data)
        return data


    def shuffle_dataset(self):
        data_img = {}
        for class_name_one in CLASS_NAMES:
            if class_name_one != self.class_name:
                data_img[class_name_one] = []
                img_dir = os.path.join(self.dataset_path, f'{class_name_one}_AD/test/good/img')
                for f in os.listdir(img_dir):
                    data_img[class_name_one].append((os.path.join(img_dir, f), 0))

                img_dir = os.path.join(self.dataset_path, f'{class_name_one}_AD/test/Ungood/img')
                for f in os.listdir(img_dir):
                    data_img[class_name_one].append((os.path.join(img_dir, f), 1))
                random.shuffle(data_img[class_name_one])
        data = []
        for class_name_one in data_img.keys():
            if CLASS_INDEX[class_name_one] < 0:
                for image_index in range(0, len(data_img[class_name_one]), self.batch_size):
                    file_path = []
                    img_label = []
                    for batch_count in range(0, self.batch_size):
                        if image_index + batch_count >= len(data_img[class_name_one]):
                            break
                        file_path.append(data_img[class_name_one][image_index + batch_count][0])
                        img_label.append(data_img[class_name_one][image_index + batch_count][1])
                    data.append([file_path, img_label, None, CLASS_INDEX[class_name_one]])
            else:
                for image_index in range(0, len(data_img[class_name_one]), self.batch_size):
                    file_path = []
                    img_label = []
                    gt_path = []
                    for batch_count in range(0, self.batch_size):
                        if image_index + batch_count >= len(data_img[class_name_one]):
                            break
                        single_file_path = data_img[class_name_one][image_index + batch_count][0]
                        single_label = data_img[class_name_one][image_index + batch_count][1]

                        file_path.append(single_file_path)
                        img_label.append(single_label)

                        if single_label == 0:
                            gt_path.append(None)
                        else:
                            gt_path.append(single_file_path.replace('img', 'anomaly_mask'))
                    data.append([file_path, img_label, gt_path, CLASS_INDEX[class_name_one]])
        random.shuffle(data)
        self.data = data




class MedTestDataset(Dataset):
    def __init__(self,
                 dataset_path='/data/',
                 class_name='Brain',
                 resize=240
                 ):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)

        self.dataset_path = os.path.join(dataset_path, f'{class_name}_AD')
        self.class_name = class_name
        self.seg_flag = CLASS_INDEX[class_name]
        self.resize = resize

        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder(self.seg_flag)

        # set transforms
        self.transform_x = transforms.Compose([
            transforms.Resize((resize,resize), Image.BICUBIC),
            transforms.ToTensor(),
        ])

        self.transform_mask = transforms.Compose(
            [transforms.Resize((resize,resize), Image.NEAREST),
             transforms.ToTensor()])


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

        normal_img_dir = os.path.join(self.dataset_path, 'test/good/img')
        img_fpath_list = sorted([os.path.join(normal_img_dir, f) for f in os.listdir(normal_img_dir)])
        x.extend(img_fpath_list)
        y.extend([0] * len(img_fpath_list))
        mask.extend([None] * len(img_fpath_list))

        abnorm_img_dir = os.path.join(self.dataset_path, 'test/Ungood/img')
        img_fpath_list = sorted([os.path.join(abnorm_img_dir, f) for f in os.listdir(abnorm_img_dir)])
        x.extend(img_fpath_list)
        y.extend([1] * len(img_fpath_list))

        if seg_flag > 0:
            gt_type_dir = os.path.join(self.dataset_path, 'test/Ungood/anomaly_mask')
            gt_fpath_list = sorted([os.path.join(gt_type_dir, f) for f in os.listdir(gt_type_dir)])
            mask.extend(gt_fpath_list)
        else:
            mask.extend([None] * len(img_fpath_list))

        assert len(x) == len(y), 'number of x and y should be same'
        return list(x), list(y), list(mask)

