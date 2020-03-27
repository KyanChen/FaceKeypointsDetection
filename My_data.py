import numpy as np
import cv2
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import os
import pandas as pd
import matplotlib.pyplot as plt

train_boarder = 112


class FaceLandmarksDataset(Dataset):
    def __init__(self, data_file, transform=None):
        """
        :param src_lines: src_lines
        :param train: whether we are training or not
        :param transform: data transform
        """
        # 类内变量
        self.transform = transform
        if not os.path.exists(data_file):
            print(data_file+"does not exist!")
        self.file_info = pd.read_csv(data_file, index_col=0)
        # 增加一列为正样本，人脸标签为1
        self.file_info['class'] = 1
        # 每一个正样本，生成二个负样本
        self.negative_samples = self.get_negative_samples(2)
        self.file_info = pd.concat([self.file_info, self.negative_samples])
        # self.file_info.to_csv("test.csv")
        # self.file_info = pd.read_csv("test.csv")
        self.size = len(self.file_info)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        data = self.file_info.iloc[idx]
        img_name = data['path']
        rect = np.array(eval(data['rect']), dtype=np.int)
        points = eval(data['points'])
        class_ = data['class']
        # image
        img = cv2.imdecode(np.fromfile(img_name, dtype=np.uint8), cv2.IMREAD_COLOR)

        img_crop = img[rect[1]:rect[3], rect[0]:rect[2], :]  # this is also good, but has some shift already
        if class_ == 1:
            landmarks = np.array(points).astype(np.float32)
            # [0, 1]左上点
            landmarks = landmarks - rect[0:2]
        else:
            landmarks = np.zeros((21, 2), dtype=np.float32)
        sample = {'image': img_crop, 'landmarks': landmarks, 'label': class_}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_negative_samples(self, negative_num):
        def get_iou(rect, rects):
            LT = np.maximum(rect[:2], rects[:, :2])
            RB = np.maximum(rect[2:], rects[:, 2:])
            overlap_wh = RB - LT
            overlap_wh[overlap_wh < 0] = 0
            intersection = overlap_wh[:, 0] * overlap_wh[:, 1]
            area_rect = (rect[2] - rect[0]) * (rect[3] - rect[1])
            area_rects = (rects[:, 2] - rects[:, 0]) * (rects[:, 3] - rects[:, 1])
            t = area_rect + area_rects - intersection
            iou_ = intersection / (1e-10 + area_rect + area_rects - intersection)
            return iou_

        def is_inclusion_relation(rect, rects):
            flag_w = rect[:2] > rects[:, :2]
            flag_h = rect[2:] < rects[:, 2:]
            flag_wh = np.concatenate((flag_w, flag_h), axis=1)
            return np.any(np.all(flag_wh, axis=1))

        negative_data_info = {'path': [], 'rect': []}
        for index, rows_data in self.file_info.iterrows():
            img_path = rows_data['path']
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            height, width, _ = img.shape
            # 将一张照片中的每个人脸都拿出来
            rects_in_same_img_dataframe = self.file_info[self.file_info['path'] == img_path]

            rects = []
            for index, rect_data in rects_in_same_img_dataframe.iterrows():
                rects += eval(rect_data['rect'])
            rects = np.array(rects).astype(int).reshape(-1, 4)
            wh = rects[:, 2:] - rects[:, 0:2]
            max_wh = np.max(wh, 0)
            min_wh = np.min(wh, 0)

            # 如果尝试100次还没有找到合适的negative rect则放弃
            try_times_threshold = 200
            gen_valid_rect_num = 0
            for _ in range(try_times_threshold):
                gen_w = np.random.randint(max(0.5 * min_wh[0], 2) - 1, max_wh[0])
                gen_h = np.random.randint(max(0.5 * min_wh[1], 2) - 1, max_wh[1])
                if gen_w / gen_h < 6/10 or gen_w / gen_h > 10/6:
                    continue
                gen_left = np.random.randint(0, width-gen_w)
                gen_top = np.random.randint(0, height-gen_h)

                gen_right = gen_left + gen_w
                gen_bottom = gen_top + gen_h
                gen_rect = [gen_left, gen_top, gen_right, gen_bottom]

                iou = get_iou(np.array(gen_rect), rects)
                if np.any(iou > 0.4):
                    continue
                if is_inclusion_relation(np.array(gen_rect), rects):
                    continue
                gen_valid_rect_num += 1
                if gen_valid_rect_num > negative_num:
                    break

                negative_data_info['path'].append(rows_data['path'])
                negative_data_info['rect'].append(str(gen_rect))
                # img_rect = img[gen_rect[1]: gen_rect[3], gen_rect[0]: gen_rect[2], :]
                # plt.imshow(img_rect)
                # plt.show()
        data = pd.DataFrame(negative_data_info)
        data['points'] = str([0, 0])
        data['class'] = 0
        return data


class Normalize(object):
    """
        Resieze to train_boarder x train_boarder. Here we use 112 x 112
    """
    def __call__(self, sample):
        img, landmarks, label = sample['image'], sample['landmarks'], sample['label']
        height, width, _ = img.shape
        img_resize = cv2.resize(img, (train_boarder, train_boarder))
        if label:
            landmarks[:, 0] = landmarks[:, 0] * train_boarder / width
            landmarks[:, 1] = landmarks[:, 1] * train_boarder / height
        return {'image': img_resize, 'landmarks': landmarks, 'label': label}


class RandomHorizontalFlip(object):
    """
        Horizontally flip image randomly with given probability
        Args:
            p (float): probability of the image being flipped.
                       Default value = 0.5
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        img, landmarks, label = sample['image'], sample['landmarks'], sample['label']

        if np.random.random() < self.p:
            img = img[:, ::-1].copy()
            if label:
                landmarks[:, 0] = train_boarder - landmarks[:, 0]
        return {'image': img, 'landmarks': landmarks, 'label': label}


class RandomRotate(object):
    """
        Randomly rotate image within given limits
        Args:
            p (float): probability above which the image need to be flipped. Default value = 0.25
            rotate limits by default: [-20, 20]
    """
    def __init__(self, p=0.5, a=5):
        self.p = p
        self.angle = a

    def __call__(self, sample):
        img, landmarks, label = sample['image'], sample['landmarks'], sample['label']

        if np.random.random() > self.p:
            # angle
            limit = self.angle
            angle = np.random.randint(-limit, limit)
            height, width, _ = img.shape
            center = (width // 2, height // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (width, height))
            if label == 1:
                # landmarks
                landmarks_pair = np.insert(landmarks, obj=2, values=1, axis=1)
                rotated_landmarks = []
                for point in landmarks_pair:
                    rotated_landmark = np.matmul(M, point)
                    rotated_landmarks.append(rotated_landmark)
                landmarks = np.asarray(rotated_landmarks)
            img = np.asarray(img, dtype=np.float32)
        return {'image': img, 'landmarks': landmarks, 'label': label}


class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        Tensors channel sequence: N x C x H x W
        Then do channel normalization: (image - mean) / std_variation
    """
    def channel_norm(self, img):
        mean = np.mean(img)
        std = np.std(img)
        pixels = (img - mean) / (std + 0.0000001)
        return pixels

    def __call__(self, sample):
        img, landmarks, label = sample['image'], sample['landmarks'], sample['label']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = img / 255.0
        landmarks = landmarks
        img = img.transpose((2, 0, 1))
        return {'image': torch.from_numpy(img).float(),
                'landmarks': torch.from_numpy(landmarks.reshape(-1)).float(),
                'label': torch.from_numpy(np.array([label])).float()}


def get_train_val_data():
    train_file = 'train_data.csv'
    test_file = 'val_data.csv'
    tsfm_train = transforms.Compose([
        Normalize(),                # do channel normalization
        RandomHorizontalFlip(0.5),  # randomly flip image horizontally
        RandomRotate(0.25, 5),          # randomly rotate image
        ToTensor()]                 # convert to torch type: NxCxHxW
    )
    tsfm_test = transforms.Compose([
        Normalize(),
        ToTensor()
    ])
    train_dataset = FaceLandmarksDataset(train_file, transform=tsfm_train)
    test_dataset = FaceLandmarksDataset(test_file, transform=tsfm_test)
    return train_dataset, test_dataset


def _test_My_data():
    train_set, val_set = get_train_val_data()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(val_set, batch_size=256)
    data_loaders = {'train': train_loader, 'val': valid_loader}
    for i in range(0,10):
        sample = train_loader.dataset[i]
        img = Image.fromarray(sample['image'].astype('uint8'))
        points = sample['landmarks']
        class_ = sample['label']

        landmarks = points.astype('float').reshape(-1, 2)
        draw = ImageDraw.Draw(img)
        x = landmarks[:, 0]
        y = landmarks[:, 1]
        points_zip = list(zip(x, y))
        draw.point(points_zip, (255, 0, 0))
        # img.save(r'H:\DataSet\慧科\人脸关键点检测\result\{:d}.jpg'.format(index))
        plt.imshow(img)
        plt.show()

train_set, val_set = get_train_val_data()