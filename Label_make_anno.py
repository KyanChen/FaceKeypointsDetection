import pandas as pd
import os
import cv2
import numpy as np

# 1.解析数据
# 2.建立训练集测试集数据按7/3的比例


class Data_annotation:
    def __init__(self, root):
        self.root = root

    def get_faces_keypoints(self):

        FOLDER = ['I', 'II']
        tmp_lines = []
        DATA_info = {'path': [], 'face_rect': [], 'face_keypoints': []}

        for f in FOLDER:
            DATA_DIR = os.path.join(self.root, f)
            file_path = os.path.join(DATA_DIR, 'label.txt')
            with open(file_path) as file:
                lines = file.readlines()
            # map(lambda x, y: x + y, [1, 3, 5, 7, 9], [2, 4, 6, 8, 10])

            tmp_lines.extend(list(map(lambda x: os.path.join(DATA_DIR, x), lines)))

        for file in tmp_lines:

            file = file.strip().split()
            try:
                cv2.imdecode(np.fromfile(file[0], dtype=np.uint8), cv2.IMREAD_COLOR)
            except OSError:
                pass
            else:
                DATA_info['path'].append(file[0])
                DATA_info['face_rect'].append(list(map(float, file[1:5])))
                DATA_info['face_keypoints'].append(list(map(float, file[5:])))

        ANNOTATION = pd.DataFrame(DATA_info)
        ANNOTATION.to_csv('face_keypoints_annotation.csv')
        print('face_keypoints_annotation file is saved.')

    def get_train_val_data(self):
        FILE = 'face_keypoints_annotation.csv'
        DATA_info = pd.read_csv(FILE)
        DATA_info_anno = {'path': [], 'rect': [], 'points': []}

        expand_ratio = 0.2
        self.get_valid_data(DATA_info, DATA_info_anno, expand_ratio)
        data = pd.DataFrame(DATA_info_anno)
        # numpy choice, sample
        train_data = data.sample(frac=0.8, replace=False)

        val_data = data[data['path'].isin(list(set(data['path'])-set(train_data['path'])))]

        data.to_csv('train_data.csv')
        data.to_csv('val_data.csv')
        print("train_data:{:d}".format(train_data.shape[0]))
        print("val_data:{:d}".format(val_data.shape[0]))

    def get_valid_data(self, DATA_info, DATA_info_anno, expand_ratio=0.25):
        def expand_roi(rect , expand_ratio):
            left, top, right, bottom = rect
            rect_width = right-left
            rect_height = bottom-top
            left, top, right, bottom = left - expand_ratio * rect_width, \
                                       top - expand_ratio * rect_height,\
                                       right + expand_ratio * rect_width, \
                                       bottom + expand_ratio * rect_height
            return [left, top, right, bottom]

        for index, row in DATA_info.iterrows():
            is_invalid_sample = False
            img = cv2.imdecode(np.fromfile(row.loc['path'], dtype=np.uint8), cv2.IMREAD_COLOR)
            height, width, _ = img.shape
            # "[229.0, 38.0, 289.0, 99.0]"
            rect = list(map(lambda x: float(x), eval(row.loc['face_rect'])))
            rect = expand_roi(rect, expand_ratio)
            points = list(map(float, eval(row.loc['face_keypoints'])))
            x = points[0::2]
            y = points[1::2]
            points_zip = list(zip(x, y))
            # 处理Rect不超出图像边界
            # [left, top, right, bottom]
            rect_dstxy = [0 if i < 0 else i for i in rect]
            rect_dstx = [width if rect_dstxy[i] > width else rect_dstxy[i] for i in [0, 2]]
            rect_dsty = [height if rect_dstxy[i] > height else rect_dstxy[i] for i in [1, 3]]
            rect_dstx.extend(rect_dsty)
            # [left, right, top, bottom]
            rect = [rect_dstx[i] for i in [0, 2, 1, 3]]
            # 处理Points不超出Rect边界,如果超出则舍去该样本
            left, top, right, bottom = rect
            for point in points_zip:
                x, y = point  # (x,y)
                if x < left or x > right or y < top or y > bottom:
                    is_invalid_sample = True
                    print("{:s}:Points is out of rect boundary".format(row.loc['path']))
                    break
            if is_invalid_sample:
                continue
            DATA_info_anno['path'].append(row.loc['path'])
            DATA_info_anno['rect'].append(rect)
            DATA_info_anno['points'].append(points_zip)


if __name__ == '__main__':
    ROOTS = r'F:\DataSet\慧科\人脸关键点检测'
    data_anno = Data_annotation(ROOTS)
    data_anno.get_faces_keypoints()
    data_anno.get_train_val_data()


