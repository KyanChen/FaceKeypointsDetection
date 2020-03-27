import os
import Config
import torch
from My_net import My_Net as Net
import numpy as np
import cv2
import matplotlib.pyplot as plt


def predict_img(model, test_img_name):
    if Config.MODEL_NAME:
        if not os.path.exists(Config.MODEL_NAME):
            print("No Model!")
            return
    img_src = cv2.imdecode(np.fromfile(test_img_name, dtype=np.uint8), cv2.IMREAD_COLOR)

    height, width, _ = img_src.shape
    img_src = cv2.resize(img_src, Config.NET_IMG_SIZE)
    img = img_src.transpose((2, 0, 1))
    img = img/255.0
    img = torch.from_numpy(img).unsqueeze(0).float()

    model.load_state_dict(torch.load(Config.MODEL_NAME))
    model.eval()
    output_pts, output_cls = model(img)
    pred_class = output_cls.argmax(dim=1, keepdim=True).squeeze()
    output_pts = output_pts.squeeze()
    print(pred_class)
    if pred_class:
        x = output_pts[::2]
        y = output_pts[1::2]
        points_zip = list(zip(x, y))
        for point in points_zip:
            cv2.circle(img_src, point, 1, (0, 0, 255), -1)
    img_result = Config.IMG_TO_PREDICT.replace('.png', '_test.jpg')
    cv2.imwrite(img_result, img_src)
    plt.imshow(img_src)
    plt.show()


def main():
    model = Net()
    test_img_name = Config.IMG_TO_PREDICT
    predict_img(model, test_img_name)


if __name__ == '__main__':
    main()

