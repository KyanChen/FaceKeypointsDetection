import os
import Config
import torch
import torch.nn as nn
import numpy as np
import cv2
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from My_data import get_train_val_data
from My_net import My_Net as Net

if not os.path.exists(Config.RESULTS_LOG_PATH):
    os.makedirs(Config.RESULTS_LOG_PATH)
if not os.path.exists(os.path.join(Config.RESULTS_LOG_PATH, 'log')):
    os.makedirs(os.path.join(Config.RESULTS_LOG_PATH, 'log'))
if not os.path.exists(os.path.join(Config.RESULTS_LOG_PATH, 'train')):
    os.makedirs(os.path.join(Config.RESULTS_LOG_PATH, 'train'))
if not os.path.exists(os.path.join(Config.RESULTS_LOG_PATH, 'val')):
    os.makedirs(os.path.join(Config.RESULTS_LOG_PATH, 'val'))

writer = SummaryWriter(os.path.join(Config.RESULTS_LOG_PATH, 'log'))


def train(data_loaders, model, criterion):
    if Config.MODEL_SAVE_PATH:
        if not os.path.exists(Config.MODEL_SAVE_PATH):
            os.makedirs(Config.MODEL_SAVE_PATH)

    device = torch.device('cpu')
    if Config.DEVICE == 'gpu' and torch.cuda.is_available():
        device = torch.device('cuda')
    # 准则
    pts_criterion = criterion[0]
    cls_criterion = criterion[1]

    LR = Config.LEARNING_RATE
    opt_SGD = torch.optim.SGD(model.parameters(), lr=LR)
    opt_Momentum = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.8)
    opt_RMSprop = torch.optim.RMSprop(model.parameters(), lr=LR, alpha=0.9)
    opt_Adam = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    optimizers = {"opt_SGD": opt_SGD, "opt_Momentum": opt_Momentum, "opt_RMSprop": opt_RMSprop, "opt_Adam": opt_Adam}
    optimizer = optimizers[Config.OPTIMIRZER]

    if Config.FLAG_RESTORE_MODEL and os.path.exists(Config.MODEL_NAME):
        model.load_state_dict(torch.load(Config.MODEL_NAME, map_location=device))

    iter = {'train': 0, 'val': 0}
    for epoch_id in range(Config.EPOCH):
        # monitor training loss
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            for batch_idx, batch in enumerate(data_loaders[phase]):
                iter[phase] += 1
                img = batch['image'].to(device)
                landmark = batch['landmarks'].to(device)
                landmark.requires_grad = True
                label = batch['label'].to(device)
                # for i in range(label.size(0)):
                #    plt.imshow(draw_points(img, landmark, label, i)[:, :, (2, 1, 0)])
                #    plt.show()

                optimizer.zero_grad()
                # clear the gradients of all optimized variables

                with torch.set_grad_enabled(phase == 'train'):
                    output_pts, output_cls = model(img)

                    # due with positive samples
                    positive_mask = label == 1
                    positive_mask = np.squeeze(positive_mask)
                    len_true_positive = positive_mask.sum().item()
                    if len_true_positive == 0:
                        loss_positive = 0
                        pred_class_pos_correct_acc = 1.0
                        loss_positive_pts = 0
                        loss_positive_cls = 0
                        # print(len_true_positive)
                    else:
                        loss_positive_pts = pts_criterion(output_pts[positive_mask],
                                                          landmark[positive_mask])

                        # print(output_cls[positive_mask])
                        # print(label[positive_mask].view(-1).long())
                        loss_positive_cls = 50 * cls_criterion(
                            output_cls[positive_mask], label[positive_mask].view(-1).long())

                        loss_positive = loss_positive_pts + loss_positive_cls

                        positive_pred_class = output_cls[positive_mask].argmax(dim=1, keepdim=True)
                        # print(positive_pred_class)
                        pred_class_pos_correct_acc = \
                            positive_pred_class.eq(label[positive_mask]).sum().item() / len_true_positive

                    # due with negative samples (no coordinates)
                    negative_mask = label == 0
                    negative_mask = np.squeeze(negative_mask)
                    len_true_negative = negative_mask.sum().item()
                    if len_true_negative == 0:
                        loss_negative = 0
                        pred_class_neg_correct_acc = 1
                        # print(len_true_negative)
                    else:
                        # print("1:{}".format(output_cls[negative_mask]))
                        # print("2:{}".format(label[negative_mask].long()))
                        loss_negative_cls = cls_criterion(
                            output_cls[negative_mask], label[negative_mask].view(-1).long())
                        loss_negative = 50 * loss_negative_cls

                        negative_pred_cls = output_cls[negative_mask].argmax(dim=1, keepdim=True)
                        pred_class_neg_correct_acc = \
                            negative_pred_cls.eq(label[negative_mask]).sum().item() / len_true_negative

                    pred_class_acc = (pred_class_pos_correct_acc *
                                      len_true_positive + pred_class_neg_correct_acc * len_true_negative) / \
                                     (len_true_positive + len_true_negative)

                    # sum up
                    # print(loss_positive)
                    loss = loss_positive + 50 * loss_negative

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                if batch_idx % Config.LOG_INTERVAL == 0:
                    pred_class = output_cls.argmax(dim=1, keepdim=True)
                    index_img_eval = np.random.randint(0, len(img), size=3)
                    for k in index_img_eval:
                        img_draw = draw_points(img, output_pts, pred_class, k)
                        cv2.imwrite(Config.RESULTS_LOG_PATH
                                    + '/' + phase + '/' + str(epoch_id) +
                                    '_' + str(batch_idx) + '_' + str(k) + '.jpg', img_draw)

                    print('{} Epoch: {} iter: {}\t'
                          'Loss:{:.6f}\tl_p_pts:{:.6f}\tl_p_cls: {:.6f}\tl_n_cls: {:.6f}\t'
                          'Acc:{:.2f}\tP_acc:{:.2f}\tN_acc{:.2f}'.format(
                            phase,
                            epoch_id,
                            iter[phase],
                            # training losses: total loss, regression loss, classification
                            loss,
                            loss_positive_pts,
                            loss_positive_cls,
                            loss_negative_cls,
                            # training accuracy: positive samples in a batch
                            pred_class_acc,
                            pred_class_pos_correct_acc,
                            pred_class_neg_correct_acc))
                    writer.add_scalars('iter/%s_sub_loss' % phase,
                                       {'positive_pts': loss_positive_pts,
                                        'positive_cls': loss_positive_cls,
                                        'negative_cls': loss_negative_cls}, iter[phase])

                    writer.add_scalar('iter/%s_sum_loss' % phase,
                                      loss, iter[phase])
                    writer.add_scalars('iter/%s_accuracy' % phase,
                                       {'true_positive': pred_class_pos_correct_acc,
                                        'true_negative': pred_class_neg_correct_acc,
                                        'total_accuracy': pred_class_acc},
                                       iter[phase])

                if phase == 'train' and epoch_id % Config.SAVE_MODEL_INTERVAL == 0:
                    saved_model_name = os.path.join(Config.MODEL_SAVE_PATH, 'aligner_epoch' + '_' + str(epoch_id) + '.pt')
                    torch.save(model.state_dict(), saved_model_name)


def draw_points(imgs, landmarks, labels, index):
    img = imgs[index, :, :, :] * 255
    landmark = landmarks[index, :]

    img = img.contiguous().cpu().numpy().transpose((1, 2, 0)).astype(np.uint8)
    # 放大
    img = cv2.resize(img, (112 * 4, 112 * 4))
    landmark *= 4
    # print(img_.flags)
    if labels[index]:
        x = landmark[::2].cpu().detach().numpy()
        y = landmark[1::2].cpu().detach().numpy()
        points_zip = list(zip(x, y))
        for point in points_zip:
            cv2.circle(img, point, 3, (0, 0, 255), -1)
    return img


def main():
    print('====> Loading Datasets')
    train_set, val_set = get_train_val_data()
    # train_sampler = SubsetRandomSampler()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=Config.BATCH_TRAIN, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(val_set, batch_size=Config.BATCH_VAL)
    data_loaders = {'train': train_loader, 'val': valid_loader}

    print('===> Building Model')
    device = torch.device('cpu')
    if Config.DEVICE == 'gpu' and torch.cuda.is_available():
        device = torch.device('cuda')
    # For single GPU
    model = Net().to(device)

    criterion_pts = nn.MSELoss()
    weights = [1, 3]
    class_weights = torch.FloatTensor(weights).to(device)
    criterion_cls = nn.CrossEntropyLoss()

    if Config.PHASE == 'Train' or Config.PHASE == 'train':
        print('===> Start Training')
        train(data_loaders, model, (criterion_pts, criterion_cls))
        print('=================Finished Train===================')
    elif Config.PHASE == 'Finetune' or Config.PHASE == 'finetune':
        print('===> Finetune')
        # model.load_state_dict(torch.load(os.path.join(args.save_directory, 'aligner_epoch_28.pt')))
    elif Config.PHASE == 'Predict' or Config.PHASE == 'predict':
        print('===> Predict')


if __name__ == '__main__':
    main()

