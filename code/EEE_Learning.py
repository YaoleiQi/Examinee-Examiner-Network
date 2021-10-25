# -*- coding: utf-8 -*-
import os
from os.path import join
import numpy as np
from os import listdir
# from torch.nn.functional import max_pool3d, interpolate

from DataLoader_txt import DataLoad
from E1_Net import E1_Net
from E2_Net import E2_Net

from loss import dropoutput_layer, cross_loss
from torch.utils.data import DataLoader
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_epoch(model1, model2, loader1, loader2, optimizer1, optimizer2, criterion1, criterion2, epoch, n_epochs):
    losses32 = AverageMeter()
    losses12 = AverageMeter()
    losses21 = AverageMeter()
    losses22 = AverageMeter()

    dice1 = AverageMeter()
    dice2 = AverageMeter()

    model1.train()
    model2.train()
    for batch_idx in range(loader1.__len__()):
        input1, target11, target12 = loader1.__iter__().__next__()
        input2, target21, target22 = loader2.__iter__().__next__()

        if torch.cuda.is_available():
            input1 = input1.cuda()
            input2 = input2.cuda()
            target11 = target11.cuda()
            target12 = target12.cuda()
            target21 = target21.cuda()  # During training, unlabeled data has no label.
            target22 = target22.cuda()

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        model1.zero_grad()
        model2.zero_grad()

        # stage1
        output12 = model2(target11)
        loss12 = criterion2(target12, output12)

        dice1.update((dice_coef()(target12, output12)).data, target12.size(0))
        losses12.update(loss12.data, target12.size(0))

        loss1 = loss12

        loss1.backward()
        optimizer2.step()

        res = '\t'.join(['Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                         'Iter: [%d/%d]' % (batch_idx + 1, len(loader1)),
                         'Loss %f' % (losses12.avg),
                         'Dice %f' % (dice1.avg)])
        print(res)

        # stage2
        output21 = model1(input1)
        output22 = model2(output21)

        loss21 = criterion1(target11, output21)
        loss22 = criterion2(target12, output22)

        loss2 = loss21 + loss22

        losses21.update(loss21.data, target11.size(0))
        losses22.update(loss22.data, target12.size(0))

        loss2.backward()
        optimizer1.step()

        res = '\t'.join(['Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                         'Iter: [%d/%d]' % (batch_idx + 1, len(loader1)),
                         'Loss1 %f' % (losses21.avg),
                         'Loss2 %f' % (losses22.avg)])

        print(res)

        # stage3
        output31 = model1(input2)
        output32 = model2(output31)

        loss32 = criterion2(target22, output32)

        dice2.update((dice_coef()(target11, output21)).data, target11.size(0))
        losses32.update(loss32.data, target22.size(0))

        loss3 = loss32

        loss3.backward()
        optimizer1.step()

        res = '\t'.join(['Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                         'Iter: [%d/%d]' % (batch_idx + 1, len(loader1)),
                         'Loss %f' % (losses32.avg),
                         'Dice %f' % (dice2.avg)])
        print(res)
    return losses12.avg, losses21.avg, losses22.avg, losses32.avg


def train_net(net1, net2, n_epochs=200, batch_size=1, lr=1e-4, model1_name='Evaluation1_66_1_new.dat',
              model2_name='Evaluation2_66_1_new.dat'):
    '''
    Here we provide the trained parameters to facilitate testing.
    '''

    shape = (224, 288, 288)   # The size of the crop

    train_image_dir = 'Txt/Txt_weak_acnet_66_1/image.txt'
    train_label_dir = 'Txt/Txt_weak_acnet_66_1/label.txt'
    train_mask_dir = 'Txt/Txt_weak_acnet_66_1/label_line.txt'

    train_weak_image_dir = 'Txt/Txt_weak_acnet_66_1/weak_image.txt'
    train_weak_label_dir = 'Txt/Txt_weak_acnet_66_1/weak_label.txt'
    train_weak_mask_dir = 'Txt/Txt_weak_acnet_66_1/weak_label_line.txt'

    test_image_dir = 'Txt/test_challenge_image.txt'     # ASOCA DATA
    test_label_dir = 'Txt/test_challenge_label.txt'
    # test_image_dir = 'Txt/Txt_weak_acnet_66_1/test_image.txt'    # Our DATA
    # test_label_dir = 'Txt/Txt_weak_acnet_66_1/test_label.txt'

    save_dir = 'Results/R_challenge_1'      # Save the results of the validation set.
    save_dir_m = 'Results/R_challenge_1'    # Save the final results of the testing set.
    checkpoint_dir = 'Weights'              # Save temporary parameters.
    checkpoint_dir2 = 'Weights_m'           # Save the best model parameters on the validation set.

    dice = 0
    epoch_m = 0

    net1.load_state_dict(torch.load(os.path.join(checkpoint_dir2, model1_name)))       # Only E1-Net is required for testing.
    # net2.load_state_dict(torch.load(os.path.join(checkpoint_dir, model2_name)))

    if torch.cuda.is_available():
        net1 = net1.cuda()
        net2 = net2.cuda()

    train_dataset = DataLoad2(train_image_dir, train_label_dir, train_mask_dir, shape)

    train_dataset_weak = DataLoad2(train_weak_image_dir, train_weak_label_dir, train_weak_mask_dir, shape)

    train_dataloader1 = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_dataloader2 = DataLoader(train_dataset_weak, batch_size=batch_size, shuffle=True)

    optimizer1 = torch.optim.Adam(net1.parameters(), lr=lr)
    optimizer2 = torch.optim.Adam(net2.parameters(), lr=lr)

    criterion1 = dropoutput_layer()
    criterion2 = dropoutput_layer()

    for epoch in range(n_epochs):
        loss = train_epoch(net1, net2, train_dataloader1, train_dataloader2, optimizer1, optimizer2, criterion1,
                           criterion2, epoch, n_epochs)

        torch.save(net1.state_dict(), os.path.join(checkpoint_dir, model1_name))
        torch.save(net2.state_dict(), os.path.join(checkpoint_dir, model2_name))

        if (epoch > 150):
            predict(net1, save_path=save_dir, shape=shape, img_path=test_image_dir, num_classes=1)
            tempdice = Dice(test_label_dir, save_dir)
            meandice = np.mean(tempdice)
            if (meandice > dice):
                dice = meandice
                epoch_m = epoch
                predict(net1, save_path=save_dir_m, shape=shape, img_path=test_image_dir, num_classes=1)
                torch.save(net1.state_dict(), os.path.join(checkpoint_dir2, model1_name))
                torch.save(net2.state_dict(), os.path.join(checkpoint_dir2, model2_name))
        print(dice)
        print(epoch_m)
    predict_all(net1, save_path=save_dir, shape=shape, img_path=test_image_dir, num_classes=1)


def read_file_from_txt(txt_path):
    files = []
    for line in open(txt_path, 'r'):
        files.append(line.strip())
    return files


def predict(model, save_path, shape, img_path, num_classes):
    print("Predict test data")
    model.eval()

    file = read_file_from_txt(img_path)
    file_num = len(file)

    for t in range(file_num):
        image_path = file[t]
        print(image_path)

        s = image_path.split('/')
        s = str(s[-1])
        # print(s)

        image = np.fromfile(file=image_path, dtype=np.float32)
        shape1 = np.load('Npy/' + s.rstrip('.raw') + '.npy')
        # print(shape1)
        x = shape1[2]
        y = shape1[1]
        z = shape1[0]

        image = image.reshape(z, y, x)
        image = image.astype(np.float32)

        # meanstd = np.load('image_miccai_meanstd.npy')
        # mean = meanstd[0]
        # std = meanstd[1]
        #
        # image = (image - mean) / std

        o = z
        p = y
        q = x

        if shape[0] > z:
            z = shape[0]
            image = reshape_img(image, z, y, x)
        if shape[1] > y:
            y = shape[1]
            image = reshape_img(image, z, y, x)
        if shape[2] > x:
            x = shape[2]
            image = reshape_img(image, z, y, x)

        predict = np.zeros([1, num_classes, z, y, x], dtype=np.float32)
        n_map = np.zeros([1, num_classes, z, y, x], dtype=np.float32)

        a = np.zeros(shape=shape)
        a = np.where(a == 0)
        map_kernal = 1 / ((a[0] - shape[0] // 2) ** 4 + (a[1] - shape[1] // 2) ** 4 + (a[2] - shape[2] // 2) ** 4 + 1)
        map_kernal = np.reshape(map_kernal, newshape=(1, 1,) + shape)

        # print(np.max(map_kernal))
        image = image[np.newaxis, np.newaxis, :, :, :]
        stride_x = shape[0] // 2
        stride_y = shape[1] // 2
        stride_z = shape[2] // 2
        for i in range(z // stride_x - 1):
            for j in range(y // stride_y - 1):
                for k in range(x // stride_z - 1):
                    image_i = image[:, :, i * stride_x:i * stride_x + shape[0], j * stride_y:j * stride_y + shape[1],
                              k * stride_z:k * stride_z + shape[2]]
                    image_i = torch.from_numpy(image_i)
                    if torch.cuda.is_available():
                        image_i = image_i.cuda()
                    output = model(image_i)
                    output = output.data.cpu().numpy()

                    predict[:, :, i * stride_x:i * stride_x + shape[0], j * stride_y:j * stride_y + shape[1],
                    k * stride_z:k * stride_z + shape[2]] += output * map_kernal

                    n_map[:, :, i * stride_x:i * stride_x + shape[0], j * stride_y:j * stride_y + shape[1],
                    k * stride_z:k * stride_z + shape[2]] += map_kernal

                image_i = image[:, :, i * stride_x:i * stride_x + shape[0], j * stride_y:j * stride_y + shape[1],
                          x - shape[2]:x]
                image_i = torch.from_numpy(image_i)
                if torch.cuda.is_available():
                    image_i = image_i.cuda()
                output = model(image_i)
                output = output.data.cpu().numpy()
                predict[:, :, i * stride_x:i * stride_x + shape[0], j * stride_y:j * stride_y + shape[1],
                x - shape[2]:x] += output * map_kernal

                n_map[:, :, i * stride_x:i * stride_x + shape[0], j * stride_y:j * stride_y + shape[1],
                x - shape[2]:x] += map_kernal

            for k in range(x // stride_z - 1):
                image_i = image[:, :, i * stride_x:i * stride_x + shape[0], y - shape[1]:y,
                          k * stride_z:k * stride_z + shape[2]]
                image_i = torch.from_numpy(image_i)
                if torch.cuda.is_available():
                    image_i = image_i.cuda()
                output = model(image_i)
                output = output.data.cpu().numpy()
                predict[:, :, i * stride_x:i * stride_x + shape[0], y - shape[1]:y,
                k * stride_z:k * stride_z + shape[2]] += output * map_kernal

                n_map[:, :, i * stride_x:i * stride_x + shape[0], y - shape[1]:y,
                k * stride_z:k * stride_z + shape[2]] += map_kernal

            image_i = image[:, :, i * stride_x:i * stride_x + shape[0], y - shape[1]:y, x - shape[2]:x]
            image_i = torch.from_numpy(image_i)
            if torch.cuda.is_available():
                image_i = image_i.cuda()
            output = model(image_i)
            output = output.data.cpu().numpy()

            predict[:, :, i * stride_x:i * stride_x + shape[0], y - shape[1]:y, x - shape[2]:x] += output * map_kernal
            n_map[:, :, i * stride_x:i * stride_x + shape[0], y - shape[1]:y, x - shape[2]:x] += map_kernal

        for j in range(y // stride_y - 1):
            for k in range((x - shape[2]) // stride_z):
                image_i = image[:, :, z - shape[0]:z, j * stride_y:j * stride_y + shape[1],
                          k * stride_z:k * stride_z + shape[2]]
                image_i = torch.from_numpy(image_i)
                if torch.cuda.is_available():
                    image_i = image_i.cuda()
                output = model(image_i)
                output = output.data.cpu().numpy()

                predict[:, :, z - shape[0]:z, j * stride_y:j * stride_y + shape[1],
                k * stride_z:k * stride_z + shape[2]] += output * map_kernal

                n_map[:, :, z - shape[0]:z, j * stride_y:j * stride_y + shape[1],
                k * stride_z:k * stride_z + shape[2]] += map_kernal

            image_i = image[:, :, z - shape[0]:z, j * stride_y:j * stride_y + shape[1],
                      x - shape[2]:x]
            image_i = torch.from_numpy(image_i)
            if torch.cuda.is_available():
                image_i = image_i.cuda()
            output = model(image_i)
            output = output.data.cpu().numpy()

            predict[:, :, z - shape[0]:z, j * stride_y:j * stride_y + shape[1],
            x - shape[2]:x] += output * map_kernal

            n_map[:, :, z - shape[0]:z, j * stride_y:j * stride_y + shape[1],
            x - shape[2]:x] += map_kernal

        for k in range(x // stride_z - 1):
            image_i = image[:, :, z - shape[0]:z, y - shape[1]:y,
                      k * stride_z:k * stride_z + shape[2]]
            image_i = torch.from_numpy(image_i)
            if torch.cuda.is_available():
                image_i = image_i.cuda()
            output = model(image_i)
            output = output.data.cpu().numpy()

            predict[:, :, z - shape[0]:z, y - shape[1]:y,
            k * stride_z:k * stride_z + shape[2]] += output * map_kernal

            n_map[:, :, z - shape[0]:z, y - shape[1]:y,
            k * stride_z:k * stride_z + shape[2]] += map_kernal

        image_i = image[:, :, z - shape[0]:z, y - shape[1]:y, x - shape[2]:x]
        image_i = torch.from_numpy(image_i)
        if torch.cuda.is_available():
            image_i = image_i.cuda()
        output = model(image_i)
        output = output.data.cpu().numpy()

        predict[:, :, z - shape[0]:z, y - shape[1]:y, x - shape[2]:x] += output * map_kernal
        n_map[:, :, z - shape[0]:z, y - shape[1]:y, x - shape[2]:x] += map_kernal

        predict = predict / n_map
        predict[0, 0, 0:o, 0:p, 0:q].tofile((join(save_path, s)))
        print("finish!")

def predict_all(model1, save_path, shape, img_path, num_classes):
    print("Predict test data")
    model1.eval()

    file = read_file_from_txt(img_path)
    file_num = len(file)

    for t in range(file_num):
        image_path = file[t]
        print(image_path)

        s = image_path.split('/')
        s = str(s[-1])

        image = np.fromfile(file=image_path, dtype=np.float32)

        shape1 = np.load('Npy/' + s.rstrip('.raw') + '.npy')

        x = shape1[2]
        y = shape1[1]
        z = shape1[0]

        image = image.reshape(z, y, x)
        image = image.astype(np.float32)

        # meanstd = np.load('miccai_meanstd.npy')
        # mean = meanstd[0]
        # std = meanstd[1]
        #
        # image = (image - mean) / std

        o = z
        p = y
        q = x

        if shape[0] > z:
            z = shape[0]
            image = reshape_img(image, z, y, x)
        if shape[1] > y:
            y = shape[1]
            image = reshape_img(image, z, y, x)
        if shape[2] > x:
            x = shape[2]
            image = reshape_img(image, z, y, x)

        predict = np.zeros([1, num_classes, z, y, x], dtype=np.float32)
        n_map = np.zeros([1, num_classes, z, y, x], dtype=np.float32)

        a = np.zeros(shape=shape)
        a = np.where(a == 0)
        map_kernal = 1 / ((a[0] - shape[0] // 2) ** 4 + (a[1] - shape[1] // 2) ** 4 + (a[2] - shape[2] // 2) ** 4 + 1)
        map_kernal = np.reshape(map_kernal, newshape=(1, 1,) + shape)

        # print(np.max(map_kernal))
        image = image[np.newaxis, np.newaxis, :, :, :]
        stride_x = shape[0] // 2
        stride_y = shape[1] // 2
        stride_z = shape[2] // 2
        for i in range(z // stride_x - 1):
            for j in range(y // stride_y - 1):
                for k in range(x // stride_z - 1):
                    image_i = image[:, :, i * stride_x:i * stride_x + shape[0], j * stride_y:j * stride_y + shape[1],
                              k * stride_z:k * stride_z + shape[2]]
                    image_i = torch.from_numpy(image_i)
                    if torch.cuda.is_available():
                        image_i = image_i.cuda()
                    output = model1(image_i)
                    output = output.data.cpu().numpy()

                    predict[:, :, i * stride_x:i * stride_x + shape[0], j * stride_y:j * stride_y + shape[1],
                    k * stride_z:k * stride_z + shape[2]] += output * map_kernal

                    n_map[:, :, i * stride_x:i * stride_x + shape[0], j * stride_y:j * stride_y + shape[1],
                    k * stride_z:k * stride_z + shape[2]] += map_kernal

                image_i = image[:, :, i * stride_x:i * stride_x + shape[0], j * stride_y:j * stride_y + shape[1],
                          x - shape[2]:x]
                image_i = torch.from_numpy(image_i)
                if torch.cuda.is_available():
                    image_i = image_i.cuda()
                output = model1(image_i)
                output = output.data.cpu().numpy()
                predict[:, :, i * stride_x:i * stride_x + shape[0], j * stride_y:j * stride_y + shape[1],
                x - shape[2]:x] += output * map_kernal

                n_map[:, :, i * stride_x:i * stride_x + shape[0], j * stride_y:j * stride_y + shape[1],
                x - shape[2]:x] += map_kernal

            for k in range(x // stride_z - 1):
                image_i = image[:, :, i * stride_x:i * stride_x + shape[0], y - shape[1]:y,
                          k * stride_z:k * stride_z + shape[2]]
                image_i = torch.from_numpy(image_i)
                if torch.cuda.is_available():
                    image_i = image_i.cuda()
                output = model1(image_i)
                output = output.data.cpu().numpy()
                predict[:, :, i * stride_x:i * stride_x + shape[0], y - shape[1]:y,
                k * stride_z:k * stride_z + shape[2]] += output * map_kernal

                n_map[:, :, i * stride_x:i * stride_x + shape[0], y - shape[1]:y,
                k * stride_z:k * stride_z + shape[2]] += map_kernal

            image_i = image[:, :, i * stride_x:i * stride_x + shape[0], y - shape[1]:y, x - shape[2]:x]
            image_i = torch.from_numpy(image_i)
            if torch.cuda.is_available():
                image_i = image_i.cuda()
            output = model1(image_i)
            output = output.data.cpu().numpy()

            predict[:, :, i * stride_x:i * stride_x + shape[0], y - shape[1]:y, x - shape[2]:x] += output * map_kernal
            n_map[:, :, i * stride_x:i * stride_x + shape[0], y - shape[1]:y, x - shape[2]:x] += map_kernal

        for j in range(y // stride_y - 1):
            for k in range((x - shape[2]) // stride_z):
                image_i = image[:, :, z - shape[0]:z, j * stride_y:j * stride_y + shape[1],
                          k * stride_z:k * stride_z + shape[2]]
                image_i = torch.from_numpy(image_i)
                if torch.cuda.is_available():
                    image_i = image_i.cuda()
                output = model1(image_i)
                output = output.data.cpu().numpy()

                predict[:, :, z - shape[0]:z, j * stride_y:j * stride_y + shape[1],
                k * stride_z:k * stride_z + shape[2]] += output * map_kernal

                n_map[:, :, z - shape[0]:z, j * stride_y:j * stride_y + shape[1],
                k * stride_z:k * stride_z + shape[2]] += map_kernal

            image_i = image[:, :, z - shape[0]:z, j * stride_y:j * stride_y + shape[1],
                      x - shape[2]:x]
            image_i = torch.from_numpy(image_i)
            if torch.cuda.is_available():
                image_i = image_i.cuda()
            output = model1(image_i)
            output = output.data.cpu().numpy()

            predict[:, :, z - shape[0]:z, j * stride_y:j * stride_y + shape[1],
            x - shape[2]:x] += output * map_kernal

            n_map[:, :, z - shape[0]:z, j * stride_y:j * stride_y + shape[1],
            x - shape[2]:x] += map_kernal

        for k in range(x // stride_z - 1):
            image_i = image[:, :, z - shape[0]:z, y - shape[1]:y,
                      k * stride_z:k * stride_z + shape[2]]
            image_i = torch.from_numpy(image_i)
            if torch.cuda.is_available():
                image_i = image_i.cuda()
            output = model1(image_i)
            output = output.data.cpu().numpy()

            predict[:, :, z - shape[0]:z, y - shape[1]:y,
            k * stride_z:k * stride_z + shape[2]] += output * map_kernal

            n_map[:, :, z - shape[0]:z, y - shape[1]:y,
            k * stride_z:k * stride_z + shape[2]] += map_kernal

        image_i = image[:, :, z - shape[0]:z, y - shape[1]:y, x - shape[2]:x]
        image_i = torch.from_numpy(image_i)
        if torch.cuda.is_available():
            image_i = image_i.cuda()
        output = model1(image_i)
        output = output.data.cpu().numpy()

        predict[:, :, z - shape[0]:z, y - shape[1]:y, x - shape[2]:x] += output * map_kernal
        n_map[:, :, z - shape[0]:z, y - shape[1]:y, x - shape[2]:x] += map_kernal

        predict = predict / n_map
        predict[0, 0, 0:o, 0:p, 0:q].tofile((join(save_path, s)))
        print("finish!")

def is_image3d_file(filename):
    return any(filename.endswith(extension) for extension in [".raw"])

def reshape_img(image, z, y, x):
    out = np.zeros([z, y, x], dtype=np.float32)
    out[0:image.shape[0], 0:image.shape[1], 0:image.shape[2]] = image[0:image.shape[0], 0:image.shape[1],
                                                                0:image.shape[2]]
    return out

def Dice(label_dir, pred_dir):
    file = read_file_from_txt(label_dir)
    file_num = len(file)
    i = 0

    dice = np.zeros(shape=(file_num), dtype=np.float32)
    list1 = []
    name = []

    for t in range(file_num):
        image_path = file[t]
        s = image_path.split('/')
        s = str(s[-1])
        predict = np.fromfile(join(pred_dir, s), dtype=np.float32)
        predict = np.where(predict > 0.5, 1, 0)
        print(predict.shape)
        groundtruth = np.fromfile(file=image_path, dtype=np.float32)
        groundtruth = np.where(groundtruth > 0, 1, 0)
        print(groundtruth.shape)
        tmp = predict + groundtruth
        a = np.sum(np.where(tmp == 2, 1, 0))
        b = np.sum(predict)
        c = np.sum(groundtruth)
        dice[i] = (2 * a) / (b + c)
        print(s, dice[i])
        name.append(s)
        list1.append(dice[i])
        i += 1
    return dice

if __name__ == '__main__':
    net1 = E1_Net(n_channels=1, n_classes=1)
    net2 = E2_Net(n_channels=1, n_classes=1)
    # train_net(net1=net1, net2=net2, n_epochs=0, batch_size=1, lr=1e-4)     # When testing, set n_epoch to 0.

    '''
    Test
    '''
    model1_name = 'Evaluation1_66_1_new.dat'
    test_image_dir = 'Txt/test_challenge_image.txt'     # ASOCA DATA
    test_label_dir = 'Txt/test_challenge_label.txt'
    # test_image_dir = 'Txt/Txt_weak_acnet_66_1/test_image.txt'    # Our DATA
    # test_label_dir = 'Txt/Txt_weak_acnet_66_1/test_label.txt'
    save_dir = 'Results/R_challenge'      # Save the results of the validation set.
    checkpoint_dir2 = 'Weights_m'  # Save the best model parameters on the validation set.
    net1.load_state_dict(torch.load(os.path.join(checkpoint_dir2, model1_name)))  # Only E1-Net is required for testing.
    predict_all(net1, save_path=save_dir, shape=shape, img_path=test_image_dir, num_classes=1)