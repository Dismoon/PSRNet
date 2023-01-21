import cv2
import numpy as np
import math
import glob
import os
import torch.utils.data as data
from torchvision import transforms
import torch
import h5py
import random

class DataFromH5File(data.Dataset):
    def __init__(self, filepath):
        h5File = h5py.File(filepath, 'r')
        self.hr = h5File['label']
        self.lr = h5File['input']

    def __getitem__(self, idx):
        label = torch.from_numpy(self.hr[idx]).float()
        data = torch.from_numpy(self.lr[idx]).float()
        return data, label

    def __len__(self):
        assert self.hr.shape[0] == self.lr.shape[0], "Wrong data length"
        return self.hr.shape[0]

def input_setup(config):
    input_list, label_list,eval_input_list,eval_label_list = prepare_data(config)
    print('Prepare training data...')
    make_sub_data(input_list,label_list,config,'train')
    print('Prepare evaluating data...')
    make_sub_data(eval_input_list,eval_label_list,config,'eval')

def prepare_data(config):#11
    if config.is_train:
        input_dir = os.path.join(os.path.join(os.getcwd(),config.train_set_input))
        input_list = glob.glob(os.path.join(input_dir, "*png"))
        label_dir = os.path.join(os.path.join(os.getcwd(),config.train_set_label))
        label_list = glob.glob(os.path.join(label_dir, "*png"))

        eval_input_dir = os.path.join(os.path.join(os.getcwd(),config.eval_set_input))
        eval_input_dir = glob.glob(os.path.join(eval_input_dir, "*png"))
        eval_label_dir = os.path.join(os.path.join(os.getcwd(), config.eval_set_label))
        eval_label_dir = glob.glob(os.path.join(eval_label_dir, "*png"))
        return input_list, label_list,eval_input_dir,eval_label_dir

    else:
        data_dir = os.path.join(os.getcwd(),config.test_set)
        data = glob.glob(os.path.join(data_dir, "*.png"))

        return data

def get_eval_data(config):#11
    input_dir = os.path.join(os.getcwd(), config.eval_set_input)
    input_list = glob.glob(os.path.join(input_dir, "*.png"))
    label_dir = os.path.join(os.getcwd(), config.eval_set_label)
    label_list = glob.glob(os.path.join(label_dir, "*.png"))
    return input_list, label_list

def make_sub_data(input_list,label_list,config,str):
    assert len(input_list) == len(label_list)
    times = 0
    for i in range(len(input_list)):
        input_ = cv2.imread(input_list[i], -1)
        label_ = cv2.imread(label_list[i], -1)
        # print(label.shape)

        if len(input_.shape) == 3:
            h, w, c = input_.shape
        else:
            h,w = input_.shape
        for x in range(0, h * config.scale - config.image_size * config.scale + 1, config.stride * config.scale):
            for y in range(0, w * config.scale - config.image_size * config.scale + 1, config.stride * config.scale):
                sub_label = label_[x: x + config.image_size * config.scale, y: y + config.image_size * config.scale]
                sub_label = sub_label.reshape([config.image_size * config.scale, config.image_size * config.scale, config.c_dim])
                sub_label = sub_label / 255.0  # 归一化
                totensor = transforms.ToTensor()
                sub_label=totensor(sub_label)#pytorch的图片格式为[batchsize c w h]与ndarray的[batchsize w h c]不同

                x_i = x // config.scale
                y_i = y // config.scale
                sub_input = input_[x_i: x_i + config.image_size, y_i: y_i + config.image_size]
                sub_input = sub_input.reshape([config.image_size, config.image_size, config.c_dim])
                sub_input = sub_input / 255.0
                sub_input=totensor(sub_input)
                save_flag = make_data_hf(sub_input, sub_label,config,str,times,1)
                if not save_flag:
                    return input_list,label_list
                times += 1
        print("image: [%2d], total: [%2d]" % (i, len(input_list)))
    return input_list,label_list

def make_data_hf(input_,label_,config,str,times,is_train):
    if not os.path.isdir(os.path.join(os.getcwd(),config.checkpoint_dir)):
        os.makedirs(os.path.join(os.getcwd(),"checkpoint"))
    if str=='train':
        savepath = os.path.join(os.path.join(os.getcwd(), config.checkpoint_dir), 'train.h5')
    elif str=='eval':
        savepath = os.path.join(os.path.join(os.getcwd(),"checkpoint"), 'eval.h5')
    else:      savepath = os.path.join(os.path.join(os.getcwd(), config.checkpoint_dir), 'test.h5')

    if times == 0:
        if os.path.exists(savepath):
            print("\n%s have existed!\n" % (savepath))
            return False
        else:
            hf = h5py.File(savepath, 'w')
            if is_train:
                input_h5 = hf.create_dataset("input",(1,config.c_dim, config.image_size, config.image_size),
                                         maxshape = (None,config.c_dim,config.image_size, config.image_size ),
                                         chunks=(1,config.c_dim, config.image_size, config.image_size),
                                         dtype='float32')

                label_h5 = hf.create_dataset("label",(1, config.c_dim,config.image_size* config.scale, config.image_size* config.scale),
                                         maxshape = (None,config.c_dim,config.image_size* config.scale, config.image_size* config.scale),
                                         chunks=(1,config.c_dim, config.image_size* config.scale, config.image_size* config.scale),
                                         dtype='float32')

            else:
                input_h5 = hf.create_dataset("input", (1, input_.shape[0], input_.shape[1], input_.shape[2]),
                                             maxshape=(None, input_.shape[0], input_.shape[1], input_.shape[2]),
                                             chunks=(1, input_.shape[0], input_.shape[1], input_.shape[2]),
                                             dtype='float32')
                label_h5 = hf.create_dataset("label", (1, label_.shape[0], label_.shape[1], label_.shape[2]),
                                             maxshape=(None, label_.shape[0], label_.shape[1], label_.shape[2]),
                                             chunks=(1, label_.shape[0], label_.shape[1], label_.shape[2]),
                                             dtype='float32')
    else:
        hf = h5py.File(savepath, 'a')
        input_h5 = hf["input"]
        label_h5 = hf["label"]
    if config.is_train:
        input_h5.resize([times + 1,config.c_dim, config.image_size, config.image_size])
        input_h5[times: times + 1] = input_
        label_h5.resize([times + 1,config.c_dim, config.image_size* config.scale, config.image_size* config.scale])
        label_h5[times: times + 1] = label_
    else:
        input_h5.resize([times + 1, input_.shape[0], input_.shape[1], input_.shape[2]])
        input_h5[times: times + 1] = input_
        label_h5.resize([times + 1, label_.shape[0], label_.shape[1], label_.shape[2]])
        label_h5[times: times + 1] = label_
    hf.close()
    return True

def get_batch(inputs, labels):
    input_ = np.array(inputs)
    label_ = np.array(labels)
    random_aug = np.random.rand(2)
    batch_images = augmentation(input_, random_aug)
    batch_labels = augmentation(label_, random_aug)
    batch_images = batch_images.copy()
    batch_labels = batch_labels.copy()
    tensor_images = torch.from_numpy(batch_images)
    tensor_labels = torch.from_numpy(batch_labels)
    return tensor_images, tensor_labels
def augmentation(batch, random):
    # 在batch的第shape[1]上，上下翻转
    if random[0] < 0.3:
        batch_flip = np.flip(batch, 2)
        # 在batch的第shape[2]上，左右翻转
    elif random[0] > 0.7:
        batch_flip = np.flip(batch, 3)
        # 不翻转
    else:
        batch_flip = batch
    # 在翻转的基础上旋转
    if random[1] < 0.5:
        # 逆时针旋转90度
        batch_rot = np.rot90(batch_flip, 1, [2, 3])
    else:
        batch_rot = batch_flip

    return batch_rot
