"""This program is used for double cubic downsampling to generate LR images."""
import cv2
import glob
import os
import numpy as np
from utils import *
def preprocess(path, scale=2, eng=None, mdouble=None):
    img = path
    label_ = modcrop(img, scale)
    if eng is None:
        input_ = cv2.resize(label_, None, fx=1.0/scale, fy=1.0/scale, interpolation=cv2.INTER_CUBIC)
    else:
        input_ = np.asarray(eng.imresize(mdouble(label_.tolist()), 1.0/scale, 'bicubic'))
    return input_, label_


def modcrop(img, scale=3):
    if len(img.shape) == 3:
        h, w, _ = img.shape
        h = (h // scale) * scale
        w = (w // scale) * scale
        img = img[0:h, 0:w, :]
    else:
        h, w = img.shape
        h = (h // scale) * scale
        w = (w // scale) * scale
        img = img[0:h, 0:w]
    return img
def main():
    # eng = matlab.engine.start_matlab()
    # mdouble = matlab.double
    Dataset_to_be_processed='/test/'
    label_dir=os.getcwd()+Dataset_to_be_processed+'crop/'
    label_dir_list=glob.glob(os.path.join(label_dir,"*png"))
    label_dir_list.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    path1=os.getcwd()+Dataset_to_be_processed+'input'
    path2=os.getcwd()+Dataset_to_be_processed+'label'
    path3=os.getcwd()+Dataset_to_be_processed+'input_aop'
    path4 = os.getcwd() + Dataset_to_be_processed + 'input_dolp'
    path5 = os.getcwd() + Dataset_to_be_processed + 'merge'
    if not os.path.isdir(path1):
        os.mkdir(path1)
    if not os.path.isdir(path2):
        os.mkdir(path2)
    if not os.path.isdir(path3):
        os.mkdir(path3)
    if not os.path.isdir(path4):
        os.mkdir(path4)
    if not os.path.isdir(path5):
        os.mkdir(path5)
    print("total:%2d" %len(label_dir_list))
    for i in range(len(label_dir_list)):
        print("{}".format(i + 1))
        img = cv2.imread(label_dir_list[i], -1)
        _,label_ = preprocess(img, 4)
        img=img.copy()
        hr_img = cv2.GaussianBlur(img, (9, 9), 1, 1)
        x=label_dir_list[i]
        y = int(x.split('/')[-1].split('.')[0])
        input_,_ = preprocess(hr_img,4)
        input_=gasuss_noise(input_,0,1)
        s0, dolp,aop = cal_stokes_dolp(input_)
        merge=cal_merge(input_)
        cv2.imwrite(os.getcwd() + Dataset_to_be_processed + 'merge/L_%2d.png' % y, merge)
        cv2.imwrite(os.getcwd() + Dataset_to_be_processed + 'input_aop/L_%2d.png' % y, aop)
        cv2.imwrite(os.getcwd() + Dataset_to_be_processed + 'input_dolp/L_%2d.png' % y, dolp)
        cv2.imwrite(os.getcwd()+Dataset_to_be_processed+'input/%2d.png'% y,input_)
        cv2.imwrite(os.getcwd() + Dataset_to_be_processed + 'label/%2d.png' % y, label_)
    print("Finished")
if __name__=='__main__':
    main()
