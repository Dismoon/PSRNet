import torch
import math
import numpy
import numpy as np
def cal_merge(x):
    dofp = np.zeros((x.shape[0] * 2, x.shape[1] * 2))
    # print(dofp.shape)
    img0 = x[:, :, 0]
    img45 = x[:, :, 1]
    img90 = x[:, :, 2]
    img135 = x[:, :, 3]

    dofp[0:dofp.shape[0]:2, 0:dofp.shape[1]:2] = img90
    dofp[1:dofp.shape[0]:2, 0:dofp.shape[1]:2] = img135
    dofp[0:dofp.shape[0]:2, 1:dofp.shape[1]:2] = img45
    dofp[1:dofp.shape[0]:2, 1:dofp.shape[1]:2] = img0

    return dofp
def cal_stokes_dolp(img):

    img = np.array(img)
    img0 = img[:, :, 0]
    img45 = img[:, :, 1]
    img90 = img[:, :, 2]
    img135 = img[:, :, 3]

    S0 = (img0.astype(np.float32) + img45.astype(np.float32) +
          img90.astype(np.float32) + img135.astype(np.float32)) * 0.5
    S1 = img0.astype(np.float32) - img90.astype(np.float32)
    S2 = img45.astype(np.float32) - img135.astype(np.float32)
    S0 = S0 * (255 / np.nanmax(S0))
    DoLP = np.sqrt((S1 ** 2 + S2 ** 2) / (S0+0.000001) ** 2)
    DoLP = np.clip(DoLP * 255, 0, 255)
    AoP = 1 / 2 * np.arctan2(S2, S1)
    AoP = (AoP + math.pi/2)/math.pi
    #AoP = AoP * (1 / np.nanmax(AoP))  # 归一化到1内
    AoP = np.clip(AoP * 255, 0, 255)
    return  S0,DoLP,AoP
def gasuss_noise(image, mean=0, var=0.001):
    image = np.array(image/1.0, dtype=float)
    noise = np.random.normal(mean, var, image.shape)
    out = image + noise
    out = np.clip(out, 0, 255.0)
    out = np.uint8(out)
    #cv.imshow("gasuss", out)
    return out
def cal_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(1.0 ** 2 / mse)
def cal_stokes(img):
    img0=img[:,0,:,:]
    img45=img[:,1,:,:]
    img90=img[:,2,:,:]
    img135=img[:,3,:,:]
    S0 = (img0 + img45 + img90 + img135 )* 0.5
    S0 = S0 * (1 / S0.max())
    S1 = img0 - img90
    S2 = img45 - img135
    return S0,S1,S2
def cal_dolp(S0,S1,S2):
    DoLP = torch.sqrt((S1 ** 2 + S2 ** 2) / ((S0+0.000001) ** 2))
    DoLP = torch.clamp(DoLP,0,1)
    return DoLP

def cal_aop(S1,S2):
    AoP = 1 / 2 * torch.atan2(S2, S1)
    AoP = (AoP + math.pi/2.0 ) / (math.pi)
    AoP = torch.clamp(AoP,0,1)
    return AoP
