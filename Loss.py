import torch.nn.functional as F
import torch.nn as nn
import torch
import torchvision.models as models
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
class PALoss(nn.Module):
    def __init__(self):
        super(PALoss,self).__init__()
        model = models.vgg19(pretrained=True).features
        model = model.cuda()
        self.features_low = model[0:4]
    def forward(self,image,label):
        S0, S1, S2 = self.cal_stokes(image)
        img_S0_lowfeature = self.get_feature(S0)
        img_S1_lowfeature = self.get_feature(S1)
        img_S2_lowfeature = self.get_feature(S2)
        img_aop = self.cal_aop(S1, S2)
        img_aop_lowfeature=self.get_feature(img_aop)
        img_dolp=self.cal_dolp(S0,S1,S2)
        img_dolp_lowfeature = self.get_feature(img_dolp)
        S0_gt, S1_gt, S2_gt = self.cal_stokes(label)
        label_S0_lowfeature = self.get_feature(S0_gt)
        label_S1_lowfeature = self.get_feature(S1_gt)
        label_S2_lowfeature = self.get_feature(S2_gt)
        label_aop = self.cal_aop(S1_gt, S2_gt)
        label_aop_lowfeature = self.get_feature(label_aop)
        label_dolp = self.cal_dolp(S0_gt, S1_gt, S2_gt)
        label_dolp_lowfeature = self.get_feature(label_dolp)
        loss_p = F.mse_loss(image, label)
        loss_a=F.mse_loss(img_aop_lowfeature,label_aop_lowfeature)*0.01
        loss_d=F.mse_loss(img_dolp_lowfeature,label_dolp_lowfeature)*0.01


        loss =loss_p+loss_a+loss_d
        return loss,loss_p,loss_a,loss_d

    def cal_stokes(self,img):
        img0=img[:,0,:,:]
        img45=img[:,1,:,:]
        img90=img[:,2,:,:]
        img135=img[:,3,:,:]
        S0 = (img0 + img45 + img90 + img135 )* 0.5
        S0 = S0 * (1 / torch.max(S0))  # 归一化在1内
        S1 = img0 - img90
        S2 = img45 - img135
        return S0,S1,S2

    def cal_dolp(self,S0,S1,S2):
        DoLP = torch.sqrt((S1 ** 2 + S2 ** 2) / ((S0+0.000001) ** 2))#dolp会大于1，需要裁剪或者归一化
        DoLP = torch.clamp(DoLP,0,1)
        return DoLP

    def cal_aop(self,S1,S2):
        AoP = 1 / 2 * torch.atan2(S2, S1)
        AoP = (AoP + math.pi/2.0 ) / (math.pi)#归一化
        AoP = torch.clamp(AoP,0,1)
        return AoP

    def get_feature(self,img_input):
        img_tensor=img_input
        img=torch.stack([img_tensor,img_tensor,img_tensor],dim=1)
        block2_conv2_features = self.features_low(img)
        return block2_conv2_features

if __name__ == '__main__':
    vgg = models.vgg19(pretrained=True).features
    print(vgg)
