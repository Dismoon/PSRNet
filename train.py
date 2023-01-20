import time
import argparse
import random
from model import *
from utils import *
import os
from datasets import *
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from Loss import *
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
print(f'cuda version:{torch.version.cuda}')

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device:{device}')

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Set the parameters of the model
    parser.add_argument('--is_train', type=bool, default=True, help='if split the dataset')
    parser.add_argument('--image_size',type=int, default=32,help='the size of image input')
    parser.add_argument('--c_dim', type=int,default=4,help='the size of channel')
    parser.add_argument('--scale', type=int,default=4,help='the size of scale factor for preprocessing input image')
    parser.add_argument('--stride',type=int, default=16,help='the size of stride')
    parser.add_argument('--epoch', type=int,default=110,help='number of epoch')
    parser.add_argument('--batch_size',type=int, default=16,help='the size of batch')
    parser.add_argument('--learning_rate', type=float,default=1e-4,help='the learning rate')
    parser.add_argument('--is_lr_decay', type=bool,default=True,help='if lr decay')
    parser.add_argument('--lr_decay_steps',type=int, default=20,help='steps of learning rate decay')
    parser.add_argument('--lr_decay_rate',type=float, default=0.5,help='rate of learning rate decay')
    parser.add_argument('--eval_set_input',type=str, default='eval/input',help='eval_set_input')
    parser.add_argument('--eval_set_label', type=str,default='eval/label',help='the size of image input')
    parser.add_argument('--checkpoint_dir', type=str,default='checkpoint',help='name of the checkpoint directory')
    parser.add_argument('--result_dir', type=str,default='test_result',help='name of the result directory')
    parser.add_argument('--train_set_input', type=str,default='train/input',help='name of the input of train set')
    parser.add_argument('--train_set_label', type=str,default='train/label',help='name of the label of train set')
    parser.add_argument('--D', type=int,default=16,help='the number of D')
    parser.add_argument('--C', type=int,default=8,help='the number of C')
    parser.add_argument('--G', type=int,default=64,help='the number of G')
    parser.add_argument('--G0', type=int,default=64,help='the number of G0')
    parser.add_argument('--kernel_size',type=int, default=3,help='the size of kernel')
    args = parser.parse_args(args=[])
    date = time.strftime('%Y.%m.%d', time.localtime(time.time()))
    # set random seed
    set_seed()
    # Crop the training dataset
    print('\nPreparing data...\n')
    input_setup(args)

    # # load data
    train_data_path =os.path.join(args.checkpoint_dir, "train.h5")
    train_data=DataFromH5File(train_data_path)
    valid_data_path=os.path.join(args.checkpoint_dir, "eval.h5")
    valid_data=DataFromH5File(valid_data_path)
    # train_data = imgdata(args.train_set_input, args.train_set_label)
    # valid_data = imgdata_val(args.eval_set_input, args.eval_set_label)
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True,num_workers=8)
    valid_loader = DataLoader(valid_data, 1)
    print(f'Number of validating images:{len(valid_loader)}')

    # set the model
    net = PSRNet(scale_factor = args.scale,
              num_channels=args.c_dim,
              num_features = args.G0,
              growth_rate = args.G,
              num_blocks = args.D,
              num_layers = args.C,
              ks = args.kernel_size)
    # Initialize the parameters of the net权重初始化
    net.initialize_weight()
    net.to(device)

    # loss function
    loss_fn = PALoss()
    loss_pixel=nn.MSELoss()
    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), args.learning_rate)
    print("初始化的学习率：", optimizer.defaults['lr'])
    # scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,args.lr_decay_steps*len(train_loader),args.lr_decay_rate)

    # summary
    log_path = os.path.join('log_dir', date)
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(log_path, filename_suffix='event')

    # Set initial checkpoint
    iter = 0
    last_epoch = 0
    best_psnr=0
    best_epoch=0
    # If there exists a checkpoint, load it.
    checkpoint_path = os.path.join(args.checkpoint_dir, date)
    Epoch = args.epoch
    if os.path.isdir(checkpoint_path):
        checkpoint = torch.load(checkpoint_path + '/checkpoint_epoch.pkl')
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        last_epoch = checkpoint['epoch']
        iter = checkpoint['iter']
        print('\ncheckpoint load success!')

    else:
        print('\ncheckpoint load failed!')

    # Start the training process
    print("\nNow start training!\n")
    time0 = time.time()

    for epoch in range(last_epoch, Epoch):
        loss_mean = 0
        net.train()
        for i, data in enumerate(train_loader):
            # forward
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            pred = net(inputs)
            loss,loss_p,loss_a,loss_d,loss_s0,wa,wd,ws0 = loss_fn(pred, labels)
            # backward
            optimizer.zero_grad()
            loss.backward()
            iter += 1
            loss_mean += loss.item()

            # Update weight
            optimizer.step()
            scheduler.step()
            # print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
            if iter % 20 == 0:
                psnr = cal_psnr(pred, labels).item()

                S0, S1, S2 = cal_stokes(pred)
                pred_aop = cal_aop(S1, S2)
                pred_dolp = cal_dolp(S0, S1, S2)
                S0_gt, S1_gt, S2_gt = cal_stokes(labels)
                labels_aop = cal_aop(S1_gt, S2_gt)
                labels_dolp = cal_dolp(S0_gt, S1_gt, S2_gt)

                S0_psnr=cal_psnr(S0, S0_gt).item()
                aop_psnr=cal_psnr(pred_aop, labels_aop).item()
                dolp_psnr=cal_psnr(pred_dolp, labels_dolp).item()
                print('Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] time[{:.4f}min]'
                      ' Loss_mean[{:.9f}] PSNR[{:.4f}] PSNR_AoP[{:.4f}] PSNR_DoLP[{:.4f}] LR[{:.7f}] Loss_total[{:.9f}] Loss_P[{:.9f}] Loss_A[{:.9f}] Loss_D[{:.9f}] Loss_S0[{:.9f}] wa, wd, ws0[{:.4f},{:.4f},{:.4f}]'
                      .format(epoch + 1,
                              Epoch,
                              i + 1,
                              len(train_loader),
                              (time.time() - time0) / 60,
                              loss_mean / (20*args.batch_size),
                              psnr,
                              aop_psnr,
                              dolp_psnr,
                              optimizer.param_groups[0]['lr'],
                              loss.item(),
                              loss_p,
                              loss_a,
                              loss_d,
                              loss_s0,
                              wa, wd, ws0
                              )
                      )
                loss_mean = 0
                writer.add_scalars('AoP_PSNR', {'Train': aop_psnr}, iter)
                writer.add_scalars('DoLP_PSNR', {'Train': dolp_psnr}, iter)
                writer.add_scalars('PSNR', {'Train': psnr}, iter)
            writer.add_scalars('Loss', {'Train': loss.item()}, iter)
        net.eval()
        psnr_mean=0
        counter=0
        for i, data in enumerate(valid_loader):
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = net(inputs)
            # preds=preds.squeeze(0)
            # labels=labels.squeeze(0)

            # psnr = cal_psnr(preds, labels).item()

            S0, S1, S2 = cal_stokes(preds)
            pred_aop = cal_aop(S1, S2)
            pred_dolp = cal_dolp(S0, S1, S2)
            S0_gt, S1_gt, S2_gt = cal_stokes(labels)
            labels_aop = cal_aop(S1_gt, S2_gt)
            labels_dolp = cal_dolp(S0_gt, S1_gt, S2_gt)

            S0_psnr=cal_psnr(S0, S0_gt).item()
            aop_psnr=cal_psnr(pred_aop, labels_aop).item()
            dolp_psnr=cal_psnr(pred_dolp, labels_dolp).item()

            counter=counter+1
            psnr_mean =psnr_mean+aop_psnr+dolp_psnr#item()取出单元素张量的元素值并返回该值，保持原元素类型不变
        psnr_mean=psnr_mean/counter
        print('eval psnr: {:.2f}'.format(psnr_mean))

        checkpoint = {"model_state_dict": net.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(),
                      "scheduler_state_dict": scheduler.state_dict(),
                      "epoch": epoch +  1,
                      'iter': iter}
        path_checkpoint = os.path.join(checkpoint_path, "checkpoint_epoch.pkl")
        best_path_checkpoint = os.path.join(checkpoint_path, "best_checkpoint_epoch.pkl")
        if not os.path.isdir(checkpoint_path):
            os.makedirs(checkpoint_path)
        if psnr_mean > best_psnr:
            best_epoch = epoch
            best_psnr = psnr_mean
            torch.save(checkpoint, best_path_checkpoint)
        torch.save(checkpoint, path_checkpoint)
        print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
