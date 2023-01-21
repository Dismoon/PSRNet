import argparse
from utils import *
from model import *
import glob
from torchvision import transforms
import cv2
from Save import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
print(f'cuda version:{torch.version.cuda}')

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device:{device}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, default='test_result', help="test output")
    parser.add_argument('--scale', type=int,default=4,help='the size of scale factor for preprocessing input image')
    parser.add_argument('--test_set', type=str, default='test1/9x9/input', help="test input")
    parser.add_argument('--c_dim', type=int,default=4,help='the size of channel')
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoint", help="name of the checkpoint directory")
    parser.add_argument('--D', type=int, default=16, help="the number of RDBs")
    parser.add_argument('--C', type=int, default=8, help="the number of conv layers in each RDB")
    parser.add_argument('--G', type=int, default=64, help="the channel of feature maps")
    parser.add_argument('--G0', type=int, default=64, help="the channel of feature maps")
    parser.add_argument('--kernel_size', type=int, default=3, help="the size of kernel")
    args = parser.parse_args(args=[])

    date = '2022.09.18'
    save_path=args.result_dir
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    net = PSRNet(scale_factor = args.scale,
              in_channels=args.c_dim,
              num_blocks = args.D,
              num_layers = args.C,
              )
    net.to(device)
    checkpoint_path = os.path.join(args.checkpoint_dir, date)
    checkpoint = torch.load(checkpoint_path + '/best_checkpoint_epoch.pkl')
    net.load_state_dict(checkpoint['model_state_dict'])

    print('\ncheckpoint load success!')
    img_list = sorted(glob.glob(os.path.join(args.test_set, '*.png')))
    # label_list = sorted(glob.glob(os.path.join('test/label', '*.png')))
    net.eval()
    with torch.no_grad():
        totensor = transforms.ToTensor()
        for j in range(len(img_list)):
            img=cv2.imread(img_list[j], -1)/255.0
            img = totensor(img)
            img.unsqueeze_(0)
            img=img.type(torch.FloatTensor)
            img = img.to(device)
            # 前向传播forward
            pred = net(img)

            output = pred.squeeze(0)
            output=output.cuda().data.cpu().numpy()
            output=output.transpose(1,2,0)
            output=np.clip(output*255.0,0,255)
            s0, dolp,aop = cal_stokes_dolp(output)
            img_name = img_list[j].split('/')[-1].split('.')[0]
            save_merge_result(output,img_name,save_path)
            save_depart_result(output,s0,dolp,aop,img_name,save_path)
            print(f'Testing. Date: {date}. Process:{j + 1}/{len(img_list)}.')
