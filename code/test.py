import os, glob, random, torch, cv2
import numpy as np
from argparse import ArgumentParser
from model import Net
from utils import *
from skimage.metrics import structural_similarity as ssim
from time import time
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('--epoch', type=int, default=30000)
parser.add_argument('--phase_num', type=int, default=9)
parser.add_argument('--model_dir', type=str, default='model')
parser.add_argument('--data_dir', type=str, default='/workspace/14/CV/data')
parser.add_argument('--data_type', type=str, default='20240127_flureoscent_microsphere')
parser.add_argument('--result_dir', type=str, default='result_20240217')
parser.add_argument('--gpu_list', type=str, default='0')
parser.add_argument('--num_feature', type=int, default=128)
parser.add_argument('--gq', type=int, default=25)

args = parser.parse_args()

epoch = args.epoch
Np = args.phase_num
nf = args.num_feature
gq = args.gq

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_list
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# fixed seed for reproduction
seed = 2023
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
GPU_num = torch.cuda.device_count()

# training set info
print('reading files...')
start_time = time()
A32 = np.load(os.path.join(args.data_dir, 'A_32.npy')).astype(np.float32) / 32.0
A128 = np.load(os.path.join(args.data_dir, 'A_128.npy')).astype(np.float32) / 128
test_X = np.load(os.path.join(args.data_dir, args.data_type, 'test_X.npy')).astype(np.float32) * 255.0 / 65535.0
test_X_WF = np.load(os.path.join(args.data_dir, args.data_type, 'test_X_WF.npy')).astype(np.float32) * 255.0 / 65535.0
test_Y32 = np.load(os.path.join(args.data_dir, args.data_type, 'test_Y32_%d.npy' % (gq,))).astype(np.float32) / 32.0
test_Y128 = np.load(os.path.join(args.data_dir, args.data_type, 'test_Y128_%d.npy' % (gq,))).astype(np.float32) / 128.0
test_Y32, test_Y128, A32, A128 = map(torch.from_numpy, [test_Y32, test_Y128, A32, A128])
print('read time', time() - start_time)

model = Net(Np, A32, A128, nf)
model = torch.nn.DataParallel(model).to(device)
model_dir = './%s/%s/layer_%d_f_%d_gq_%d' % (args.model_dir, args.data_type, Np, nf, gq)
model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (model_dir, epoch))['model'])

result_dir = os.path.join(args.result_dir, args.data_type)
os.makedirs(result_dir, exist_ok=True)

test_X = test_X.reshape(-1, 128, 128)
test_X_WF = test_X_WF.reshape(-1, 128, 128)
test_Y32 = test_Y32.to(device)
test_Y128 = test_Y128.to(device)

def test(q, sr):
    if sr == 4:
        test_Y = test_Y32
    elif sr == 1:
        test_Y = test_Y128
    with torch.no_grad():
        pred_test_X = model(test_Y, torch.tensor([[q]], device=device), sr).reshape(-1, 128, 128)
        pred_test_X = (pred_test_X.clamp(0, 1) * 255.0).detach().cpu().numpy()
        PSNR_list, SSIM_list = [], []
        for i in range(pred_test_X.shape[0]):
            test_image = pred_test_X[i]
            ground_truth = test_X[i]
            ground_truth_WF = test_X_WF[i]
            
            PSNR = psnr(test_image, ground_truth)
            SSIM = ssim(test_image, ground_truth, data_range=255)

            output_path = os.path.join(result_dir, 'img_%d_%dx_q_%d_gq_%d_K_%d_f_%d_PSNR_%.2f_SSIM_%.4f.png' % (i, sr, q, gq, Np, nf, PSNR, SSIM))
            cv2.imwrite(output_path, test_image)

            output_path = os.path.join(result_dir, 'img_%d_GT.png' % (i))
            cv2.imwrite(output_path, ground_truth)

            output_path = os.path.join(result_dir, 'img_%d_WF.png' % (i))
            cv2.imwrite(output_path, ground_truth_WF)

            PSNR_list.append(PSNR)
            SSIM_list.append(SSIM)

    return float(np.mean(PSNR_list)), float(np.mean(SSIM_list))

for sr in [1, 4]:
    for q in range(50, 501, 10):
        avg_psnr, avg_ssim = test(q, sr)
        print('Q is %d, %dx PSNR/SSIM is %.2f/%.4f' % (q, sr, avg_psnr, avg_ssim))
