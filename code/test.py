import os, random, torch, cv2
import numpy as np
from argparse import ArgumentParser
from model import Net
from utils import *
from skimage.metrics import structural_similarity as ssim
from time import time

parser = ArgumentParser("DCCM-Net")
parser.add_argument("--epoch", type=int, default=30000)
parser.add_argument("--phase_num", type=int, default=9)
parser.add_argument("--model_dir", type=str, default="weight")
parser.add_argument("--data_dir", type=str, default="../data")
parser.add_argument("--data_type", type=str, default="nucleus")
parser.add_argument("--gpu_list", type=str, default="0")
parser.add_argument("--num_feature", type=int, default=128)
parser.add_argument("--result_dir", type=str, default="result")

args = parser.parse_args()

epoch = args.epoch
Np = args.phase_num
nf = args.num_feature

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_list
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# fixed seed for reproduction
seed = 2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# training set info
print("reading files ...")
start_time = time()
A32 = np.load(os.path.join(args.data_dir, "A_32.npy")).astype(np.float32) / 32.0
A128 = np.load(os.path.join(args.data_dir, "A_128.npy")).astype(np.float32) / 128.0
test_X = np.load(os.path.join(args.data_dir, args.data_type, "test_X.npy")).astype(np.float32)
test_X = test_X / test_X.max() * 255.0
test_Y32 = np.load(os.path.join(args.data_dir, args.data_type, "test_Y32.npy")).astype(np.float32) / 32.0
test_Y128 = np.load(os.path.join(args.data_dir, args.data_type, "test_Y128.npy")).astype(np.float32) / 128.0
test_Y32, test_Y128, A32, A128 = map(torch.from_numpy, [test_Y32, test_Y128, A32, A128])
print("read time", time() - start_time)

model = Net(Np, A32, A128, nf).to(device)
print("device =", device)

model_dir = "./%s/%s/layer_%d_f_%d" % (args.model_dir, args.data_type, Np, nf)
model.load_state_dict(torch.load("./%s/net_params_%d.pkl" % (model_dir, epoch))["model"])

result_dir = os.path.join(args.result_dir, args.data_type)
os.makedirs(result_dir, exist_ok=True)

test_X = test_X.reshape(-1, 128, 128)
test_Y32 = test_Y32.to(device)
test_Y128 = test_Y128.to(device)

def test(q, sr):
    if sr == 4:
        test_Y = test_Y32
    elif sr == 1:
        test_Y = test_Y128
    with torch.no_grad():
        cur_q = torch.tensor([[q]], device=device)
        pred_test_X = model(test_Y, cur_q, sr).reshape(-1, 128, 128)
        pred_test_X = (pred_test_X.clamp(0.0, 1.0) * 255.0).detach().cpu().numpy()
        PSNR_list, SSIM_list = [], []
        for i in range(pred_test_X.shape[0]):
            test_image = pred_test_X[i]
            ground_truth = test_X[i]
            PSNR = psnr(test_image, ground_truth)
            SSIM = ssim(test_image, ground_truth, data_range=255)
            PSNR_list.append(PSNR)
            SSIM_list.append(SSIM)
            cv2.imwrite(os.path.join(result_dir, "%d_q_%d_sr_%d_PSNR_%.2f_SSIM_%.4f.png" % (i, q, sr, PSNR, SSIM)), test_image)
            cv2.imwrite(os.path.join(result_dir, "%d_GT.png" % (i,)), ground_truth)
    return np.mean(PSNR_list), np.mean(SSIM_list)

for sr, q in [(4, 50), (4, 512), (1, 1000), (1, 2500), (1, 5000)]:
    cur_psnr, cur_ssim = test(q, sr)
    log_data = "SR is %d, Q is %d, PSNR is %.2f, SSIM is %.4f." % (sr, q, cur_psnr, cur_ssim)
    print(log_data)