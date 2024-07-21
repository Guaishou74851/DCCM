import os, random, torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser
from model import Net
from utils import *
from skimage.metrics import structural_similarity as ssim
from time import time
from tqdm import tqdm

parser = ArgumentParser("DCCM-Net")
parser.add_argument("--start_epoch", type=int, default=0)
parser.add_argument("--end_epoch", type=int, default=30000)
parser.add_argument("--phase_num", type=int, default=9)
parser.add_argument("--learning_rate", type=float, default=2e-4)
parser.add_argument("--model_dir", type=str, default="weight")
parser.add_argument("--data_dir", type=str, default="../data")
parser.add_argument("--data_type", type=str, default="nucleus")
parser.add_argument("--log_dir", type=str, default="log")
parser.add_argument("--save_interval", type=int, default=1000)
parser.add_argument("--gpu_list", type=str, default="0")
parser.add_argument("--num_feature", type=int, default=128)

args = parser.parse_args()

start_epoch, end_epoch = args.start_epoch, args.end_epoch
lr = args.learning_rate
Np = args.phase_num
nf = args.num_feature
batch_size = 16

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_list
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# fixed seed for reproduction
seed = 2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

GPU_num = torch.cuda.device_count()

# training set info
print("reading files ...")
start_time = time()
A32 = np.load(os.path.join(args.data_dir, "A_32.npy")).astype(np.float32) / 32.0
A128 = np.load(os.path.join(args.data_dir, "A_128.npy")).astype(np.float32) / 128.0
train_X = np.load(os.path.join(args.data_dir, args.data_type, "train_X.npy")).astype(np.float32)
train_X = train_X / train_X.max()
train_Y32 = np.load(os.path.join(args.data_dir, args.data_type, "train_Y32.npy")).astype(np.float32) / 32.0
train_Y128 = np.load(os.path.join(args.data_dir, args.data_type, "train_Y128.npy")).astype(np.float32) / 128.0
test_X = np.load(os.path.join(args.data_dir, args.data_type, "test_X.npy")).astype(np.float32)
test_X = test_X / test_X.max() * 255.0
test_Y32 = np.load(os.path.join(args.data_dir, args.data_type, "test_Y32.npy")).astype(np.float32) / 32.0
test_Y128 = np.load(os.path.join(args.data_dir, args.data_type, "test_Y128.npy")).astype(np.float32) / 128.0
train_X, train_Y32, train_Y128, test_Y32, test_Y128, A32, A128 = map(torch.from_numpy, [train_X, train_Y32, train_Y128, test_Y32, test_Y128, A32, A128])
print("read time", time() - start_time)

model = Net(Np, A32, A128, nf).to(device)
print("device =", device)

class MyDataset(Dataset):
    def __init__(self):
        self.train_XY = torch.cat([train_X, train_Y32, train_Y128], dim=1)

    def __getitem__(self, index):
        return self.train_XY[random.randint(0, self.train_XY.shape[0] - 1)]

    def __len__(self):
        return batch_size * 50

dataloader = DataLoader(dataset=MyDataset(), batch_size=batch_size, num_workers=8)
optimizer = torch.optim.Adam([{"params":model.parameters(), "initial_lr":lr}], lr=lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25000, 29000], gamma=0.1, last_epoch=start_epoch-1)

model_dir = "./%s/%s/layer_%d_f_%d" % (args.model_dir, args.data_type, Np, nf)
log_path = "./%s/%s/layer_%d_f_%d.txt" % (args.log_dir, args.data_type, Np, nf)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(os.path.join(args.log_dir, args.data_type), exist_ok=True)

test_X = test_X.reshape(-1, 128, 128)
test_Y32 = test_Y32.to(device)
test_Y128 = test_Y128.to(device)

def test(q, sr):
    if sr == 4:
        test_Y = test_Y32
    elif sr == 1:
        test_Y = test_Y128
    with torch.no_grad():
        cur_q = torch.tensor([[q]], device=device).expand(GPU_num, 1)
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
    return np.mean(PSNR_list), np.mean(SSIM_list)

if start_epoch > 0:
    checkpoint = torch.load("./%s/net_params_%d.pkl" % (model_dir, start_epoch))
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

print("start training...")
for epoch_i in range(start_epoch+1, end_epoch+1):
    start_time = time()
    loss_avg, iter_num = 0.0, 0
    for data in tqdm(dataloader):
        data = data.to(device)
        x, y = data[:, :16384], data[:, 16384:]
        y32, y128 = y[:, :512], y[:, 512:]
        cur_q = torch.randint(low=1, high=512, size=(GPU_num,batch_size//GPU_num), device=device)
        loss32 = (model(y32, cur_q, 4) - x).abs().mean()
        cur_q = torch.randint(low=1, high=5001, size=(GPU_num,batch_size//GPU_num), device=device)
        loss128 = (model(y128, cur_q, 1) - x).abs().mean()
        loss = loss32 + loss128
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        iter_num += 1
        loss_avg += loss.item()
    scheduler.step()
    loss_avg /= iter_num
    log_data = "[%d/%d] Average loss: %f, time cost: %.2fs, current lr is %f." % (epoch_i, end_epoch, loss_avg, time() - start_time, scheduler.get_last_lr()[0])
    print(log_data)
    with open(log_path, "a") as log_file:
        log_file.write(log_data + "\n")
    if epoch_i % args.save_interval == 0:
        torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict()}, "./%s/net_params_%d.pkl" % (model_dir, epoch_i))
    if epoch_i == 1 or epoch_i % 300 == 0:
        for sr, q in [(4, 50), (4, 512), (1, 1000), (1, 2500), (1, 5000)]:
            cur_psnr, cur_ssim = test(q, sr)
            log_data = "SR is %d, Q is %d, PSNR is %.2f, SSIM is %.4f." % (sr, q, cur_psnr, cur_ssim)
            print(log_data)
            with open(log_path, "a") as log_file:
                log_file.write(log_data + "\n")