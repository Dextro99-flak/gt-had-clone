# https://github.com/jeline0110/GT-HAD
import os
import numpy as np
import torch
import torch.optim as Optim
import scipy.io as sio
import pdb
from net import Net
from sklearn.metrics import roc_auc_score, roc_curve
import shutil
from utils import get_params, img2mask, seed_dict
import random
from progress.bar import Bar
import time 
import torch.nn as nn 
from torch.utils.data import DataLoader
from data import DatasetHsi
from block import Block_fold, Block_embedding

dtype = torch.cuda.FloatTensor
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
data_dir = '../../data/'
save_dir = '../../results/'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(file):
    # set random seed 
    # **************************************************************************************************************
    seed = seed_dict[file]
    set_seed(seed)
    # data process
    # **************************************************************************************************************
    print(file)
    data_path = data_dir + file + '.mat'
    save_subdir = os.path.join(save_dir, file)
    if not os.path.exists(save_subdir):
        os.makedirs(save_subdir)
    # load data
    mat = sio.loadmat(data_path)
    img_np = mat['data']
    img_np = img_np.transpose(2, 0, 1) # b, h, w
    img_np = img_np - np.min(img_np)
    img_np = img_np / np.max(img_np) # [0, 1]
    gt = mat['map']
    img_var = torch.from_numpy(img_np).type(dtype)
    band, row, col = img_var.size()
    img_var = img_var[None, :]
    # set block functions and init dataloader
    # **************************************************************************************************************
    patch_size = 3 
    patch_stride = 3
    block_size = patch_size * patch_stride # block_size is the sliding window size
    data_set = DatasetHsi(img_var, wsize=block_size, wstride=3)
    block_fold = Block_fold(wsize=block_size, wstride=3)
    data_loader = DataLoader(data_set, batch_size=64, shuffle=True, drop_last=False)
    # model setup
    # **************************************************************************************************************
    # net with differentiable soft gating
    lambda_gate = 0.01  # Hyperparameter for gate entropy regularization
    net = Net(in_chans=band, embed_dim=64, patch_size=patch_size, 
        patch_stride=patch_stride, mlp_ratio=2.0, attn_drop=0.0, drop=0.0, lambda_gate=lambda_gate)
    net = net.cuda()
    s = sum(np.prod(list(p.size())) for p in net.parameters())
    print ('Number of params: %d' % s)
    # loss
    mse = torch.nn.MSELoss().type(dtype)
    # optim
    LR = 1e-3 # 2e-5
    p = get_params(net)
    optimizer = Optim.Adam(p, lr=LR)
    print('Starting optimization with ADAM')
    print(f'Gate regularization weight (lambda_gate): {lambda_gate}')
    # train
    # **************************************************************************************************************
    end_iter = 150
    bar = Bar('Processing', max=end_iter)
    data_num = data_set.__len__()
    avgpool = nn.AvgPool3d(kernel_size=(5, 3, 3), stride=(1, 1, 1), padding=(2, 1, 1))
    
    # start train
    start = time.time()
    for iter in range(1, end_iter + 1):
        for idx, batch_data in enumerate(data_loader):
            optimizer.zero_grad()
            # input -> net -> output (single forward pass)
            net_gt, net_input, block_idx = batch_data['block_gt'], batch_data['block_input'], batch_data['index'].cuda()
            
            # Forward pass through network with differentiable soft gating
            net_out = net(net_input)
            
            # Reconstruction loss
            recon_loss = mse(net_out, net_gt)
            
            # Gate entropy regularization loss (prevents gate collapse)
            gate_loss = net.compute_gate_loss()
            
            # Combined loss
            total_loss = recon_loss + lambda_gate * gate_loss
            
            total_loss.backward()
            optimizer.step()
        
        bar.next()

        # start test 
        if iter == end_iter:
            bar.finish()
            infer_loader = DataLoader(data_set, batch_size=64, shuffle=False, drop_last=False)
            net = net.eval()
            infer_res_list = []

            for idx, data in enumerate(infer_loader):
                infer_in = data['block_input']
                infer_idx = data['index'].cuda()
                # Inference: single forward pass (no external routing state needed)
                with torch.no_grad():
                    infer_out = net(infer_in)
                infer_res = torch.abs(infer_in - infer_out) ** 2
                infer_res = avgpool(infer_res)
                infer_res_list.append(infer_res)

            infer_res_out = torch.cat(infer_res_list, dim=0)
            infer_res_back = block_fold(infer_res_out.detach(), data_set.padding, row, col)
            residual_np = img2mask(infer_res_back)
            # cal auc
            auc = roc_auc_score(gt.flatten(), residual_np.flatten())
            print('Auc: %.4f' % auc)
            # running time
            end = time.time()
            print("Runtime：%.2f" % (end - start)) 
            # save results
            fpr, tpr, thre = roc_curve(gt.flatten(), residual_np.flatten())
            map_path = os.path.join(save_subdir, "GT-HAD_map.mat")
            sio.savemat(map_path, {'show': residual_np})
            roc_path = os.path.join(save_subdir, "GT-HAD_roc.mat")
            sio.savemat(roc_path, {'PD': tpr, 'PF': fpr})

            return

if __name__ == "__main__":
    for file in ['los-angeles-1']: 
            #     ['los-angeles-1', 'los-angeles-2', 'gulfport', 
            # 'texas-goast', 'cat-island', 'pavia']:
        main(file)
