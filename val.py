from __future__ import print_function, division
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import os
import numpy as np
from dataset import CrowdCountingDataset
from model import PSNet
# %matplotlib inline


################################################################################
# test model
################################################################################
def val():
    """
        val model
    """
    # set hyperparameter
    TEST_IMG_DIR = 'datasets/ShanghaiTech_Dataset/part_A_final/test_data/images' # the directory path for storing test set images
    workers = 2  # Number of workers for dataloader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # choose to run on cpu or cuda

    # bset MAE, MSE
    # BEST_MAE = float("inf")
    BEST_MAE = 300
    # BEST_MSE = float("inf")
    BEST_MSE = 300

    # load data
    MEAN = [0.485, 0.456, 0.406] # mean
    STD = [0.229, 0.224, 0.225] # std
    normalize = transforms.Normalize(
        mean=MEAN,
        std=STD
    )
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize]
    )
    # define valloader
    val_dataset = CrowdCountingDataset(TEST_IMG_DIR, transforms = val_transform, scale = 8, mode = 'test')
    val_loader = DataLoader(val_dataset, batch_size = 1, num_workers=workers)

    # define model
    model = PSNet().float()
    model = model.to(device)

    # load model weights
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('Checkpoint'), 'Error: no Checkpoint directory found!'
    state = torch.load('Checkpoint/models/ckpt_best.pth')
    model.load_state_dict(state['net'])
    BEST_MAE = state['mae']
    BEST_MSE = state['mse']
    epoch = state['epoch']

    # loss function
    cosloss = nn.CosineSimilarity(dim=1, eps=1e-6).to(device)

    ############################
    # test
    ############################
    # test mode
    model.eval()
    # loss
    mae = 0.0
    mse = 0.0
    sum_att_loss = 0.0
    # number of iterations
    cnt = 0
    with torch.no_grad():
        for data in val_loader:
            cnt += 1

            # load data
            image, gt, gt_density = data
            image, gt_density = image.float(), gt_density.float()
            image, gt_density = image.to(device), gt_density.to(device)

            # forward and backward propagation
            pr_density, attention_map_1, attention_map_2, attention_map_3 = model(image)

            attention_loss = 0.
            for attention_map in (attention_map_1, attention_map_2, attention_map_3):
                attention_map_sum = attention_map[:, 0:1] + attention_map[:, 1:2] + attention_map[:, 2:3] + \
                                    attention_map[:, 3:4]
                attention_loss_temp = 0.
                for i in range(4):
                    attention_loss_temp += torch.sum(
                        cosloss(attention_map[:, i:(i + 1)].contiguous().view(image.size(0), -1),
                                ((attention_map_sum - attention_map[:, i:(i + 1)]) / 3).contiguous().view(image.size(0), -1))) / image.size(0)
                attention_loss += (attention_loss_temp / 4)
            attention_loss /= 3
            sum_att_loss += attention_loss.item()

            # record real results and predicted results
            pr_density = pr_density.cpu().detach().numpy()
            pr = np.sum(pr_density)
            mae += np.abs(gt - pr)
            mse += np.abs(gt - pr) ** 2

    # calculate loss
    mae_loss = mae / cnt
    mse_loss = np.sqrt(mse / cnt)
    att_loss = sum_att_loss / cnt

    # print log
    print('EPOCH: %d\tMAE: %.4f\tMSE: %.4f\tBEST_MAE: %.4f\tBEST_MSE: %.4f\tATT_LOSS: %.4f'
          % (epoch, mae_loss, mse_loss, BEST_MAE, BEST_MSE, att_loss))


################################################################################
# main function
################################################################################
if __name__ == '__main__':
    # test model
    val()

    pass
