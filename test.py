import numpy as np
import argparse
import time
import random
import os
from os import path as osp
from termcolor import colored
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler
from Datasets.ChangeAir_cvpr import Changeair_cvpr_Dataset
from TartanVO import TartanVO
from utils_training.utils_CNN import load_checkpoint, save_checkpoint, boolean_string
from utils_training.utils_load_model import load_model
from tensorboardX import SummaryWriter

from utils.image_transforms import ArrayToTensor, ToTensor, Compose, CropCenter, dataset_intrinsics, DownscaleFlow, plot_traj, visflow, plot_traj_3d, RandomCropandResize, RandomResizeCrop
from itertools import chain
from evaluator.tartanair_evaluator import TartanAirEvaluator
from Datasets.transformation import ses2poses_quat
from utils.evaluate import test
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel


if __name__ == "__main__":
 # Argument parsing
    parser = argparse.ArgumentParser(description='GLU-Net train script')
    # Paths
    parser.add_argument('--name_exp', type=str,
                        default=time.strftime('%Y_%m_%d_%H_%M'),
                        help='name of the experiment to save')
    parser.add_argument('--pre_loaded_training_dataset', default=False, type=boolean_string,
                        help='Synthetic training dataset is already created and saved in disk ? default is False')
    parser.add_argument('--training_data_dir', type=str,
                        help='path to directory containing original images for training if --pre_loaded_training_'
                             'dataset is False or containing the synthetic pairs of training images and their '
                             'corresponding flow fields if --pre_loaded_training_dataset is True')
    parser.add_argument('--data_name', default=False, type=str,
                        help='dataset name')
    parser.add_argument('--evaluation_data_dir', type=str, default='/home/jovyan/datasets/tartanair_cvpr/mono',
                        help='path to directory containing original images for validation')
    parser.add_argument('--snapshots', type=str, default='./snapshots')
    parser.add_argument('--pretrained_flownet', dest='pretrained_flownet', default=None,
                       help='path to pre-trained flownet model')
    parser.add_argument('--pretrained_posenet', dest='pretrained_posenet', default=None,
                       help='path to pre-trained posenet model')
    parser.add_argument('--pretrained_model', dest='pretrained_model', default=None,
                       help='path to pre-trained vo model')

    # Optimization parameters
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--momentum', type=float,
                        default=4e-4, help='momentum constant')
    parser.add_argument('--start_epoch', type=int, default=-1,
                        help='start epoch')
    parser.add_argument('--n_epoch', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='training batch size')
    parser.add_argument('--n_threads', type=int, default=8,
                        help='number of parallel threads for dataloaders')
    parser.add_argument('--weight-decay', type=float, default=4e-4,
                        help='weight decay constant')
    parser.add_argument('--div_flow', type=float, default=1.0,
                        help='div flow')
    parser.add_argument('--seed', type=int, default=1986,
                        help='Pseudo-RNG seed')
    parser.add_argument('--image-width', type=int, default=640,
                        help='image width (default: 640)')
    parser.add_argument('--image-height', type=int, default=448,
                        help='image height (default: 480)') 
    parser.add_argument('--random-crop-center',  action='store_true', default=False)
    parser.add_argument('--fix-ratio',  action='store_true',  default=False)
    parser.add_argument('--test-seq',  type=str, default='MH001')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # datasets, pre-processing of the images is done within the network function !
    source_img_transforms = transforms.Compose([ArrayToTensor(get_float=False)])
    target_img_transforms = transforms.Compose([ArrayToTensor(get_float=False)])
    valid_transform = Compose([CropCenter((args.image_height, args.image_width)), DownscaleFlow(), ToTensor()]) #only when valid
    #train_transform = Compose([RandomCropandResize((args.image_height, args.image_width), scale=(0.4, 1.0), ratio=(0.5, 2.0)), DownscaleFlow(), ToTensor()])
    train_transform = Compose([RandomResizeCrop((args.image_height, args.image_width), max_scale=2.5, keep_center=args.random_crop_center, fix_ratio=args.fix_ratio), DownscaleFlow(), ToTensor()])
    #train_transform = valid_transform

    flow_transform = transforms.Compose([ArrayToTensor()]) # just put channels first and put it to float
    test_dataset = Changeair_cvpr_Dataset(root=args.evaluation_data_dir, data_name = args.data_name, 
                                        source_image_transform=source_img_transforms,
                                        target_image_transform=target_img_transforms,
                                        flow_transform=flow_transform,
                                        co_transform=None, valid_transform = valid_transform, train_transform = train_transform,
                                        focalx = 320.0, focaly = 320.0, centerx = 320.0, centery = 240.0)

    # Dataloader
    test_dataloader = DataLoader(test_dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.n_threads)

    # models
    vonet = TartanVO()
    print(colored('==> ', 'blue') + 'TartanVO created.')

    if args.pretrained_model is not None:
        modelname = args.pretrained_model
        vonet = load_model(vonet, modelname)

    vonet = nn.DataParallel(vonet)
    vonet = vonet.to(device)

    cur_snapshot = args.name_exp
    save_path = osp.join(args.snapshots, cur_snapshot)
    if not osp.isdir(save_path):
        os.makedirs(save_path)

    results_dir = os.path.join(save_path, 'results')
    if not osp.isdir(results_dir):
        os.mkdir(results_dir)

    test_motionlist = np.array([])
    test_motionlist = test(vonet, test_motionlist, 
                   test_dataloader, 
                   device,
                   save_path=os.path.join(save_path, 'test'),
                   apply_mask=False,
                   sparse=False)

    gt_pose_file = os.path.join('/home/jovyan/datasets/tartanair_cvpr', 'mono_gt', args.data_name + '.txt')
    print(gt_pose_file)
    poselist = ses2poses_quat(np.array(test_motionlist))
    # calculate ATE, RPE, KITTI-RPE
    if gt_pose_file.endswith('.txt'):
        evaluator = TartanAirEvaluator()
        results = evaluator.evaluate_one_trajectory(gt_pose_file, poselist, scale=True, kittitype=False)

        print("==> valid ATE: %.4f,\t KITTI-R/t: %.4f, %.4f" %(results['ate_score'], results['kitti_score'][0], results['kitti_score'][1]))

        # save results and visualization
        epoch=0
        plot_traj_3d(results['gt_aligned'], results['est_aligned'], vis=False, savefigname=results_dir+'/traj_epoch_{}'.format(epoch)+'.png', title='ATE %.4f' %(results['ate_score']))
        np.savetxt(results_dir+ '/aligned_estimated_pose_epoch{}'.format(epoch) +'.txt',results['est_aligned'])
        np.savetxt(results_dir+ '/estimated_pose_epoch{}'.format(epoch) +'.txt', poselist)
    else:
        np.savetxt(results_dir+ '/estimated_pose_epoch{}'.format(epoch) +'.txt', poselist)