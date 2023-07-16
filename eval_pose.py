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
from datasets.training_dataset import HomoAffTps_Dataset
from datasets.load_pre_made_dataset import PreMadeDataset, Tartanair_TrainigDataset, Tartanair_TestDataset
from utils_training.optimize_GLUNet_with_adaptive_resolution import train_epoch, validate_epoch
from models.our_models.GLUNet import GLUNet_model
from models.our_models.VOFlowNet import VOFlowRes as FlowPoseNet
from utils_training.utils_CNN import load_checkpoint, save_checkpoint, boolean_string
from tensorboardX import SummaryWriter

from utils.image_transforms import ArrayToTensor, ToTensor, Compose, CropCenter, dataset_intrinsics, DownscaleFlow, plot_traj, visflow, plot_traj_3d, RandomCropandResize, RandomResizeCrop
from itertools import chain
from pose_evaluator.tartanair_evaluator import TartanAirEvaluator
from datasets.transformation import ses2poses_quat
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
    parser.add_argument('--evaluation_data_dir', type=str, default='/home/jovyan/datasets/tartanair_cvpr/mono',
                        help='path to directory containing original images for validation')
    parser.add_argument('--snapshots', type=str, default='./snapshots')
    parser.add_argument('--pretrained_flownet', dest='pretrained_flownet', default=None,
                       help='path to pre-trained flownet model')
    parser.add_argument('--pretrained_posenet', dest='pretrained_posenet', default=None,
                       help='path to pre-trained posenet model')
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
########################################################################################
# add vpr_args 

    parser.add_argument("--num_workers", type=int, default=8, help="_")
    parser.add_argument("--infer_batch_size", type=int, default=16,
                        help="Batch size for inference (validating and testing)")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="_")
    parser.add_argument("--fc_output_dim", type=int, default=512,
                        help="Output dimension of final fully connected layer")
    

########################################################################################

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
    test_dataset = Tartanair_TestDataset(root=args.evaluation_data_dir,
                                        source_image_transform=source_img_transforms,
                                        target_image_transform=target_img_transforms,
                                        flow_transform=flow_transform,
                                        co_transform=None, valid_transform = valid_transform, train_transform = train_transform,
                                        focalx = 320.0, focaly = 320.0, centerx = 320.0, centery = 240.0, seq= args.test_seq)
    
    # Dataloader
    test_dataloader = DataLoader(test_dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.n_threads)
    
    # models
    flownet = GLUNet_model(batch_norm=True, pyramid_type='VGG',
                         div=args.div_flow, evaluation=False, #오잉 evaluation=False해야지만 돌아감 
                         consensus_network=False,
                         cyclic_consistency=True,
                         dense_connection=True,
                         decoder_inputs='corr_flow_feat',
                         refinement_at_all_levels=False,
                         refinement_at_adaptive_reso=True)
    print(colored('==> ', 'blue') + 'GLU-Net created.')
    posenet = FlowPoseNet()
    print(colored('==> ', 'blue') + 'FlowPoseNet created.')
    
    if args.pretrained_flownet:
        checkpoint = torch.load(args.pretrained_flownet)
        #flownet.load_state_dict(checkpoint['state_dict'])
        state_dict = flownet.state_dict()
        for k1 in checkpoint['state_dict'].keys():
            if k1 in state_dict.keys():
                state_dict[k1] = checkpoint['state_dict'][k1].to(device)
        flownet.load_state_dict(state_dict)
        '''
        checkpoint = torch.load(args.pretrained_flownet)
        flownet.load_state_dict(checkpoint['state_dict'])
        '''
        cur_snapshot = args.name_exp
        

    if args.pretrained_posenet:
        checkpoint = torch.load(args.pretrained_posenet)
        state_dict = posenet.state_dict()
        for k1 in state_dict.keys():
            if 'state_dict' in checkpoint.keys():
                state_dict[k1] = checkpoint['state_dict'][k1].to(device)
            else:
                state_dict[k1] = checkpoint['module.flowPoseNet.'+str(k1)].to(device)
        posenet.load_state_dict(state_dict)
 
    flownet = nn.DataParallel(flownet)
    flownet = flownet.to(device)
    posenet = nn.DataParallel(posenet)
    posenet = posenet.to(device)

    save_path = osp.join(args.snapshots, cur_snapshot)
    if not osp.isdir(save_path):
        os.makedirs(save_path)

    results_dir = os.path.join(save_path, 'trajectory')
    if not osp.isdir(results_dir):
        os.mkdir(results_dir)
    
    # Validation
    test_motionlist = np.array([])
    test_motionlist = test(flownet, posenet, test_motionlist, 
                   test_dataloader, 
                   device,
                   save_path=os.path.join(save_path, 'test'),
                   apply_mask=False,
                   sparse=False)

    gt_pose_file = os.path.join(args.evaluation_data_dir, 'mono_gt', args.test_seq + '.txt')
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
    '''
############################## VPR validation ###################################################################
    from for_vpr.vpr_test import test as test_vpr
    if "changeair" in args.evaluation_data_dir:
        from for_vpr.vpr_test_dataset import ChangeAir_CVPR as Changeair_VPR_TestDataset
        vpr_test_dataset = Changeair_VPR_TestDataset(dataset_folder= '/home/jovyan/datasets/changeair_cvpr/mono' , positive_dist_threshold=0.5)
    elif "tartanair" in  args.evaluation_data_dir:
        from for_vpr.vpr_test_dataset import TartanAir_CVPR as TartanAir_VPR_TestDataset
        vpr_test_dataset = TartanAir_VPR_TestDataset(dataset_folder = args.evaluation_data_dir, positive_dist_threshold=0.5)
    vpr_result = test_vpr(args, vpr_test_dataset, flownet.module.pyramid)


    print("==> vpr result: " + vpr_result[1])
    '''
######################################################################################################################################