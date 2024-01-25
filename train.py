import argparse
import time
import cv2
import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity
import torch.optim as optim
from workflow import WorkFlow, TorchFlow
from arguments import get_args
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from Datasets.TartanAir import Tartanair_TrainigDataset
from Datasets.ChangeAir import Changeair_Dataset
from Datasets.ChangeAir_cvpr import Changeair_cvpr_Dataset
from Datasets.GSVCitiesDataset import GSVCitiesDataset
from Datasets import PittsburgDataset
import torch.optim.lr_scheduler as lr_scheduler
from utils_training.optimize_VONet_with_adaptive_resolution import train_epoch, validate_epoch
from utils.image_transforms import ArrayToTensor, ToTensor, Compose, CropCenter, dataset_intrinsics, DownscaleFlow, plot_traj, visflow, plot_traj_3d, RandomCropandResize, RandomResizeCrop
from utils_training.utils_CNN import load_checkpoint, save_checkpoint, boolean_string
from utils_training.utils_load_model import load_model
from tensorboardX import SummaryWriter
from Datasets.transformation import ses2poses_quat
from evaluator.tartanair_evaluator import TartanAirEvaluator
from Datasets.utils import ToTensor, Compose, CropCenter, dataset_intrinsics, DownscaleFlow, plot_traj, plot_traj_3d, visflow
from evaluator.evaluator_base import quats2SEs
from evaluator.trajectory_transform import trajectory_transform, rescale
from evaluator.transformation import pos_quats2SE_matrices, SE2pos_quat
from Datasets.transformation import ses2poses_quat
from utils_training.optimize_VONet_with_adaptive_resolution import ATEEvaluator, transform_trajs
import random
import os
from os import path as osp
from termcolor import colored
import pickle
import time # for testing
from itertools import chain

from configs.default import get_cfg
from core.utils.misc import process_cfg
from loguru import logger as loguru_logger
from pathlib import Path


#from PretrainedVONet import PretrainedVONet
#from TartanVO import TartanVO

#from Network.GLAM import GLAM
from core.FlowFormer import build_flowformer
from Network.VO_mixvpr import VPRPosenet
#from Network.mixvpr import MixVPR
#from Network.VOFlowNet import VOFlowRes as FlowPoseNet
#from Network.helper import get_aggregator

if __name__ == '__main__':
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
    parser.add_argument('--test_data_dir', type=str,
                        help='path to directory containing original images for test if --pre_loaded_test_'
                             'dataset is False or containing the synthetic pairs of test images and their '
                             'corresponding flow fields if --pre_loaded_training_dataset is True')
    parser.add_argument('--evaluation_data_dir', type=str,
                        help='path to directory containing original images for validation if --pre_loaded_training_'
                             'dataset is False or containing the synthetic pairs of validation images and their '
                             'corresponding flow fields if --pre_loaded_training_dataset is True')
    parser.add_argument('--snapshots', type=str, default='./snapshots')
    parser.add_argument('--pretrained_flownet', dest='pretrained_flownet', default=None,
                       help='path to pre-trained flownet model')
    parser.add_argument('--pretrained_posenet', dest='pretrained_posenet', default=None,
                       help='path to pre-trained posenet model')
    parser.add_argument('--pretrained_model', dest='pretrained_model', default=None,
                       help='path to pre-trained vo model')
    parser.add_argument('--pose-file', default='',
                        help='test trajectory gt pose file, used for scale calculation, and visualization (default: "")')
    parser.add_argument('--img_per_place', type=int, default=4,
                        help='number of training epochs')
    parser.add_argument('--min_img_per_place', type=int, default=4,
                        help='number of training epochs')


    # Optimization parameters
    parser.add_argument('--momentum', type=float,
                        default=4e-4, help='momentum constant')
    parser.add_argument('--start_epoch', type=int, default=-1,
                        help='start epoch')
    parser.add_argument('--n_epoch', type=int, default=25,
                        help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='training batch size')
    parser.add_argument('--n_threads', type=int, default=2,
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
                        help='image height (default: 640)') 
    parser.add_argument('--random-crop-center',  action='store_true', default=False)
    parser.add_argument('--fix-ratio',  action='store_true',  default=False)
    parser.add_argument('--resume-e2e',  action='store_true',  default=False)
    parser.add_argument('--worker-num', type=int, default=16,
                        help='data loader worker number (default: 1)')
    parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 3e-4)')
    args = parser.parse_args()
    
    cfg = get_cfg()
    cfg.update(vars(args))
    process_cfg(cfg)
    loguru_logger.add(str(Path(cfg.log_dir) / 'log.txt'), encoding="utf8")
    loguru_logger.info(cfg)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406], 
                        'std': [0.229, 0.224, 0.225]}

    VIT_MEAN_STD = {'mean': [0.5, 0.5, 0.5], 
                    'std': [0.5, 0.5, 0.5]}

    TRAIN_CITIES = [
        'Bangkok',
        'BuenosAires',
        'LosAngeles',
        'MexicoCity',
        'OSL',
        'Rome',
        'Barcelona',
        'Chicago',
        'Madrid',
        'Miami',
        'Phoenix',
        'TRT',
        'Boston',
        'Lisbon',
        'Medellin',
        'Minneapolis',
        'PRG',
        'WashingtonDC',
        'Brussels',
        'London',
        'Melbourne',
        'Osaka',
        'PRS',
    ]
    
    image_size = (320, 320)
    source_img_transforms = transforms.Compose([ArrayToTensor(get_float=False)])
    target_img_transforms = transforms.Compose([ArrayToTensor(get_float=False)])
    valid_transform = Compose([CropCenter((args.image_height, args.image_width)), ToTensor()]) #only when valid
    train_transform = Compose([RandomResizeCrop((args.image_height, args.image_width), max_scale=2.5, keep_center=args.random_crop_center, fix_ratio=args.fix_ratio), ToTensor()])
    flow_transform = transforms.Compose([ArrayToTensor()])
    
    VPR_train_transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandAugment(num_ops=3, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN_STD['mean'], std=IMAGENET_MEAN_STD['std']),
    ])

    VPR_valid_transform = Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN_STD['mean'], std=IMAGENET_MEAN_STD['std'])])
    # training and validation dataset
    '''
    train_dataset = Changeair_Dataset(root=args.training_data_dir,
                                    source_image_transform=source_img_transforms,
                                    target_image_transform=target_img_transforms,
                                    flow_transform=flow_transform,
                                    co_transform=None, valid_transform = valid_transform, train_transform = train_transform,
                                    focalx = 320.0, focaly = 320.0, centerx = 320.0, centery = 240.0)

    val_dataset = Changeair_cvpr_Dataset(root=args.test_data_dir,
                                    source_image_transform=source_img_transforms,
                                    target_image_transform=target_img_transforms,
                                    flow_transform=flow_transform,
                                    co_transform=None, valid_transform = valid_transform, train_transform = train_transform,
                                    focalx = 320.0, focaly = 320.0, centerx = 320.0, centery = 240.0)
    '''
    train_dataset, val_dataset = Tartanair_TrainigDataset(root=args.training_data_dir,
                                    source_image_transform=source_img_transforms,
                                    target_image_transform=target_img_transforms,
                                    flow_transform=flow_transform,
                                    co_transform=None, valid_transform = valid_transform, train_transform = train_transform,
                                    focalx = 320.0, focaly = 320.0, centerx = 320.0, centery = 240.0)
    '''
    VPR_train_dataset = GSVCitiesDataset(
        cities=TRAIN_CITIES,
        img_per_place=args.img_per_place,
        min_img_per_place=args.min_img_per_place,
        random_sample_from_each_place=True,
        transform=VPR_train_transform)
    #import pdb; pdb.set_trace()
    VPR_val_dataset = PittsburgDataset.get_whole_test_set(
                        input_transform=VPR_valid_transform)
    
    #import pdb; pdb.set_trace()
    '''
    #Dataloader

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                        shuffle=True, drop_last=False, pin_memory=True, num_workers=args.worker_num//2)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, 
                                    shuffle=False, drop_last=False, pin_memory=True, num_workers=args.worker_num//4)

    '''
    VPR_train_dataloader = DataLoader(dataset=VPR_train_dataset, batch_size=4, shuffle=True, drop_last=False, pin_memory=True, num_workers=args.worker_num//2)
    
    VPR_valid_dataloader = DataLoader(dataset=VPR_val_dataset, batch_size=4, shuffle=False, drop_last=False, pin_memory=True, num_workers=args.worker_num//4)
    
    self.vonet = PretrainedVONet(intrinsic=self.args.intrinsic_layer, 
                        flowNormFactor=1.0, down_scale=args.downscale_flow, 
                        fixflow=args.fix_flow, pretrain=args.pretrain_model_name,
                        use_gru=args.use_gru)
    '''
    # models
    #vonet = TartanVO()
    #attention_net = GLAM() # 파라미터 입력해야됨
    #flownet = build_flowformer(cfg)
    #print(colored('==> ', 'blue') + 'Flowformer created.')
    #import pdb; pdb.set_trace()

    #target_layer_parameters = model.memory_encoder.cost_perceiver_encoder.input_layer_local.parameters()
    #total_parameters = sum(p.numel() for p in target_layer_parameters)

    #print("Total Parameters in target layer:", total_parameters)
    posenet = VPRPosenet(in_channels=256, in_h=40, in_w=40, out_channels=256, mix_depth=4, mlp_ratio=1, out_rows=9)
    print(colored('==> ', 'blue') + 'Posenet created.')    
    #vpr_net = get_aggregator(agg_arch, agg_config)
    #scd_net = TANet(self.args.encoder_arch, self.args.local_kernel_size, self.args.attn_stride,
                           #self.args.attn_padding, self.args.attn_groups, self.args.drtam, self.args.refinement)
    #loguru_logger.info("Parameter Count: %d" % count_parameters(model))
    
    if cfg.restore_ckpt is not None:
        print("[Loading ckpt from {}]".format(cfg.restore_ckpt))
        model.load_state_dict(torch.load(cfg.restore_ckpt), strict=True)
    
    optimizer = \
        optim.Adam(filter(lambda p: p.requires_grad, posenet.parameters()),
                   lr=args.lr,)
                   #weight_decay=args.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer,
                                         #milestones=[65, 75, 95],
                                         milestones=[12,21],#e2e 25
                                         gamma=0.2)#poselr
    '''
    optimizer = optim.AdamW(chain(flownet.parameters(),posenet.parameters()), lr=cfg.trainer.canonical_lr, weight_decay=cfg.trainer.adamw_decay, eps=cfg.trainer.epsilon)
    scheduler = lr_scheduler.OneCycleLR(optimizer, cfg.trainer.canonical_lr, epochs=25, steps_per_epoch=76567,
                pct_start=0.05, cycle_momentum=False, anneal_strategy=cfg.trainer.anneal_strategy)

    
    # load the whole model
    if args.pretrained_model is not None:
        modelname = args.pretrained_model
        vonet = load_model(vonet, modelname)
    '''
    #import pdb; pdb.set_trace()
    if args.pretrained_flownet:
        checkpoint = torch.load(args.pretrained_flownet)
        #flownet.load_state_dict(checkpoint['state_dict'])
        state_dict = flownet.state_dict()
        #import pdb; pdb.set_trace()
        for k1 in checkpoint['state_dict'].keys():
            #k1 = k1[7:]
            if k1 in state_dict.keys():
                state_dict[k1] = checkpoint['state_dict'][k1].to(device)
        flownet.load_state_dict(state_dict)
        '''
        checkpoint = torch.load(args.pretrained_flownet)
        flownet.load_state_dict(checkpoint['state_dict'])
        '''
        #cur_snapshot = args.name_exp
        

    if args.pretrained_posenet:
        checkpoint = torch.load(args.pretrained_posenet)
        state_dict = posenet.state_dict()
        for k1 in checkpoint['state_dict'].keys():
            if k1 in state_dict.keys():
                state_dict[k1] = checkpoint['state_dict'][k1].to(device)
        posenet.load_state_dict(state_dict)
    
    if not osp.isdir(args.snapshots):
        os.mkdir(args.snapshots)

    cur_snapshot = args.name_exp
    if not osp.isdir(osp.join(args.snapshots, cur_snapshot)):
        os.makedirs(osp.join(args.snapshots, cur_snapshot))

    with open(osp.join(args.snapshots, cur_snapshot, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)

    best_train = float("inf")
    start_epoch = 0

    save_path = osp.join(args.snapshots, cur_snapshot)
    train_writer = SummaryWriter(os.path.join(save_path, 'train'))
    test_writer = SummaryWriter(os.path.join(save_path, 'test'))

    #attention_net = nn.DataParallel(attention_net)
    #flownet = nn.DataParallel(flownet)
    posenet = nn.DataParallel(posenet)
    #vpr_net = nn.DataParallel(vpr_net)
    #scd_net = nn.DataParallel(TANet)
    #vonet = nn.DataParallel(vonet)
    #flownet = flownet.to(device)
    posenet = posenet.to(device)
    train_started = time.time()
    datastr = 'tartanair'
    for epoch in range(start_epoch, args.n_epoch):
        scheduler.step()
        print('starting epoch {}:  learning rate is {}'.format(epoch, scheduler.get_last_lr()[0]))
        results_dir = os.path.join(save_path, 'result')
        if not osp.isdir(results_dir):
            os.mkdir(results_dir)  
        
        train_loss_pose, train_last_batch_ate = train_epoch(posenet,
                                 optimizer,
                                 train_dataloader,
                                 #VPR_train_dataloader,
                                 device,
                                 epoch,
                                 train_writer,
                                 cfg,
                                 div_flow=args.div_flow,
                                 save_path=os.path.join(save_path, 'train'),
                                 apply_mask=False, results_dir=results_dir)#수정
        #train_writer.add_scalar('train loss flow', train_loss_flow, epoch)
        train_writer.add_scalar('train loss pose', train_loss_pose, epoch)
        #train_writer.add_scalar('train loss vpr', train_loss_vpr, epoch)
        #train_writer.add_scalar('train last batch ate', train_last_batch_ate, epoch)
        train_writer.add_scalar('learning_rate', scheduler.get_lr()[0], epoch)
        #print(colored('==> ', 'green') + 'Train average flow loss:', train_loss_flow)  
        print(colored('==> ', 'green') + 'Train average pose loss:', train_loss_pose)
        #print(colored('==> ', 'green') + 'Train average vpr loss:', train_loss_vpr)
        print(colored('==> ', 'green') + 'Train last batch ate:', train_last_batch_ate)  
        '''       
        # Validation
        valid_motionlist = np.array([])
        valid_motionlist_gt = np.array([])
        valid_loss_pose, val_last_batch_ate, motionlist, motionlist_gt = \
        validate_epoch(flownet, posenet, valid_motionlist, valid_motionlist_gt, val_dataloader, device, epoch=epoch, save_path=os.path.join(save_path, 'test'), div_flow=args.div_flow,
                           apply_mask=False, results_dir=results_dir) #수정)
        #test_writer.add_scalar('train loss flow', valid_loss_flow, epoch)
        test_writer.add_scalar('test loss pose', valid_loss_pose, epoch)
        #test_writer.add_scalar('test last batch ate', val_last_batch_ate, epoch)        
        #print(colored('==> ', 'blue') + 'Val average flow loss :', valid_loss_flow)
        print(colored('==> ', 'blue') + 'Val average pose loss:', valid_loss_pose)
        #print(colored('==> ', 'blue') + 'Val last batch ate:', val_last_batch_ate)

        print(colored('==> ', 'blue') + 'finished epoch :', epoch)
        
        poselist = ses2poses_quat(np.array(motionlist))
        poselist_gt = ses2poses_quat(np.array(motionlist_gt))

        gt_traj_trans, est_traj_trans, s = transform_trajs(poselist_gt, poselist, True)
        gt_SEs, est_SEs = quats2SEs(gt_traj_trans, est_traj_trans)
        ate_eval = ATEEvaluator()
        ate_score, gt_ate_aligned, est_ate_aligned = ate_eval.evaluate(poselist_gt, poselist, True)
        #ate_scorelist.append(ate_score)
        plot_traj_3d(gt_ate_aligned, est_ate_aligned, vis=False, savefigname=results_dir+'/valid_traj_epoch_{}'.format(str(epoch) + '.png'), title='ATE %.4f' %(ate_score))
        print('ate:', ate_score)
        '''
        is_best = train_loss_pose < best_train
        best_train = min(train_loss_pose, best_train)

        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': posenet.module.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict(),
                         'best_loss': best_train},
                        is_best, save_path, 'epoch_{}.pth'.format(epoch + 1))

    print(args.seed, 'Training took:', time.time()-train_started, 'seconds')  

        

    '''
    # load flow
    if args.load_flow_model:
        modelname1 = self.args.working_dir + '/models/' + args.flow_model
        if args.flow_model.endswith('tar'): # load pwc net
            data = torch.load(modelname1)
            self.vonet.flowNet.load_state_dict(data)
            print('load pwc network...')
        else:
            self.load_model(self.vonet.flowNet, modelname1)

    mean = None 
    std = None

    self.pose_norm = [0.13,0.13,0.13,0.013,0.013,0.013] # hard code, use when save motionfile when testing

    # load pose
    if args.load_pose_model:
        modelname2 = self.args.working_dir + '/models/' + args.pose_model
        self.load_model(self.vonet.flowPoseNet, modelname2)

    # load the whole model
    if self.args.load_model:
        modelname = self.args.working_dir + '/models/' + self.args.model_name
        self.load_model(self.vonet, modelname)
        print('load tartanvo network...')

    self.LrDecrease = [int(self.args.train_step/2), 
                        int(self.args.train_step*3/4), 
                        int(self.args.train_step*7/8)]
    self.lr = self.args.lr
    self.lr_flow = self.args.lr_flow


    self.pretrain_lr = self.args.pretrain_lr_scale
    self.voflowOptimizer  = optim.Adam([{'params':self.vonet.flowPoseNet.flowPoseNet.parameters(), 'lr': self.lr}, 
                                        {'params':self.vonet.flowPoseNet.preEncoder.parameters(), 'lr': self.lr*self.pretrain_lr}], lr = self.lr)

    self.criterion = nn.L1Loss()

    if self.args.multi_gpu>1:
        self.vonet = nn.DataParallel(self.vonet)

    self.vonet.cuda()
    
    # Instantiate an object for MyWF.
    trainVOFlow = TrainVONet(args.working_dir, args, prefix = args.exp_prefix, plotterType = plottertype) # flownet + posenet 불러오기
    trainVOFlow.initialize() # parameter 초기화...?
        trainVOFlow.train() #line 223
        if (trainVOFlow.count >= args.train_step):
            break
    trainVOFlow.finalize()

    print("Done.")
    '''

