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
import torch.optim.lr_scheduler as lr_scheduler
from utils_training.optimize_VONet_with_adaptive_resolution import train_epoch, validate_epoch
from utils.image_transforms import ArrayToTensor, ToTensor, Compose, CropCenter, dataset_intrinsics, DownscaleFlow, plot_traj, visflow, plot_traj_3d, RandomCropandResize, RandomResizeCrop
from utils_training.utils_CNN import load_checkpoint, save_checkpoint, boolean_string
from utils_training.utils_load_model import load_model
from tensorboardX import SummaryWriter
import random
import os
from os import path as osp
from termcolor import colored
import pickle
import time # for testing

from PretrainedVONet import PretrainedVONet
from TartanVO import TartanVO

'''
class TrainVONet(TorchFlow.TorchFlow):
    def __init__(self, workingDir, args, prefix = "", suffix = "", plotterType = 'Visdom'):
        super(TrainVONet, self).__finit__(workingDir, prefix, suffix, disableStreamLogger = False, plotterType = plotterType)
        self.args = args    
        self.saveModelName = 'vonet'
        self.source_img_transforms = transforms.Compose([ArrayToTensor(get_float=False)])
        self.target_img_transforms = transforms.Compose([ArrayToTensor(get_float=False)])
        self.valid_transform = Compose([CropCenter((args.image_height, args.image_width)), DownscaleFlow(), ToTensor()]) #only when valid
        self.train_transform = Compose([RandomResizeCrop((args.image_height, args.image_width), max_scale=2.5, keep_center=args.random_crop_center, fix_ratio=args.fix_ratio), DownscaleFlow(), ToTensor()])
        self.flow_transform = transforms.Compose([ArrayToTensor()])       
        # import ipdb;ipdb.set_trace()
        
        self.vonet = PretrainedVONet(intrinsic=self.args.intrinsic_layer, 
                            flowNormFactor=1.0, down_scale=args.downscale_flow, 
                            fixflow=args.fix_flow, pretrain=args.pretrain_model_name,
                            use_gru=args.use_gru)
        
        #self.vonet = TartanVO(args.model_name)

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

        if not self.args.test: 
            if self.args.train_vo: # dataloader for end2end flow vo
                # import ipdb;ipdb.set_trace()
                self.trainDataloader = EndToEndMultiDatasets(self.args.data_file, self.args.train_data_type, self.args.train_data_balence, 
                                                args, self.args.batch_size, self.args.worker_num,  
                                                mean=mean, std=std)
                self.pretrain_lr = self.args.pretrain_lr_scale
                self.voflowOptimizer = optim.Adam([{'params':self.vonet.flowPoseNet.flowPoseNet.parameters(), 'lr': self.lr}, 
                                                    {'params':self.vonet.flowPoseNet.preEncoder.parameters(), 'lr': self.lr*self.pretrain_lr}], lr = self.lr)

            if self.args.train_flow: # dataloader for flow 
                self.trainFlowDataloader = FlowMultiDatasets(self.args.flow_file, self.args.flow_data_type, self.args.flow_data_balence,
                                                        self.args, self.args.batch_size, self.args.worker_num,
                                                        mean = mean, std = std)
                self.flowOptimizer = optim.Adam(self.vonet.flowNet.parameters(),lr = self.lr_flow)

            self.testDataloader = EndToEndMultiDatasets(self.args.val_file, self.args.test_data_type, '1',
                                                        self.args, self.args.batch_size, self.args.worker_num, 
                                                        mean=mean, std=std)
        else: 
            self.testDataloader = EndToEndMultiDatasets(self.args.val_file, self.args.test_data_type, '1',
                                                        self.args, self.args.batch_size, self.args.worker_num, 
                                                        mean=mean, std=std, shuffle= (not args.test_traj))

        train_dataset, val_dataset = Tartanair_TrainigDataset(root=args.training_data_dir,
                                          source_image_transform=self.source_img_transforms,
                                          target_image_transform=self.target_img_transforms,
                                          flow_transform=self.flow_transform,
                                          co_transform=None, valid_transform = self.valid_transform, train_transform = self.train_transform,
                                          focalx = 320.0, focaly = 320.0, centerx = 320.0, centery = 240.0)
        self.pretrain_lr = self.args.pretrain_lr_scale
        self.voflowOptimizer = optim.Adam([{'params':self.vonet.flowPoseNet.flowPoseNet.parameters(), 'lr': self.lr}, 
                                            {'params':self.vonet.flowPoseNet.preEncoder.parameters(), 'lr': self.lr*self.pretrain_lr}], lr = self.lr)

        self.criterion = nn.L1Loss()

        if self.args.multi_gpu>1:
            self.vonet = nn.DataParallel(self.vonet)

        self.vonet.cuda()

    def initialize(self):
        super(TrainVONet, self).initialize()

        self.AV['loss'].avgWidth = 100
        self.add_accumulated_value('flow', 100)
        self.add_accumulated_value('pose', 100)
        self.add_accumulated_value('vo_flow', 100)

        self.add_accumulated_value('test', 1)
        self.add_accumulated_value('t_flow', 1)
        self.add_accumulated_value('t_pose', 1)

        self.add_accumulated_value('t_trans', 1)
        self.add_accumulated_value('t_rot', 1)
        self.add_accumulated_value('trans', 100)
        self.add_accumulated_value('rot', 100)
        self.append_plotter("loss", ['loss', 'test'], [True, False])
        self.append_plotter("loss_flow", ['flow', 'vo_flow', 't_flow'], [True, True, False])
        self.append_plotter("loss_pose", ['pose', 't_pose'], [True, False])
        self.append_plotter("trans_rot", ['trans', 'rot', 't_trans', 't_rot'], [True, True, False, False])

        logstr = ''
        for param in self.args.__dict__.keys(): # record useful params in logfile 
            logstr += param + ': '+ str(self.args.__dict__[param]) + ', '
        self.logger.info(logstr) 

        self.count = 0
        self.test_count = 0
        self.epoch = 0

        super(TrainVONet, self).post_initialize()

    def dumpfiles(self):
        self.save_model(self.vonet, self.saveModelName+'_'+str(self.count))
        self.write_accumulated_values()
        self.draw_accumulated_values()

    def forward_flow(self, sample, use_mask=False): 
        # if self.args.combine_lr: # Not compatible w/ PWC yet!
        #     rgbs = sample['rgbs']
        #     output = self.vonet.flowNet(rgbs.cuda(), True)
        # else:
        img1Tensor = sample['img0'].cuda()
        img2Tensor = sample['img0n'].cuda()
        output = self.vonet([img1Tensor,img2Tensor], only_flow=True)
        targetflow = sample['flow'].cuda()

        # import ipdb;ipdb.set_trace()
        if not use_mask:
            mask = None
        else:
            mask = sample['fmask'].cuda()
        if self.args.multi_gpu>1:
            loss = self.vonet.module.get_flow_loss(output, targetflow, self.criterion, mask=mask)
        else:
            loss = self.vonet.get_flow_loss(output, targetflow, self.criterion, mask=mask) #flow_loss(output, targetflow, use_mask, mask)
        return loss/self.args.normalize_output, output

    def forward_vo(self, sample, use_mask=False):
        #use_gtflow = random.random()<self.args.vo_gt_flow # use gt flow as the input of the posenet
        use_gttflow = False
        # load the variables
        if use_gtflow and self.args.fix_flow: # flownet is not trained, neither forward nor backward
            img0, img1 = None, None
            compute_flowloss = False
        else: 
            img0   = sample['img0'].cuda()
            img1   = sample['img0n'].cuda()
            compute_flowloss = True

        if self.args.intrinsic_layer:
            intrinsic = sample['intrinsic'].cuda()
        else: 
            intrinsic = None

        flow, mask = None, None
        if 'flow' in sample:
            flow = sample['flow'].cuda()
            if use_mask:
                mask = sample['fmask'].cuda()
        elif 'flow2' in sample:
            flow = sample['flow2'].cuda()
            if use_mask:
                mask = sample['fmask2'].cuda()

        if use_gtflow: # the gt flow will be input to the posenet
            # import ipdb;ipdb.set_trace()
            flow_output, pose_output = self.vonet([img0, img1, intrinsic, flow], only_pose=self.args.fix_flow, gt_flow=True)
        else: # use GT flow as the input
            flow_output, pose_output = self.vonet([img0, img1, intrinsic])
        pose_output_np = pose_output.data.cpu().detach().numpy().squeeze()

        if self.args.no_gt: 
            return 0., 0., 0., 0., pose_output_np

        # calculate flow loss
        if flow is not None and compute_flowloss:
            if self.args.multi_gpu>1:
                flowloss = self.vonet.module.get_flow_loss(flow_output, flow, self.criterion, mask=mask, small_scale=self.args.downscale_flow) /self.args.normalize_output
            else:
                flowloss = self.vonet.get_flow_loss(flow_output, flow, self.criterion, mask=mask, small_scale=self.args.downscale_flow) /self.args.normalize_output #flow_loss(flow_output, flow, use_mask, mask, small_scale=self.args.downscale_flow )/self.args.normalize_output
        else:
            flowloss = torch.FloatTensor([0])

        # calculate vo loss
        motion = sample['motion'].cuda()
        lossPose, trans_loss, rot_loss = self.linear_norm_trans_loss(pose_output, motion)

        return flowloss, lossPose, trans_loss, rot_loss, pose_output_np

    def linear_norm_trans_loss(self, output, motion, mask=None):
        output_trans = output[:, :3]
        output_rot = output[:, 3:]

        trans_norm = torch.norm(output_trans, dim=1).view(-1, 1)
        output_norm = output_trans/trans_norm

        if mask is None:
            trans_loss = self.criterion(output_norm, motion[:, :3])
            rot_loss = self.criterion(output_rot, motion[:, 3:])
        else:
            trans_loss = self.criterion(output_norm[mask,:], motion[mask, :3])
            rot_loss = self.criterion(output_rot[mask,:], motion[mask, 3:])

        loss = (rot_loss + trans_loss)/2.0

        return loss, trans_loss.item() , rot_loss.item()

    def train(self):
        super(TrainVONet, self).train()

        self.count = self.count + 1
        self.vonet.train()

        starttime = time.time()

        # train flow
        if self.args.train_flow: # not a vo only training
            flowsample, flowmask = self.trainFlowDataloader.load_sample()
            self.flowOptimizer.zero_grad()
            flowloss, _ = self.forward_flow(flowsample, use_mask=flowmask)
            flowloss.backward()
            self.flowOptimizer.step()
            self.AV['flow'].push_back(flowloss.item(), self.count)

        flowtime = time.time() 

        if self.args.train_vo: # not a flow only training
            self.voflowOptimizer.zero_grad()
            sample, vo_flowmask = self.trainDataloader.load_sample()
            loadtime = time.time()
            flowloss, poseloss, trans_loss, rot_loss, _ = self.forward_vo(sample, use_mask=vo_flowmask)
            if self.args.fix_flow:
                loss = poseloss
            else:
                loss = flowloss * self.args.lambda_flow + poseloss  # 
            loss.backward()
            self.voflowOptimizer.step()

            # import ipdb;ipdb.set_trace()
            self.AV['loss'].push_back(loss.item(), self.count)
            self.AV['vo_flow'].push_back(flowloss.item(), self.count)
            self.AV['pose'].push_back(poseloss.item(), self.count)
            self.AV['trans'].push_back(trans_loss, self.count)
            self.AV['rot'].push_back(rot_loss, self.count)

        nntime = time.time()

        # update Learning Rate
        if self.args.lr_decay:
            if self.count in self.LrDecrease:
                self.lr = self.lr*0.4
                self.lr_flow = self.lr_flow*0.4
                if self.args.train_vo:
                    assert len(self.voflowOptimizer.param_groups)==2
                    self.voflowOptimizer.param_groups[0]['lr'] = self.lr
                    self.voflowOptimizer.param_groups[1]['lr'] = self.lr * self.pretrain_lr
                if self.args.train_flow:
                    for param_group in self.flowOptimizer.param_groups: # ed_optimizer is defined in derived class
                        param_group['lr'] = self.lr_flow

        if self.count % self.args.print_interval == 0:
            losslogstr = self.get_log_str()
            self.logger.info("%s #%d - %s lr: %.6f - time (%.2f, %.2f)"  % (self.args.exp_prefix[:-1], 
                self.count, losslogstr, self.lr, flowtime-starttime, nntime-flowtime))

        if self.count % self.args.plot_interval == 0: 
            self.plot_accumulated_values()

        if self.count % self.args.test_interval == 0:
            if not (self.count)%self.args.snapshot==0:
                self.test()

        if (self.count)%self.args.snapshot==0:
            self.dumpfiles()
            # for k in range(self.args.test_num):
            #     self.test(save_img=True, save_surfix='test_'+str(k))

    def test(self):
        super(TrainVONet, self).test()
        self.test_count += 1

        self.vonet.eval()
        sample, mask = self.testDataloader.load_sample()

        with torch.no_grad():
            flowloss, poseloss, trans_loss, rot_loss, motion = self.forward_vo(sample, use_mask=mask)

        motion_unnorm = motion * self.pose_norm
        finish = self.test_count>= self.testDataloader.datalens[0]

        if self.args.no_gt:
            if self.test_count % self.args.print_interval == 0:
                self.logger.info("  TEST %s #%d - output : %s"  % (self.args.exp_prefix[:-1], 
                    self.test_count, motion_unnorm))
            return 0, 0, 0, 0, 0, motion_unnorm, finish

        if self.args.fix_flow:
            loss = poseloss
        else:
            loss = flowloss * self.args.lambda_flow  + poseloss # 

        lossnum = loss.item()
        self.AV['test'].push_back(lossnum, self.count)
        self.AV['t_flow'].push_back(flowloss.item(), self.count)
        self.AV['t_pose'].push_back(poseloss.item(), self.count)
        self.AV['t_trans'].push_back(trans_loss, self.count)
        self.AV['t_rot'].push_back(rot_loss, self.count)

        self.logger.info("  TEST %s #%d - (loss, flow, pose, rot, trans) %.4f  %.4f  %.4f  %.4f  %.4f"  % (self.args.exp_prefix[:-1], 
            self.test_count, loss.item(), flowloss.item(), poseloss.item(), rot_loss, trans_loss))

        return lossnum, flowloss.item(), poseloss.item(), trans_loss, rot_loss, motion_unnorm, finish

    def finalize(self):
        super(TrainVONet, self).finalize()
        if self.count < self.args.train_step and not self.args.test and not self.args.test_traj:
            self.dumpfiles()

        if self.args.test and not self.args.no_gt:
            self.logger.info('The average loss values: (t-trans, t-rot, t-flow, t-pose)')
            self.logger.info('%.4f \t %.4f \t %.4f \t %.4f \t %.4f' % (self.AV['loss'].last_avg(100), 
                self.AV['t_trans'].last_avg(100),
                self.AV['t_rot'].last_avg(100),
                self.AV['t_flow'].last_avg(100),
                self.AV['t_pose'].last_avg(100)))

        else:
            self.logger.info('The average loss values: (trans, rot, test, t_trans, t_rot)')
            self.logger.info('%.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f' % (self.AV['loss'].last_avg(100), 
                self.AV['trans'].last_avg(100),
                self.AV['rot'].last_avg(100),
                self.AV['test'].last_avg(100),
                self.AV['t_trans'].last_avg(100),
                self.AV['t_rot'].last_avg(100)))
'''
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
    # Optimization parameters
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
    parser.add_argument('--resume-e2e',  action='store_true',  default=False)
    parser.add_argument('--worker-num', type=int, default=1,
                        help='data loader worker number (default: 1)')
    parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 3e-4)')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_img_transforms = transforms.Compose([ArrayToTensor(get_float=False)])
    target_img_transforms = transforms.Compose([ArrayToTensor(get_float=False)])
    valid_transform = Compose([CropCenter((args.image_height, args.image_width)), DownscaleFlow(), ToTensor()]) #only when valid
    train_transform = Compose([RandomResizeCrop((args.image_height, args.image_width), max_scale=2.5, keep_center=args.random_crop_center, fix_ratio=args.fix_ratio), DownscaleFlow(), ToTensor()])
    flow_transform = transforms.Compose([ArrayToTensor()])

    # training and validation dataset
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
    
    #Dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                        shuffle=True, num_workers=args.worker_num)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, 
                                    shuffle=True, num_workers=args.worker_num)
    '''
    self.vonet = PretrainedVONet(intrinsic=self.args.intrinsic_layer, 
                        flowNormFactor=1.0, down_scale=args.downscale_flow, 
                        fixflow=args.fix_flow, pretrain=args.pretrain_model_name,
                        use_gru=args.use_gru)
    '''
    # models
    vonet = TartanVO()
    print(colored('==> ', 'blue') + 'TartanVO created.')

    optimizer = \
        optim.Adam(filter(lambda p: p.requires_grad, vonet.parameters()),
                   lr=args.lr)
                   #weight_decay=args.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer,
                                         #milestones=[65, 75, 95],
                                         milestones=[int(args.n_epoch/2),
                                                    int(args.n_epoch*7/8)],#e2e 25
                                         gamma=0.2)#poselr

    # load the whole model
    if args.pretrained_model is not None:
        modelname = args.pretrained_model
        vonet = load_model(vonet, modelname)
    
    if not osp.isdir(args.snapshots):
        os.mkdir(args.snapshots)

    cur_snapshot = args.name_exp
    if not osp.isdir(osp.join(args.snapshots, cur_snapshot)):
        os.makedirs(osp.join(args.snapshots, cur_snapshot))

    with open(osp.join(args.snapshots, cur_snapshot, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)

    best_val = float("inf")
    start_epoch = 0

    save_path = osp.join(args.snapshots, cur_snapshot)
    train_writer = SummaryWriter(os.path.join(save_path, 'train'))
    test_writer = SummaryWriter(os.path.join(save_path, 'test'))

    vonet = nn.DataParallel(vonet)
    vonet = vonet.to(device)
    train_started = time.time()

    for epoch in range(start_epoch, args.n_epoch):
        scheduler.step()
        print('starting epoch {}:  learning rate is {}'.format(epoch, scheduler.get_last_lr()[0]))
        results_dir = os.path.join(save_path, 'result')
        if not osp.isdir(results_dir):
            os.mkdir(results_dir)    

        # Validation
        valid_motionlist = np.array([])
        valid_loss_pose, motionlist = \
        validate_epoch(vonet, valid_motionlist, val_dataloader, device, epoch=epoch, save_path=os.path.join(save_path, 'test'), div_flow=args.div_flow,
                           apply_mask=False) #수정)
        #test_writer.add_scalar('train loss flow', valid_loss_flow, epoch)
        test_writer.add_scalar('train loss pose', valid_loss_pose, epoch)
        #print(colored('==> ', 'blue') + 'Val average flow loss :', valid_loss_flow)
        print(colored('==> ', 'blue') + 'Val average pose loss:', valid_loss_pose)

        print(colored('==> ', 'green') + 'finished epoch :', epoch + 1)

        is_best = valid_loss_pose < best_val
        best_val = min(valid_loss_pose, best_val)

        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': vonet.module.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict(),
                         'best_loss': best_val},
                        is_best, save_path, 'epoch_{}.pth'.format(epoch + 1))
        


        
        train_loss_flow, train_loss_pose, train_last_batch_ate = train_epoch(vonet, 
                                 optimizer,
                                 train_dataloader,
                                 device,
                                 epoch,
                                 train_writer,
                                 div_flow=args.div_flow,
                                 save_path=os.path.join(save_path, 'train'),
                                 apply_mask=False, results_dir=results_dir)#수정
        train_writer.add_scalar('train loss flow', train_loss_flow, epoch)
        train_writer.add_scalar('train loss pose', train_loss_pose, epoch)
        train_writer.add_scalar('train last batch ate', train_last_batch_ate, epoch)
        train_writer.add_scalar('learning_rate', scheduler.get_lr()[0], epoch)
        print(colored('==> ', 'green') + 'Train average flow loss:', train_loss_flow)  
        print(colored('==> ', 'green') + 'Train average pose loss:', train_loss_pose)



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

