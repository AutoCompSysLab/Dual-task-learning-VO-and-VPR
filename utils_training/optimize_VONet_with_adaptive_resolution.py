import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from utils.pixel_wise_mapping import remap_using_flow_fields
from utils.losses import get_loss, get_miner
from matplotlib import pyplot as plt
from evaluator.evaluator_base import quats2SEs
from evaluator.trajectory_transform import trajectory_transform, rescale
from evaluator.transformation import pos_quats2SE_matrices, SE2pos_quat
from Datasets.transformation import ses2poses_quat
from Datasets.utils import plot_traj_3d, plot_traj, visflow
from core.loss import sequence_loss
import torch.nn as nn
import pdb

def transform_trajs(gt_traj, est_traj, cal_scale):
    gt_traj, est_traj = trajectory_transform(gt_traj, est_traj)
    if cal_scale :
        est_traj, s = rescale(gt_traj, est_traj)
        #print('  Scale, {}'.format(s))
    else:
        s = 1.0
    return gt_traj, est_traj, s

from evaluator.evaluate_ate_scale import align

class ATEEvaluator(object):
    def __init__(self):
        super(ATEEvaluator, self).__init__()
    def evaluate(self, gt_traj, est_traj, scale):
        gt_xyz = np.matrix(gt_traj[:,0:3].transpose())
        est_xyz = np.matrix(est_traj[:, 0:3].transpose())

        rot, trans, trans_error, s = align(gt_xyz, est_xyz, scale)
        #print('  ATE scale: {}'.format(s))
        error = np.sqrt(np.dot(trans_error,trans_error) / len(trans_error))

        # align two trajs 
        est_SEs = pos_quats2SE_matrices(est_traj)
        T = np.eye(4) 
        T[:3,:3] = rot
        T[:3,3:] = trans 
        T = np.linalg.inv(T)
        est_traj_aligned = []
        for se in est_SEs:
            se[:3,3] = se[:3,3] * s
            se_new = T.dot(se)
            se_new = SE2pos_quat(se_new)
            est_traj_aligned.append(se_new)

        est_traj_aligned = np.array(est_traj_aligned)
        return error, gt_traj, est_traj_aligned

def pre_process_data(source_img, target_img, device):
    '''
    Pre-processes source and target images before passing it to the network
    :param source_img: Torch tensor Bx3xHxW
    :param target_img: Torch tensor Bx3xHxW
    :param device: cpu or gpu
    :return:
    source_img_copy: Torch tensor Bx3xHxW, source image scaled to 0-1 and mean-centered and normalized
                     using mean and standard deviation of ImageNet
    target_img_copy: Torch tensor Bx3xHxW, target image scaled to 0-1 and mean-centered and normalized
                     using mean and standard deviation of ImageNet
    source_img_256: Torch tensor Bx3x256x256, source image rescaled to 256x256, scaled to 0-1 and mean-centered and normalized
                    using mean and standard deviation of ImageNet
    target_img_256: Torch tensor Bx3x256x256, target image rescaled to 256x256, scaled to 0-1 and mean-centered and normalized
                    using mean and standard deviation of ImageNet
    '''
    # img has shape bx3xhxw
    b, _, h_scale, w_scale = target_img.shape
    mean_vector = np.array([0.485, 0.456, 0.406])
    std_vector = np.array([0.229, 0.224, 0.225])
    
    # original resolution
    source_img_copy = source_img.float().to(device).div(255.0)
    target_img_copy = target_img.float().to(device).div(255.0)
    
    mean = torch.as_tensor(mean_vector, dtype=source_img_copy.dtype, device=source_img_copy.device)
    std = torch.as_tensor(std_vector, dtype=source_img_copy.dtype, device=source_img_copy.device)
    '''
    source_img_copy.sub_(mean[:, None, None]).div_(std[:, None, None])
    target_img_copy.sub_(mean[:, None, None]).div_(std[:, None, None])
    '''
    # resolution 256x256
    source_img_640 = torch.nn.functional.interpolate(input=source_img.float().to(device),
                                                      size=(640, 640),
                                                      mode='area').byte()
    target_img_640 = torch.nn.functional.interpolate(input=target_img.float().to(device),
                                                      size=(640, 640),
                                                      mode='area').byte()

    source_img_640 = source_img_640.float().div(255.0)
    target_img_640 = target_img_640.float().div(255.0)
    source_img_640.sub_(mean[:, None, None]).div_(std[:, None, None])
    target_img_640.sub_(mean[:, None, None]).div_(std[:, None, None])

    return source_img_640, target_img_640


def plot_during_training(save_path, epoch, batch, apply_mask,
                         h_original, w_original, h_256, w_256,
                         source_image, target_image, source_image_256, target_image_256, div_flow,
                         flow_gt_original, flow_gt_256, output_net,  output_net_256, mask=None, mask_256=None):
    # resolution original
    flow_est_original = F.interpolate(output_net, (h_original, w_original),
                                      mode='bilinear', align_corners=False)  # shape Bx2xHxW
    flow_target_x = div_flow * flow_gt_original.detach().permute(0, 2, 3, 1)[0, :, :, 0]  # shape HxW
    flow_target_y = div_flow * flow_gt_original.detach().permute(0, 2, 3, 1)[0, :, :, 1]
    flow_est_x = div_flow * flow_est_original.detach().permute(0, 2, 3, 1)[0, :, :, 0]  # shape BxHxW
    flow_est_y = div_flow * flow_est_original.detach().permute(0, 2, 3, 1)[0, :, :, 1]
    assert flow_est_x.shape == flow_target_x.shape

    mean_values = torch.tensor([0.485, 0.456, 0.406],
                               dtype=source_image.dtype).view(3, 1, 1)
    std_values = torch.tensor([0.229, 0.224, 0.225],
                              dtype=source_image.dtype).view(3, 1, 1)
    image_1 = (source_image.detach()[0].cpu() * std_values +
               mean_values).clamp(0, 1).permute(1, 2, 0)
    image_2 = (target_image.detach()[0].cpu() * std_values +
               mean_values).clamp(0, 1).permute(1, 2, 0)
    remapped_gt = remap_using_flow_fields(image_1.numpy(),
                                          flow_target_x.cpu().numpy(),
                                          flow_target_y.cpu().numpy())
    remapped_est = remap_using_flow_fields(image_1.numpy(), flow_est_x.cpu().numpy(),
                                           flow_est_y.cpu().numpy())

    # resolution 256x256
    flow_est_256 = F.interpolate(output_net_256, (h_256, w_256),
                                 mode='bilinear', align_corners=False)  # shape Bx2xHxW
    # for batch 0
    flow_target_x_256 = div_flow * flow_gt_256.detach().permute(0, 2, 3, 1)[0, :, :, 0]  # shape HxW
    flow_target_y_256 = div_flow * flow_gt_256.detach().permute(0, 2, 3, 1)[0, :, :, 1]
    flow_est_x_256 = div_flow * flow_est_256.detach().permute(0, 2, 3, 1)[0, :, :, 0]  # shape BxHxW
    flow_est_y_256 = div_flow * flow_est_256.detach().permute(0, 2, 3, 1)[0, :, :, 1]
    assert flow_est_x_256.shape == flow_target_x_256.shape

    image_1_256 = (source_image_256.detach()[0].cpu() * std_values +
                   mean_values).clamp(0, 1).permute(1, 2, 0)
    image_2_256 = (target_image_256.detach()[0].cpu() * std_values +
                   mean_values).clamp(0, 1).permute(1, 2, 0)
    remapped_gt_256 = remap_using_flow_fields(image_1_256.numpy(),
                                              flow_target_x_256.cpu().numpy(),
                                              flow_target_y_256.cpu().numpy())
    remapped_est_256 = remap_using_flow_fields(image_1_256.numpy(), flow_est_x_256.cpu().numpy(),
                                               flow_est_y_256.cpu().numpy())

    fig, axis = plt.subplots(2, 5, figsize=(20, 20))
    axis[0][0].imshow(image_1.numpy())
    axis[0][0].set_title("original reso: \nsource image")
    axis[0][1].imshow(image_2.numpy())
    axis[0][1].set_title("original reso: \ntarget image")
    if apply_mask:
        mask = mask.detach()[0].cpu().numpy().astype(np.float32)
    else:
        mask = np.ones((h_original, w_original))
    axis[0][2].imshow(mask, vmin=0.0, vmax=1.0)
    axis[0][2].set_title("original reso: \nmask applied during training")
    axis[0][3].imshow(remapped_gt)
    axis[0][3].set_title("original reso : \nsource remapped with ground truth")
    axis[0][4].imshow(remapped_est)
    axis[0][4].set_title("original reso: \nsource remapped with network")
    axis[1][0].imshow(image_1_256.numpy())
    axis[1][0].set_title("reso 256: \nsource image")
    axis[1][1].imshow(image_2_256.numpy())
    axis[1][1].set_title("reso 256:\ntarget image")
    if apply_mask:
        mask_256 = mask_256.detach()[0].cpu().numpy().astype(np.float32)
    else:
        mask_256 = np.ones((h_256, w_256))
    axis[1][2].imshow(mask_256, vmin=0.0, vmax=1.0)
    axis[1][2].set_title("reso 256: \nmask applied during training")
    axis[1][3].imshow(remapped_gt_256)
    axis[1][3].set_title("reso 256: \nsource remapped with ground truth")
    axis[1][4].imshow(remapped_est_256)
    axis[1][4].set_title("reso 256: \nsource remapped with network")
    fig.savefig('{}/epoch{}_batch{}.png'.format(save_path, epoch, batch),
                bbox_inches='tight')
    plt.close(fig)

def linear_norm_trans_loss(output, motion):
        output_trans = output[:, :3]
        output_rot = output[:, 3:]

        trans_norm = torch.linalg.norm(output_trans, dim=1).view(-1, 1)
        output_norm = output_trans/trans_norm

        motion_trans_norm = torch.linalg.norm(motion[:, :3], dim=1).view(-1, 1)
        motion_norm = motion[:, :3] / motion_trans_norm

        trans_loss = F.l1_loss(output_norm, motion_norm) 
        rot_loss = F.l1_loss(output_rot, motion[:, 3:])

        loss = (rot_loss + trans_loss)/2.0#2 -> 200 for AMP
        return loss, trans_loss.item() , rot_loss.item()

def COMPASS_linear_norm_trans_loss(output, motion, mask=None):
    criterion = torch.nn.L1Loss()

    output_trans = output[:, :3]
    output_rot = output[:, 3:]

    trans_norm = torch.linalg.norm(output_trans, dim=1).view(-1, 1)
    output_norm = output_trans/trans_norm

    if mask is None:
        trans_loss = criterion(output_norm, motion[:, :3])
        rot_loss = criterion(output_rot, motion[:, 3:])
    else:
        trans_loss = criterion(output_norm[mask,:], motion[mask, :3])
        rot_loss = criterion(output_rot[mask,:], motion[mask, 3:])

    loss = (rot_loss + trans_loss)/2.0
    return loss, trans_loss.item() , rot_loss.item()


def train_epoch(flownet, 
                posenet,
                optimizer,
                train_loader,
                VPR_train_loader,
                device,
                epoch,
                train_writer,
                cfg,
                div_flow=1.0,
                save_path=None,
                loss_grid_weights=None,
                apply_mask=False,
                robust_L1_loss=False,
                sparse=False,
                results_dir=False):
    """
    Training epoch script
    Args:
        net: model architecture
        optimizer: optimizer to be used for traninig `net`
        train_loader: dataloader
        device: `cpu` or `gpu`
        epoch: epoch number for plotting
        train_writer: for tensorboard
        div_flow: multiplicative factor to apply to the estimated flow
        save_path: path to folder to save the plots
        loss_grid_weights: weight coefficients for each level of the feature pyramid
        apply_mask: bool on whether or not to apply a mask for the loss
        robust_L1_loss: bool on the loss to use
        sparse: bool on sparsity of ground truth flow field
    Output:
        running_total_loss: total training loss

        here output of the network at every level is flow interpolated but not scaled.
        we only use the ground truth flow as highest resolution and downsample it without scaling.
    """
    n_iter = epoch*len(train_loader)
    flownet.train()
    posenet.train()
    running_total_loss_flow = 0
    running_total_loss_trans = 0
    running_total_loss_rot = 0
    running_total_loss_pose = 0
    running_total_loss_vpr = 0
    running_total_batch_acc_vpr = 0

    criterion = nn.L1Loss()
    running_total_loss = 0

    #pbar = tqdm(enumerate(zip(train_loader, VPR_train_loader)), total=len(train_loader))
    pbar_train = tqdm(train_loader)
    pbar = enumerate(zip(pbar_train, VPR_train_loader))
    #pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, mini_batch in pbar:
        optimizer.zero_grad()
        
        source_image, target_image = pre_process_data(mini_batch[0]['source_image'],
                                                      mini_batch[0]['target_image'],
                                                      device=device)
        #intrinsic, flow, mask = None, None, None
        
        intrinsic = mini_batch['intrinsic'].float().to(device)
        
        flow = mini_batch['flow'].float().to(device)
        bs, _, h_original, w_original = flow.shape
        flow = F.interpolate(flow, (640, 640),
                                    mode='bilinear', align_corners=False)
        flow[:, 0, :, :] *= 640.0 / float(w_original)
        flow[:, 1, :, :] *= 640.0 / float(h_original)
        intrinsic = F.interpolate(intrinsic, (640, 640),
                                    mode='bilinear', align_corners=False)
        intrinsic[:, 0, :, :] *= 640.0 / float(w_original)
        intrinsic[:, 1, :, :] *= 640.0 / float(h_original)
        
        places = mini_batch[1][0]
        labels = mini_batch[1][1]
        BS, N, ch, h, w = places.shape
        images = places.view(BS*N, ch, h, w)
        labels = labels.view(-1)
        
        flow_output = flownet(stage='vo', image1=source_image, image2=target_image)

        flow_scale = 20.0
        flow_input = flow_output[-1].clone()/flow_scale
        pose_output = posenet(stage='vo', x=torch.cat((flow_input, intrinsic),1))
        features = flownet(stage='vpr', vpr_img=images)
        descriptors = posenet(stage='vpr', vpr_feature=features)

        pose_output_np = pose_output.data.cpu().detach().numpy().squeeze()
        # calculate flow loss
        
        valid = None
        
        if flow is not None:
            #flowloss = net.module.vonet.get_flow_loss(flow_output, flow, criterion, mask=mask, training = True, small_scale=True)
            flowloss, metrics = sequence_loss(flow_output, flow, valid, cfg)
        else:
            flowloss = torch.FloatTensor([0])
        
        
        #flow_scale = 140.0

        motion_gt = mini_batch['motion'].float()
    
        pose_std = np.array([ 0.13,  0.13,  0.13, 0.013, 0.013,  0.013], dtype=np.float32)
        motion_gt = motion_gt / pose_std

    
        motion_gt = motion_gt.to(device)
        poseloss, trans_loss, rot_loss = linear_norm_trans_loss(pose_output, motion_gt)
        
        vpr_loss_func = get_loss('MultiSimilarityLoss')
        vpr_miner = get_miner('MultiSimilarityMiner', 0.1)
        
        miner_outputs = vpr_miner(descriptors, labels)
        vpr_loss = vpr_loss_func(descriptors, labels, miner_outputs)

        nb_samples = descriptors.shape[0]
        nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
        batch_acc = 1.0 - (nb_mined/nb_samples)
        
        Loss = flowloss/flow_scale + poseloss + vpr_loss
        Loss.backward()
        optimizer.step()
        n_iter += 1
        running_total_loss_pose += poseloss.item()
        running_total_loss_flow += flowloss.item()
        running_total_loss_vpr += vpr_loss.item()
        running_total_batch_acc_vpr += batch_acc
        
        pbar_train.set_description(
            'train R_total_loss: %.3f/%.3f/%.3f/%.3f/%.3f/%.3f/%.3f/%.3f' % (running_total_loss_flow / (i + 1), flowloss.item(),
                                                            running_total_loss_pose / (i + 1), poseloss.item(),
                                                            running_total_loss_vpr / (i + 1), vpr_loss.item(),
                                                            running_total_batch_acc_vpr / (i+1), batch_acc))
        
    running_total_loss_flow /= len(train_loader)
    running_total_loss_pose /= len(train_loader)
    running_total_loss_vpr /= len(train_loader)
    running_total_batch_acc_vpr /= len(train_loader)    
    
    ##only for last batch
    motionnp = pose_output.clone().cpu().detach().numpy()
    pose_std = np.array([ 0.13,  0.13,  0.13, 0.013, 0.013,  0.013], dtype=np.float32) 
    motionnp = motionnp * pose_std
    if 'motion' in mini_batch[0]:
        motions_gt = mini_batch[0]['motion']
        scale = np.linalg.norm(motions_gt[:,:3], axis=1)
        trans_est = motionnp[:,:3]
        trans_est = trans_est/np.linalg.norm(trans_est,axis=1).reshape(-1,1)*scale.reshape(-1,1)
        motionnp[:,:3] = trans_est 
    else:
        print('    scale is not given, using 1 as the default scale value..')
    est_traj = ses2poses_quat(np.array(motionnp))
    gt_traj = ses2poses_quat(np.array(motions_gt.numpy()))
    gt_traj_trans, est_traj_trans, s = transform_trajs(gt_traj, est_traj, True)
    gt_SEs, est_SEs = quats2SEs(gt_traj_trans, est_traj_trans)
    ate_eval = ATEEvaluator()
    ate_score, gt_ate_aligned, est_ate_aligned = ate_eval.evaluate(gt_traj, est_traj, True)
    plot_traj_3d(gt_ate_aligned, est_ate_aligned, vis=False, savefigname=results_dir+'/train_traj_epoch_{}'.format(epoch)+'.png', title='ATE %.4f' %(ate_score))

    return running_total_loss_flow, running_total_loss_pose, running_total_loss_vpr, ate_score


def validate_epoch(flownet,
                   posenet,
                   motionlist,
                   motionlist_gt,
                   val_loader,
                   device,
                   epoch,
                   save_path,
                   div_flow=1,
                   loss_grid_weights=None,
                   apply_mask=False,
                   sparse=False,
                   robust_L1_loss=False,
                   results_dir=False):
    """
    Validation epoch script
    Args:
        net: model architecture
        val_loader: dataloader
        device: `cpu` or `gpu`
        epoch: epoch number for plotting
        train_writer: for tensorboard
        div_flow: multiplicative factor to apply to the estimated flow
        save_path: path to folder to save the plots
        loss_grid_weights: weight coefficients for each level of the feature pyramid
        apply_mask: bool on whether or not to apply a mask for the loss
        robust_L1_loss: bool on the loss to use
        sparse: bool on sparsity of ground truth flow field
    Output:
        running_total_loss: total validation loss,
        EPE_0, EPE_1, EPE_2, EPE_3: EPEs corresponding to each level of the network (after upsampling
        the estimated flow to original resolution and scaling it properly to compare to ground truth).

        here output of the network at every level is flow interpolated but not scaled.
        we only use the ground truth flow as highest resolution and downsample it without scaling.

    """

    flownet.eval()
    posenet.eval()
    #if loss_grid_weights is None:
        #loss_grid_weights = [0.32, 0.08, 0.02, 0.01, 0.005]

    #running_total_loss_flow = 0
    #running_total_loss_trans = 0
    #running_total_loss_rot = 0
    running_total_loss_pose = 0

    criterion = nn.L1Loss()
    ate_scorelist=[]
    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader), total=len(val_loader))
        #EPE_array = torch.zeros([len(loss_grid_weights), len(val_loader)], dtype=torch.float32, device=device)
        for i, mini_batch in pbar:
            source_image, target_image = pre_process_data(
                mini_batch['source_image'],
                mini_batch['target_image'],
                device=device)
            #intrinsic, flow, mask = None, None, None
            #intrinsic = mini_batch['intrinsic'].float().to(device)
            #import pdb; pdb.set_trace()
            flow = mini_batch['flow'].float().to(device)
            flow_output = flownet(source_image, target_image)
            #import pdb; pdb.set_trace()
            flow_scale = 20.0
            flow_input = flow_output[0].clone()/flow_scale
            #import pdb; pdb.set_trace()
            pose_output = posenet(torch.cat((flow_input, mini_batch['intrinsic'].to(device)),1))
            #flow_output, pose_output = net([source_image, target_image, intrinsic])
            #flow_output, pose_output = net([target_image, source_image, intrinsic])
            #motionnp = pose_output.cpu().numpy()
            #pose_std = np.array([ 0.13,  0.13,  0.13, 0.013, 0.013,  0.013], dtype=np.float32)
            '''
            motionnp = motionnp * pose_std

            if 'motion' in mini_batch:
                motions_gt = mini_batch['motion']
                scale = np.linalg.norm(motions_gt[:,:3], axis=1)
                trans_est = motionnp[:,:3]
                trans_est = trans_est/np.linalg.norm(trans_est,axis=1).reshape(-1,1)*scale.reshape(-1,1)
                motionnp[:,:3] = trans_est
            else:
                print('scale is not given, using 1 as the default scale value..')

            # calculate flow loss
            if flow is not None:
                flowloss = net.module.vonet.get_flow_loss(flow_output, flow, criterion, mask=mask, training = True, small_scale=False) / 0.05
            else:
                flowloss = torch.FloatTensor([0])
            '''
            
            motion = mini_batch['motion'].float()
            pose_std = np.array([ 0.13,  0.13,  0.13, 0.013, 0.013,  0.013], dtype=np.float32)
            motion_gt = motion / pose_std
            trans = motion_gt[:,:3]
            trans_norm = np.linalg.norm(trans, axis=1)
            motion_gt[:,:3] = motion_gt[:,:3]/trans_norm.reshape(-1,1)
            motion_gt = motion_gt.to(device)

            #motionnp = motionnp / pose_std
            #poseloss, trans_loss, rot_loss = linear_norm_trans_loss(torch.from_numpy(motionnp), motions_gt)
            poseloss, trans_loss, rot_loss = COMPASS_linear_norm_trans_loss(pose_output, motion_gt)
            '''
            if len(motionlist) == 0:
                motionlist = motionnp
            else:
                motionlist = np.append(motionlist, motionnp, axis = 0)
            '''

            running_total_loss_pose += poseloss.item()
            #running_total_loss_flow += flowloss.item()

            motionnp = pose_output.clone().cpu().detach().numpy()
            pose_std = np.array([ 0.13,  0.13,  0.13, 0.013, 0.013,  0.013], dtype=np.float32) 
            motionnp = motionnp * pose_std
            if 'motion' in mini_batch:
                motions_gt = mini_batch['motion']
                scale = np.linalg.norm(motions_gt[:,:3], axis=1)
                trans_est = motionnp[:,:3]
                trans_est = trans_est/np.linalg.norm(trans_est,axis=1).reshape(-1,1)*scale.reshape(-1,1)
                motionnp[:,:3] = trans_est 
            else:
                print('    scale is not given, using 1 as the default scale value..')
            if len(motionlist) == 0:
                motionlist = motionnp
            else:
                motionlist = np.append(motionlist, motionnp, axis = 0)

            motion_gt = motion_gt.clone().cpu().detach().numpy()
            if len(motionlist_gt) == 0:
                motionlist_gt = motions_gt
            else:
                motionlist_gt = np.append(motionlist_gt, motions_gt, axis = 0)
            

            est_traj = ses2poses_quat(np.array(motionnp))
            gt_traj = ses2poses_quat(np.array(motions_gt.numpy()))
            gt_traj_trans, est_traj_trans, s = transform_trajs(gt_traj, est_traj, True)
            gt_SEs, est_SEs = quats2SEs(gt_traj_trans, est_traj_trans)
            ate_eval = ATEEvaluator()
            ate_score, gt_ate_aligned, est_ate_aligned = ate_eval.evaluate(gt_traj, est_traj, True)
            ate_scorelist.append(ate_score)
            plot_traj_3d(gt_ate_aligned, est_ate_aligned, vis=False, savefigname=results_dir+'/valid_traj_epoch_{}_{}'.format((epoch), str(i)+'.png'), title='ATE %.4f' %(ate_score))


            pbar.set_description(
                ' validation R_total_loss: %.3f/%.3f' % (running_total_loss_pose / (i + 1), poseloss.item()))

        #running_total_loss_flow /= len(val_loader)
        running_total_loss_pose /= len(val_loader)


        

        '''
        est_traj = ses2poses_quat(np.array(motionnp))
        gt_traj = ses2poses_quat(np.array(motions_gt.numpy()))
        gt_traj_trans, est_traj_trans, s = transform_trajs(gt_traj, est_traj, True)
        gt_SEs, est_SEs = quats2SEs(gt_traj_trans, est_traj_trans)
        ate_eval = ATEEvaluator()
        ate_score, gt_ate_aligned, est_ate_aligned = ate_eval.evaluate(gt_traj, est_traj, True)
        plot_traj_3d(gt_ate_aligned, est_ate_aligned, vis=False, savefigname=results_dir+'/valid_traj_epoch_{}'.format(epoch)+'.png', title='ATE %.4f' %(ate_score))
        '''

    return running_total_loss_pose, ate_scorelist, motionlist, motionlist_gt