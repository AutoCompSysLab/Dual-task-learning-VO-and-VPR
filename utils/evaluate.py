from tqdm import tqdm
import torch
import numpy as np
from utils.io import writeFlow
import torch.nn as nn
import torch.nn.functional as F


def epe(input_flow, target_flow, mean=True):
    """
    End-point-Error computation
    Args:
        input_flow: estimated flow [BxHxW,2]
        target_flow: ground-truth flow [BxHxW,2]
    Output:
        Averaged end-point-error (value)
    """
    EPE = torch.norm(target_flow - input_flow, p=2, dim=1)
    if mean:
        EPE = EPE.mean()
    return EPE


def correct_correspondences(input_flow, target_flow, alpha, img_size):
    """
    Computation PCK, i.e number of the pixels within a certain threshold
    Args:
        input_flow: estimated flow [BxHxW,2]
        target_flow: ground-truth flow [BxHxW,2]
        alpha: threshold
        img_size: image size
    Output:
        PCK metric
    """
    # input flow is shape (BxHgtxWgt,2)
    dist = torch.norm(target_flow - input_flow, p=2, dim=1)
    # dist is shape BxHgtxWgt
    pck_threshold = alpha * img_size
    mask = dist.le(pck_threshold) # Computes dist ≤ pck_threshold element-wise (element then equal to 1)
    return mask.sum().item()


def F1_kitti_2015(input_flow, target_flow, tau=[3.0, 0.05]):
    """
    Computation number of outliers
    for which error > 3px(tau[0]) and error/magnitude(ground truth flow) > 0.05(tau[1])
    Args:
        input_flow: estimated flow [BxHxW,2]
        target_flow: ground-truth flow [BxHxW,2]
        alpha: threshold
        img_size: image size
    Output:
        PCK metric
    """
    # input flow is shape (BxHgtxWgt,2)
    dist = torch.norm(target_flow - input_flow, p=2, dim=1)
    gt_magnitude = torch.norm(target_flow, p=2, dim=1)
    # dist is shape BxHgtxWgt
    mask = dist.gt(3.0) & (dist/gt_magnitude).gt(0.05)
    return mask.sum().item()


def calculate_epe_and_pck_per_dataset(test_dataloader, network, device, threshold_range, path_to_save=None,
                                      compute_F1=False, save=False):
    aepe_array = []
    pck_alpha_0_01_over_image = []
    pck_alpha_0_05_over_image = []
    pck_alpha_0_1_over_image = []
    pck_thresh_1_over_image = []
    pck_thresh_3_over_image = []
    pck_thresh_5_over_image = []
    F1 = 0.0

    n_registered_pxs = 0.0
    array_n_correct_correspondences = np.zeros(threshold_range.shape, dtype=np.float32)

    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    for i_batch, mini_batch in pbar:
        source_img = mini_batch['source_image']
        target_img = mini_batch['target_image']
        mask_gt = mini_batch['correspondence_mask'].to(device)
        flow_gt = mini_batch['flow_map'].to(device)
        if flow_gt.shape[1] != 2:
            # shape is BxHxWx2
            flow_gt = flow_gt.permute(0,3,1,2)
        bs, ch_g, h_g, w_g = flow_gt.shape

        flow_estimated = network.estimate_flow(source_img, target_img, device, mode='channel_first')

        # torch tensor of shape Bx2xH_xW_, will be the same types (cuda or cpu) depending on the device
        # H_ and W_ could be smaller than the ground truth flow (ex DCG Net takes only 240x240 images)
        if flow_estimated.shape[2] != h_g or flow_estimated.shape[3] != w_g:
            '''
            the estimated flow is downscaled (the original images were downscaled before 
            passing through the network)
            as it is the case with DCG Net, the estimate flow will have shape 240x240
            it needs to be upscaled to the same size as flow_target_x and rescaled accordingly:
            '''
            ratio_h = float(h_g) / float(flow_estimated.shape[2])
            ratio_w = float(w_g) / float(flow_estimated.shape[3])
            flow_estimated = nn.functional.interpolate(flow_estimated, size=(h_g, w_g), mode='bilinear',
                                                       align_corners=False)
            flow_estimated[:, 0, :, :] *= ratio_w
            flow_estimated[:, 1, :, :] *= ratio_h
        assert flow_estimated.shape == flow_gt.shape

        flow_target_x = flow_gt.permute(0, 2, 3, 1)[:, :, :, 0]
        flow_target_y = flow_gt.permute(0, 2, 3, 1)[:, :, :, 1]
        flow_est_x = flow_estimated.permute(0, 2, 3, 1)[:, :, :, 0]  # B x h_g x w_g
        flow_est_y = flow_estimated.permute(0, 2, 3, 1)[:, :, :, 1]

        flow_target = \
            torch.cat((flow_target_x[mask_gt].unsqueeze(1),
                       flow_target_y[mask_gt].unsqueeze(1)), dim=1)
        flow_est = \
            torch.cat((flow_est_x[mask_gt].unsqueeze(1),
                       flow_est_y[mask_gt].unsqueeze(1)), dim=1)
        # flow_target_x[mask_gt].shape is (number of pixels), then with unsqueze(1) it becomes (number_of_pixels, 1)
        # final shape is (B*H*W , 2), B*H*W is the number of registered pixels (according to ground truth masks)

        # let's calculate EPE per batch
        aepe = epe(flow_est, flow_target)  # you obtain the mean per pixel
        aepe_array.append(aepe.item())

        # let's calculate PCK values
        img_size = max(mini_batch['source_image_size'][0], mini_batch['source_image_size'][1]).float().to(device)
        alpha_0_01 = correct_correspondences(flow_est, flow_target, alpha=0.01, img_size=img_size)
        alpha_0_05 = correct_correspondences(flow_est, flow_target, alpha=0.05, img_size=img_size)
        alpha_0_1 = correct_correspondences(flow_est, flow_target, alpha=0.1, img_size=img_size)
        px_1 = correct_correspondences(flow_est, flow_target, alpha=1.0/float(img_size), img_size=img_size) # threshold of 1 px
        px_3 = correct_correspondences(flow_est, flow_target, alpha=3.0/float(img_size), img_size=img_size) # threshold of 3 px
        px_5 = correct_correspondences(flow_est, flow_target, alpha=5.0/float(img_size), img_size=img_size) # threshold of 5 px

        # percentage per image is calculated for each
        pck_alpha_0_01_over_image.append(alpha_0_01/flow_target.shape[0])
        pck_alpha_0_05_over_image.append(alpha_0_05/flow_target.shape[0])
        pck_alpha_0_1_over_image.append(alpha_0_1/flow_target.shape[0])
        pck_thresh_1_over_image.append(px_1/flow_target.shape[0])
        pck_thresh_3_over_image.append(px_3/flow_target.shape[0])
        pck_thresh_5_over_image.append(px_5/flow_target.shape[0])

        # PCK curve for different thresholds ! ATTENTION, here it is over the whole dataset and not per image
        n_registered_pxs += flow_target.shape[0]  # also equal to number of correspondences that should be correct
        # according to ground truth mask
        for t_id, threshold in enumerate(threshold_range):
            array_n_correct_correspondences[t_id] += correct_correspondences(flow_est,
                                                                             flow_target,
                                                                             alpha=threshold,
                                                                             img_size=img_size)
            # number of correct pixel correspondence below a certain threshold, added for each batch

        if compute_F1:
            F1 += F1_kitti_2015(flow_est, flow_target)

        if save:
            writeFlow(np.dstack([flow_est_x[0].cpu().numpy(), flow_est_y[0].cpu().numpy()]),
            'batch_{}'.format(i_batch), path_to_save)

    output = {'final_eape': np.mean(aepe_array),
              'pck_alpha_0_01_average_per_image': np.mean(pck_alpha_0_01_over_image),    
              'pck_alpha_0_05_average_per_image': np.mean(pck_alpha_0_05_over_image),
              'pck_alpha_0_1_average_per_image': np.mean(pck_alpha_0_1_over_image),
              'pck_thresh_1_average_per_image': np.mean(pck_thresh_1_over_image),
              'pck_thresh_3_average_per_image': np.mean(pck_thresh_3_over_image),
              'pck_thresh_5_average_per_image': np.mean(pck_thresh_5_over_image),
              'alpha_threshold': threshold_range.tolist(),
              'pixel_threshold': np.round(threshold_range * img_size.cpu().numpy(), 2).tolist(),
              'pck_per_threshold_over_dataset': np.float32(array_n_correct_correspondences /
                                                           (n_registered_pxs + 1e-6)).tolist()}

    print("Validation EPE: %f, pck_alpha=0_01: %f, pck_alpha=0_05: %f, pck_alpha=0_1: %f, PCK_1: %f, PCK_3: %f, PCK_5: %f" % (output['final_eape'], output['pck_alpha_0_01_average_per_image'], output['pck_alpha_0_05_average_per_image'], 
                                                                  output['pck_alpha_0_1_average_per_image'], output['pck_thresh_1_average_per_image'], output['pck_thresh_3_average_per_image'], output['pck_thresh_5_average_per_image']))

    if compute_F1:
        output['kitti2015-F1'] = F1 / float(n_registered_pxs)
    return output

def test(flownet, posenet, motionlist, 
                   test_loader, 
                   device,
                   save_path,
                   apply_mask=False,
                   sparse=False):
    """
    evaluate epoch script
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

    #vonet.eval()
    flownet.eval()
    posenet.eval()
    with torch.no_grad():
        pbar = tqdm(enumerate(test_loader), total=len(test_loader))

        for i, mini_batch in pbar:
            source_image, target_image = pre_process_data(mini_batch['source_image'],
                                                        mini_batch['target_image'],
                                                        device=device)
            #intrinsic, flow, mask = None, None, None
            intrinsic = mini_batch['intrinsic'].float().to(device)
            #flow = mini_batch['flow'].float().to(device)
            bs, _, h_original, w_original = intrinsic.shape
            '''
            flow = F.interpolate(flow, (640, 640),
                                        mode='bilinear', align_corners=False)
            flow[:, 0, :, :] *= 640.0 / float(w_original)
            flow[:, 1, :, :] *= 640.0 / float(h_original)
            '''
            intrinsic = F.interpolate(intrinsic, (640, 640),
                                        mode='bilinear', align_corners=False)
            intrinsic[:, 0, :, :] *= 640.0 / float(w_original)
            intrinsic[:, 1, :, :] *= 640.0 / float(h_original)

            '''
            #TartanAir
            output_net_256, output_net_original = flownet(source_image, target_image, source_image_256, target_image_256)
            
            #pose net
            flow_scale = 20.0
            '''
            #intrinsic = mini_batch['intrinsic'].float().to(device)
            #flow_output = flownet(source_image, target_image)
            flow_output = flownet(stage='vo', image1=source_image, image2=target_image)            
            #import pdb; pdb.set_trace()
            flow_scale = 20.0
            flow_input = flow_output[-1].clone()/flow_scale
            #pose_output = posenet(torch.cat((flow_input, mini_batch['intrinsic'].to(device)),1))  
            pose_output = posenet(stage='vo', x=torch.cat((flow_input, intrinsic),1))
            #flow_output, pose_output = vonet([target_image, source_image, intrinsic])
            #flow_output, pose_output = vonet([source_image, target_image, intrinsic])
            
            #with flow_output 
            '''
            flow_input = output_net_original[1].clone()/flow_scale #20은 tartanvo값임
            motion = posenet(torch.cat((flow_input, mini_batch['intrinsic'].to(device)),1))
            '''
            motionnp = pose_output.cpu().numpy()
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
    return motionlist

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
