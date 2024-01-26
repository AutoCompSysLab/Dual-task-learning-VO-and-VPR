from __future__ import division
import torch
import numpy as np
import torch.nn.functional as F
import math
import random
import numbers
import cv2
import matplotlib.pyplot as plt
import os
import time
#import wandb
import matplotlib
from torchvision.transforms import transforms
matplotlib.use('Agg')
if ( not ( "DISPLAY" in os.environ ) ):
    plt.switch_backend('agg')
    print("Environment variable DISPLAY is not present in the system.")
    print("Switch the backend of matplotlib to agg.")

def TensorToArray(tensor, type):
    """Converts a torch.FloatTensor of shape (C x H x W) to a numpy.ndarray (H x W x C) """
    array=tensor.cpu().detach().numpy()
    if len(array.shape)==4:
        if array.shape[3] > array.shape[1]:
            # shape is BxCxHxW
            array = np.transpose(array, (0,2,3,1))
    else:
        if array.shape[2] > array.shape[0]:
            # shape is CxHxW
            array=np.transpose(array, (1,2,0))
    return array.astype(type)


# class ToTensor of torchvision also normalised to 0 1
class ArrayToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""
    def __init__(self, get_float=True):
        self.get_float=get_float

    def __call__(self, array):

        if not isinstance(array, np.ndarray):
            array = np.array(array)
        array = np.transpose(array, (2, 0, 1))
        # handle numpy array
        tensor = torch.from_numpy(array)
        # put it from HWC to CHW format
        if self.get_float:
            return tensor.float()
        else:
            return tensor


class ResizeFlow(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""
    def __init__(self, size):
        if not isinstance(size, tuple):
            size = (size, size)
        self.size = size

    def __call__(self, tensor):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _, h_original, w_original = tensor.shape
        resized_tensor = F.interpolate(tensor.unsqueeze(0), self.size, mode='bilinear', align_corners=False)
        resized_tensor[:, 0, :, :] *= float(self.size[1])/float(w_original)
        resized_tensor[:, 1, :, :] *= float(self.size[0])/float(h_original)
        return resized_tensor.squeeze(0)


class RGBtoBGR(object):
    """converts the RGB channels of a numpy array HxWxC into RGB"""

    def __call__(self, array):
        assert(isinstance(array, np.ndarray))
        ch_arr = [2, 1, 0]
        img = array[..., ch_arr]
        return img

##########################################################################################
class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (List[Transform]): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class DownscaleFlow(object):
    """
    Scale the flow and mask to a fixed size

    """
    def __init__(self, scale=4):
        '''
        size: output frame size, this should be NO LARGER than the input frame size! 
        '''
        self.downscale = 1.0/scale 
    
    def __call__(self, sample): 
        
        '''
        if self.downscale!=1 and 'flow_map' in sample: #sample.has_key('flow'): #(448, 640, 2)
            sample['flow_map'] = cv2.resize(sample['flow_map'], 
                (0, 0), fx=self.downscale, fy=self.downscale, interpolation=cv2.INTER_LINEAR)
        '''
        if self.downscale!=1 and 'intrinsic' in sample: #sample.has_key('intrinsic'):
            sample['intrinsic'] = cv2.resize(sample['intrinsic'], 
                (0, 0), fx=self.downscale, fy=self.downscale, interpolation=cv2.INTER_LINEAR)
        '''
        if self.downscale!=1 and 'correspondence_mask' in sample: #sample.has_key('fmask'):
            sample['correspondence_mask'] = cv2.resize(sample['correspondence_mask'],
                (0, 0), fx=self.downscale, fy=self.downscale, interpolation=cv2.INTER_LINEAR)
        '''
        return sample

class CropCenter(object):
    """Crops the a sample of data (tuple) at center
    if the image size is not large enough, it will be first resized with fixed ratio
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
    def __call__(self, sample):
        kks = list(sample)
        th, tw = self.size
        h, w = sample[kks[0]].shape[0], sample[kks[0]].shape[1]
        if w == tw and h == th:
            return sample

        # resize the image if the image size is smaller than the target size
        scale_h, scale_w, scale = 1., 1., 1.
        if th > h:
            scale_h = float(th)/h
        if tw > w:
            scale_w = float(tw)/w
        if scale_h>1 or scale_w>1:
            scale = max(scale_h, scale_w)
            w = int(round(w * scale)) # w after resize
            h = int(round(h * scale)) # h after resize

        x1 = int((w-tw)/2)
        y1 = int((h-th)/2)

        for kk in kks:
            if sample[kk] is None:
                continue
            img = sample[kk]
            if len(img.shape)==3: 
                if scale>1:
                    img = cv2.resize(img, (w,h), interpolation=cv2.INTER_LINEAR)
                sample[kk] = img[y1:y1+th,x1:x1+tw,:]
            elif len(img.shape)==2:
                if scale>1:
                    img = cv2.resize(img, (w,h), interpolation=cv2.INTER_LINEAR)
                sample[kk] = img[y1:y1+th,x1:x1+tw]

        return sample

class RandomCropandResize(object):
    def __init__(self, size, scale, ratio):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, sample):
        kks = list(sample)
        transform = transforms.RandomResizedCrop(self.size, scale = self.scale, ratio=self.ratio)
        for kk in kks:
            sample[kk] = transform(torch.tensor(sample[kk]).permute(2,1,0)) #(3, 448, 640) 또는 (2, 448, 640) tensor
            sample[kk] = sample[kk].permute(1,2,0).numpy() #(448, 640, 3) 또는 (448, 640, 2) 
        return sample
        
class ToTensor(object):
    def __call__(self, sample):
        
        sss = time.time()

        kks = list(sample)
        
        for kk in kks:
            data = sample[kk]
            data = data.astype(np.float32) 
            
            if len(data.shape) == 3: # transpose image-like data
                data = data.transpose(2,0,1)
            elif len(data.shape) == 2:
                data = data.reshape((1,)+data.shape)

            #수정
            # if len(data.shape) == 3 and data.shape[0]==3: # normalization of rgb images -> Normalize using Imagenet in GLU
            #     data = data/255.0
            
            sample[kk] = torch.from_numpy(data.copy()) # copy to make memory continuous
            #torch.Size([3, 448, 640]) torch.Size([3, 448, 640]) torch.Size([2, 112, 160])
            
        return sample


def tensor2img(tensImg,mean,std):
    """
    convert a tensor a numpy array, for visualization
    """
    # undo normalize
    for t, m, s in zip(tensImg, mean, std):
        t.mul_(s).add_(m) 
    tensImg = tensImg * float(255)
    # undo transpose
    tensImg = (tensImg.numpy().transpose(1,2,0)).astype(np.uint8)
    return tensImg

def bilinear_interpolate(img, h, w):
    # assert round(h)>=0 and round(h)<img.shape[0]
    # assert round(w)>=0 and round(w)<img.shape[1]

    h0 = int(math.floor(h))
    h1 = h0 + 1
    w0 = int(math.floor(w))
    w1 = w0 + 1

    a = h - h0 
    b = w - w0

    h0 = max(h0, 0)
    w0 = max(w0, 0)
    h1 = min(h1, img.shape[0]-1)
    w1 = min(w1, img.shape[1]-1)

    A = img[h0,w0,:]
    B = img[h1,w0,:]
    C = img[h0,w1,:]
    D = img[h1,w1,:]

    res = (1-a)*(1-b)*A + a*(1-b)*B + (1-a)*b*C + a*b*D

    return res 

def calculate_angle_distance_from_du_dv(du, dv, flagDegree=False):
    a = np.arctan2( dv, du )

    angleShift = np.pi

    if ( True == flagDegree ):
        a = a / np.pi * 180
        angleShift = 180
        # print("Convert angle from radian to degree as demanded by the input file.")

    d = np.sqrt( du * du + dv * dv )

    return a, d, angleShift

def visflow(flownp, maxF=500.0, n=8, mask=None, hueMax=179, angShift=0.0): 
    """
    Show a optical flow field as the KITTI dataset does.
    Some parts of this function is the transform of the original MATLAB code flow_to_color.m.
    """

    ang, mag, _ = calculate_angle_distance_from_du_dv( flownp[:, :, 0], flownp[:, :, 1], flagDegree=False )

    # Use Hue, Saturation, Value colour model 
    hsv = np.zeros( ( ang.shape[0], ang.shape[1], 3 ) , dtype=np.float32)

    am = ang < 0
    ang[am] = ang[am] + np.pi * 2

    hsv[ :, :, 0 ] = np.remainder( ( ang + angShift ) / (2*np.pi), 1 )
    hsv[ :, :, 1 ] = mag / maxF * n
    hsv[ :, :, 2 ] = (n - hsv[:, :, 1])/n

    hsv[:, :, 0] = np.clip( hsv[:, :, 0], 0, 1 ) * hueMax
    hsv[:, :, 1:3] = np.clip( hsv[:, :, 1:3], 0, 1 ) * 255
    hsv = hsv.astype(np.uint8)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if ( mask is not None ):
        mask = mask != 255
        bgr[mask] = np.array([0, 0 ,0], dtype=np.uint8)

    return bgr


def dataset_intrinsics(dataset='tartanair'):
    if dataset == 'kitti':
        focalx, focaly, centerx, centery = 707.0912, 707.0912, 601.8873, 183.1104
    elif dataset == 'euroc':
        focalx, focaly, centerx, centery = 458.6539916992, 457.2959899902, 367.2149963379, 248.3750000000
    elif dataset == 'tartanair':
        focalx, focaly, centerx, centery = 320.0, 320.0, 320.0, 240.0
    else:
        return None
    return focalx, focaly, centerx, centery

from mpl_toolkits.mplot3d import axes3d
def plot_traj_3d(gtposes, estposes, vis=False, savefigname=None, title=''):
    fig = plt.figure(figsize=(4,4))
    cm = plt.cm.get_cmap('Spectral')
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(gtposes[:,0],gtposes[:,1], gtposes[:,2], linestyle='dashed',c='k')
    ax.plot(estposes[:, 0], estposes[:, 1], estposes[:, 2],c='#ff7f0e')
    ax.legend(['Ground Truth', 'Ours'])
    ax.set_title(title)
    if savefigname is not None:
        plt.savefig(savefigname)
    if vis:
        plt.show()
    plt.close(fig)


def plot_traj(gtposes, estposes, vis=False, savefigname=None, title=''):
    fig = plt.figure(figsize=(4,4))
    cm = plt.cm.get_cmap('Spectral')

    plt.subplot(111)
    plt.plot(gtposes[:,0],gtposes[:,1], linestyle='dashed',c='k')
    plt.plot(estposes[:, 0], estposes[:, 1],c='#ff7f0e')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend(['Ground Truth', 'Ours'])
    plt.title(title)
    if savefigname is not None:
        plt.savefig(savefigname)
    if vis:
        plt.show()
    plt.close(fig)
'''
def make_intrinsics_layer(w, h, fx, fy, ox, oy):
    ww, hh = np.meshgrid(range(w), range(h))
    ww = (ww.astype(np.float32) - ox + 0.5 )/fx
    hh = (hh.astype(np.float32) - oy + 0.5 )/fy
    intrinsicLayer = np.stack((ww,hh)).transpose(1,2,0)

    return intrinsicLayer
'''
def load_kiiti_intrinsics(filename):
    '''
    load intrinsics from kitti intrinsics file
    '''
    with open(filename, 'r') as f:
        lines = f.readlines()
    cam_intrinsics = lines[2].strip().split(' ')[1:]
    focalx, focaly, centerx, centery = float(cam_intrinsics[0]), float(cam_intrinsics[5]), float(cam_intrinsics[2]), float(cam_intrinsics[6])

    return focalx, focaly, centerx, centery

def generate_random_scale_crop(h, w, target_h, target_w, scale_base, keep_center, fix_ratio):
    '''
    Randomly generate scale and crop params
    H: input image h
    w: input image w
    target_h: output image h
    target_w: output image w
    scale_base: max scale up rate
    keep_center: crop at center
    fix_ratio: scale_h == scale_w
    '''
    scale_w = random.random() * (scale_base - 1) + 1
    if fix_ratio:
        scale_h = scale_w
    else:
        scale_h = random.random() * (scale_base - 1) + 1

    crop_w = int(math.ceil(target_w/scale_w)) # ceil for redundancy
    crop_h = int(math.ceil(target_h/scale_h)) # crop_w * scale_w > w

    if keep_center:
        x1 = int((w-crop_w)/2)
        y1 = int((h-crop_h)/2)
    else:
        x1 = random.randint(0, w - crop_w)
        y1 = random.randint(0, h - crop_h)

    return scale_w, scale_h, x1, y1, crop_w, crop_h

class ResizeData(object):
    """Resize the data in a dict
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        kks = list(sample.keys())
        th, tw = self.size
        h, w = sample[kks[0]].shape[0], sample[kks[0]].shape[1]
        if w == tw and h == th:
            return sample
        scale_w = float(tw)/w
        scale_h = float(th)/h

        for kk in kks:
            if sample[kk] is None:
                continue
            sample[kk] = cv2.resize(sample[kk], (tw,th), interpolation=cv2.INTER_LINEAR)

        if 'flow' in sample:
            sample['flow'][...,0] = sample['flow'][...,0] * scale_w
            sample['flow'][...,1] = sample['flow'][...,1] * scale_h

        return sample
        
class RandomResizeCrop(object):
    """
    Random scale to cover continuous focal length
    Due to the tartanair focal is already small, we only up scale the image

    """
    def __init__(self, size, max_scale=2.5, keep_center=False, fix_ratio=False, scale_disp=False):
        '''
        size: output frame size, this should be NO LARGER than the input frame size! 
        scale_disp: when training the stereovo, disparity represents depth, which is not scaled with resize 
        '''
        if isinstance(size, numbers.Number):
            self.target_h = int(size)
            self.target_w = int(size)
        else:
            self.target_h = size[0]
            self.target_w = size[1]

        # self.max_focal = max_focal
        self.keep_center = keep_center
        self.fix_ratio = fix_ratio
        self.scale_disp = scale_disp
        # self.tartan_focal = 320.

        # assert self.max_focal >= self.tartan_focal
        self.scale_base = max_scale #self.max_focal /self.tartan_focal

    def __call__(self, sample): 
        for kk in sample:
            if len(sample[kk].shape)>=2:
                h, w = sample[kk].shape[0], sample[kk].shape[1]
                break
        self.target_h = min(self.target_h, h)
        self.target_w = min(self.target_w, w)

        scale_w, scale_h, x1, y1, crop_w, crop_h = generate_random_scale_crop(h, w, self.target_h, self.target_w, 
                                                    self.scale_base, self.keep_center, self.fix_ratio)

        for kk in sample:
            # if kk in ['flow', 'flow2', 'img0', 'img0n', 'img1', 'img1n', 'intrinsic', 'fmask', 'disp0', 'disp1', 'disp0n', 'disp1n']:
            if len(sample[kk].shape)>=2 or kk in ['correspondence_mask']:#['fmask', 'fmask2']:
                sample[kk] = sample[kk][y1:y1+crop_h, x1:x1+crop_w]
                sample[kk] = cv2.resize(sample[kk], (0,0), fx=scale_w, fy=scale_h, interpolation=cv2.INTER_LINEAR)
                # Note opencv reduces the last dimention if it is one
                sample[kk] = sample[kk][:self.target_h,:self.target_w]

        # scale the flow
        if 'flow_map' in sample:
            sample['flow_map'][:,:,0] = sample['flow_map'][:,:,0] * scale_w
            sample['flow_map'][:,:,1] = sample['flow_map'][:,:,1] * scale_h
        # scale the flow
        '''
        if 'flow2' in sample:
            sample['flow2'][:,:,0] = sample['flow2'][:,:,0] * scale_w
            sample['flow2'][:,:,1] = sample['flow2'][:,:,1] * scale_h

        if self.scale_disp: # scale the depth
            if 'disp0' in sample:
                sample['disp0'][:,:] = sample['disp0'][:,:] * scale_w
            if 'disp1' in sample:
                sample['disp1'][:,:] = sample['disp1'][:,:] * scale_w
            if 'disp0n' in sample:
                sample['disp0n'][:,:] = sample['disp0n'][:,:] * scale_w
            if 'disp1n' in sample:
                sample['disp1n'][:,:] = sample['disp1n'][:,:] * scale_w
        else:
            sample['scale_w'] = np.array([scale_w ])# used in e2e-stereo-vo
        '''

        return sample

