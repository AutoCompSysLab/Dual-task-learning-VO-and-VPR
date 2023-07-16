import torch.utils.data as data
import os
import os.path
from imageio import imread
import numpy as np
import torch


def get_gt_correspondence_mask(flow):
    # convert flow to mapping
    h,w = flow.shape[:2]
    X, Y = np.meshgrid(np.linspace(0, w - 1, w),
                       np.linspace(0, h - 1, h))
    map_x = (flow[:,:,0]+X).astype(np.float32)
    map_y = (flow[:,:,1]+Y).astype(np.float32)

    mask_x = np.logical_and(map_x>0, map_x< w)
    mask_y = np.logical_and(map_y>0, map_y< h)
    mask = np.logical_and(mask_x, mask_y).astype(np.uint8)
    return mask


def train_default_loader(root, path_imgs, path_flo, path_mask):
    #pose_root = '/home/jovyan/datasets/TartanAir'
    imgs = [os.path.join(root,path) for path in path_imgs]
    flo = os.path.join(root,path_flo)
    mask = os.path.join(root,path_mask)

    return [imread(img).astype(np.uint8) for img in imgs], np.load(flo), np.load(mask)

def test_default_loader(root, path_imgs):
    imgs = [os.path.join(root,path) for path in path_imgs]

    return [imread(img).astype(np.uint8) for img in imgs]

def make_intrinsics_layer(w, h, fx, fy, ox, oy):
    ww, hh = np.meshgrid(range(w), range(h))
    ww = (ww.astype(np.float32) - ox + 0.5 )/fx
    hh = (hh.astype(np.float32) - oy + 0.5 )/fy
    intrinsicLayer = np.stack((ww,hh)).transpose(1,2,0)

    return intrinsicLayer

class TrainListDataset(data.Dataset):
    def __init__(self, root, path_list, source_image_transform=None, target_image_transform=None, flow_transform=None,
                 co_transform=None, loader=train_default_loader, mask=True, size=False, transform = None, 
                    focalx = 320.0, focaly = 320.0, centerx = 320.0, centery = 240.0):
        """

        :param root: directory containing the dataset images
        :param path_list: list containing the name of images and corresponding ground-truth flow files
        :param source_image_transform: transforms to apply to source images
        :param target_image_transform: transforms to apply to target images
        :param flow_transform: transforms to apply to flow field
        :param co_transform: transforms to apply to both images and the flow field
        :param loader: loader function for the images and the flow
        :param mask: bool indicating is a mask of valid pixels needs to be loaded as well from root
        :param size: size of the original source image
        outputs:
            - source_image
            - target_image
            - flow_map
            - correspondence_mask
            - source_image_size
        """

        self.root = root
        self.path_list = path_list
        self.source_image_transform = source_image_transform
        self.target_image_transform = target_image_transform
        self.flow_transform = flow_transform
        self.co_transform = co_transform
        self.loader = loader
        self.mask = mask
        self.size = size

        self.transform = transform
        self.focalx = focalx
        self.focaly = focaly
        self.centerx = centerx
        self.centery = centery

    def __getitem__(self, index):
        # for all inputs[0] must be the source and inputs[1] must be the target
        inputs, gt_flow, gt_mask, gt_motion = self.path_list[index]

        if not self.mask:
            if self.size:
                inputs, gt_flow, gt_mask, source_size = self.loader(self.root, inputs, gt_flow, gt_mask)
            else:
                inputs, gt_flow, gt_mask = self.loader(self.root, inputs, gt_flow, gt_mask)
                source_size = inputs[0].shape
            #if self.co_transform is not None:
                #inputs, gt_flow = self.co_transform(inputs, gt_flow)

            mask = get_gt_correspondence_mask(gt_flow)
        else:
            if self.size:
                inputs, gt_flow, mask, source_size = self.loader(self.root, inputs, gt_flow, gt_mask)
            else:
                # loader comes with a mask of valid correspondences
                inputs, gt_flow, mask = self.loader(self.root, inputs, gt_flow, gt_mask)
                source_size = inputs[0].shape
            # mask is shape hxw
            #if self.co_transform is not None:
                #inputs, gt_flow, mask = self.co_transform(inputs, gt_flow, mask)
        

        # here gt_flow has shape HxWx2

        # after co transform that could be reshapping the target
        # transforms here will always contain conversion to tensor (then channel is before)
        res = {'source_image': inputs[0], 'target_image': inputs[1]}

        h, w, _ = inputs[0].shape
        intrinsicLayer = make_intrinsics_layer(w, h, self.focalx, self.focaly, self.centerx, self.centery)
        res['intrinsic'] = intrinsicLayer
        res['flow'] = gt_flow
        if self.transform:
            res = self.transform(res)
        '''
        if self.source_image_transform is not None:
            inputs[0] = self.source_image_transform(inputs[0])
        if self.target_image_transform is not None:
            inputs[1] = self.target_image_transform(inputs[1])
        if self.flow_transform is not None:
            gt_flow = self.flow_transform(gt_flow)
        '''
        res['correspondence_mask'] = mask.astype(np.bool) if float(torch.__version__[:3]) >= 1.1  else mask.astype(np.uint8)
        res['source_image_size'] =source_size
        res['motion'] = gt_motion
        return res
        '''
        return {'source_image': inputs[0],
                'target_image': inputs[1],
                'flow_map': gt_flow,
                'correspondence_mask': mask.astype(np.bool) if float(torch.__version__[:3]) >= 1.1
                else mask.astype(np.uint8),
                'source_image_size': source_size
                }
        '''

    def __len__(self):
        return len(self.path_list)

class TestListDataset(data.Dataset):
    def __init__(self, root, path_list, source_image_transform=None, target_image_transform=None, flow_transform=None,
                 co_transform=None, loader=test_default_loader, mask=True, size=False, transform = None, 
                    focalx = 320.0, focaly = 320.0, centerx = 320.0, centery = 240.0):
        """

        :param root: directory containing the dataset images
        :param path_list: list containing the name of images and corresponding ground-truth flow files
        :param source_image_transform: transforms to apply to source images
        :param target_image_transform: transforms to apply to target images
        :param flow_transform: transforms to apply to flow field
        :param co_transform: transforms to apply to both images and the flow field
        :param loader: loader function for the images and the flow
        :param mask: bool indicating is a mask of valid pixels needs to be loaded as well from root
        :param size: size of the original source image
        outputs:
            - source_image
            - target_image
            - flow_map
            - correspondence_mask
            - source_image_size
        """

        self.root = root
        self.path_list = path_list
        self.source_image_transform = source_image_transform
        self.target_image_transform = target_image_transform
        self.flow_transform = flow_transform
        self.co_transform = co_transform
        self.loader = loader
        self.mask = mask
        self.size = size

        self.transform = transform
        self.focalx = focalx
        self.focaly = focaly
        self.centerx = centerx
        self.centery = centery

    def __getitem__(self, index):
        # for all inputs[0] must be the source and inputs[1] must be the target
        inputs, gt_motion = self.path_list[index]
        '''
        if not self.mask:
            if self.size:
                inputs, gt_flow, gt_mask, source_size = self.loader(self.root, inputs, gt_flow, gt_mask)
            else:
                inputs, gt_flow, gt_mask = self.loader(self.root, inputs, gt_flow, gt_mask)
                source_size = inputs[0].shape
            if self.co_transform is not None:
                inputs, gt_flow = self.co_transform(inputs, gt_flow)

            mask = get_gt_correspondence_mask(gt_flow)
        else:
            if self.size:
                inputs, gt_flow, mask, source_size = self.loader(self.root, inputs, gt_flow, gt_mask)
            else:
                # loader comes with a mask of valid correspondences
                inputs, gt_flow, mask = self.loader(self.root, inputs, gt_flow, gt_mask)
                source_size = inputs[0].shape
            # mask is shape hxw
            if self.co_transform is not None:
                inputs, gt_flow, mask = self.co_transform(inputs, gt_flow, mask)
        '''
        if self.size:
            inputs, source_size = self.loader(self.root, inputs)
        else:
            inputs = self.loader(self.root, inputs)
            source_size = inputs[0].shape
        if self.co_transform is not None:
            inputs, gt_flow = self.co_transform(inputs, gt_flow) # -> gt flow 없는데...
        

        # here gt_flow has shape HxWx2

        # after co transform that could be reshapping the target
        # transforms here will always contain conversion to tensor (then channel is before)
        res = {'source_image': inputs[0], 'target_image': inputs[1]}

        h, w, _ = inputs[0].shape
        intrinsicLayer = make_intrinsics_layer(w, h, self.focalx, self.focaly, self.centerx, self.centery)
        res['intrinsic'] = intrinsicLayer
        #res['flow'] = gt_flow
        if self.transform:
            res = self.transform(res)
        '''
        if self.source_image_transform is not None:
            inputs[0] = self.source_image_transform(inputs[0])
        if self.target_image_transform is not None:
            inputs[1] = self.target_image_transform(inputs[1])
        if self.flow_transform is not None:
            gt_flow = self.flow_transform(gt_flow)
        '''
        #res['correspondence_mask'] = mask.astype(np.bool) if float(torch.__version__[:3]) >= 1.1  else mask.astype(np.uint8)
        res['source_image_size'] =source_size
        res['motion'] = gt_motion
        return res
        '''
        return {'source_image': inputs[0],
                'target_image': inputs[1],
                'flow_map': gt_flow,
                'correspondence_mask': mask.astype(np.bool) if float(torch.__version__[:3]) >= 1.1
                else mask.astype(np.uint8),
                'source_image_size': source_size
                }
        '''

    def __len__(self):
        return len(self.path_list)

'''
    def __getitem__(self, index):
        # for all inputs[0] must be the source and inputs[1] must be the target
        inputs, gt_flow = self.path_list[index]

        if not self.mask:
            if self.size:
                inputs, gt_flow, source_size = self.loader(self.root, inputs, gt_flow)
            else:
                inputs, gt_flow = self.loader(self.root, inputs, gt_flow)
                source_size = inputs[0].shape
            if self.co_transform is not None:
                inputs, gt_flow = self.co_transform(inputs, gt_flow)

            mask = get_gt_correspondence_mask(gt_flow)
        else:
            if self.size:
                inputs, gt_flow, mask, source_size = self.loader(self.root, inputs, gt_flow)
            else:
                # loader comes with a mask of valid correspondences
                inputs, gt_flow, mask = self.loader(self.root, inputs, gt_flow)
                source_size = inputs[0].shape
            # mask is shape hxw
            if self.co_transform is not None:
                inputs, gt_flow, mask = self.co_transform(inputs, gt_flow, mask)

        # here gt_flow has shape HxWx2

        # after co transform that could be reshapping the target
        # transforms here will always contain conversion to tensor (then channel is before)
        if self.source_image_transform is not None:
            inputs[0] = self.source_image_transform(inputs[0])
        if self.target_image_transform is not None:
            inputs[1] = self.target_image_transform(inputs[1])
        if self.flow_transform is not None:
            gt_flow = self.flow_transform(gt_flow)

        return {'source_image': inputs[0],
                'target_image': inputs[1],
                'flow_map': gt_flow,
                'correspondence_mask': mask.astype(np.bool) if float(torch.__version__[:3]) >= 1.1
                else mask.astype(np.uint8),
                'source_image_size': source_size
                }

    def __len__(self):
        return len(self.path_list)
'''

