import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from os import listdir
from .transformation import pos_quats2SEs, pose2motion, SEs2ses
from .utils_tartanvo import make_intrinsics_layer
import os.path
from .listdataset import TestListDataset

def Tartanair_cvpr_Dataset(root, test_seq, source_image_transform=None, target_image_transform=None, flow_transform=None,
                   co_transform=None, valid_transform = None, train_transform=None,
                    focalx = 320.0, focaly = 320.0, centerx = 320.0, centery = 240.0):
    '''
    # dataloader for only abandonedfactory/Easy
    train_list = []
    test_list = []
    dataset_names = ['P000', 'P001', 'P002', 'P004', 'P005', 'P006', 'P008', 'P009', 'P010', 'P011']
    for dataset_name in dataset_names:
        image_path = os.path.join(root, dataset_name, 'image_left')
        flow_path = os.path.join(root, dataset_name, 'flow')
        image_list=[]
        flow_list=[]
        mask_list=[]
        for (roots, directories, files) in os.walk(image_path):
            directories.sort()
            files.sort()
            for image_file in files:
                image_list.append(os.path.join(dataset_name, 'image_left', image_file))
        for (roots, directories, files) in os.walk(flow_path):
            directories.sort()
            files.sort()
            for flow_file in files:
                if 'flow' in flow_file:
                    flow_list.append(os.path.join(dataset_name, 'flow', flow_file))
                elif 'mask' in flow_file:
                    mask_list.append(os.path.join(dataset_name, 'flow', flow_file))                       
        for i in range(len(flow_list)):
            if dataset_name != 'P010':
                train_list.append([[image_list[i], image_list[i+1]], flow_list[i], mask_list[i]])
            else:
                test_list.append([[image_list[i], image_list[i+1]], flow_list[i], mask_list[i]])
        
    print('Loading dataset at {}'.format(root))

    train_dataset = ListDataset(root, train_list, source_image_transform=source_image_transform,
                                target_image_transform=target_image_transform,
                                flow_transform=flow_transform, co_transform=co_transform)
    test_dataset = ListDataset(root, test_list, source_image_transform=source_image_transform,
                               target_image_transform=target_image_transform,
                               flow_transform=flow_transform, co_transform=co_transform)

    return train_dataset, test_dataset
    '''
    # dataloader for whole TartanAir
    pose_root = os.path.join(root, 'mono_gt')
    root = os.path.join(root, 'mono')
    dir_list = listdir(root)

    test_list = []
    image_list = []
    image_path = os.path.join(root, test_seq)
    posefile = os.path.join(pose_root, test_seq+'.txt') 
    print("test_seq : {}".format(test_seq))
    pose_list = np.loadtxt(posefile).astype(np.float32)
    assert(pose_list.shape[1]==7) # position + quaternion
    poses = pos_quats2SEs(pose_list)
    matrix_list = pose2motion(poses)
    motion_list = SEs2ses(matrix_list).astype(np.float32)
    # self.motions = self.motions / self.pose_std

    for (roots, directories, files) in os.walk(image_path):
        directories.sort()
        files.sort()
        for image_file in files:
            image_list.append(os.path.join(roots.replace(root + '/', ''), image_file))
    for i in range(len(image_list)-1):
        test_list.append([[image_list[i], image_list[i+1]], motion_list[i]])


    print('Loading test dataset at {}'.format(os.path.join(root, test_seq)))
    print('Evaluation dataset num : {}'.format(len(test_list)))

    test_dataset = TestListDataset(root, test_list, source_image_transform=source_image_transform,
                               target_image_transform=target_image_transform,
                               flow_transform=flow_transform, co_transform=co_transform, transform = valid_transform, 
                               focalx = focalx, focaly = focaly, centerx = centerx, centery = centery)


    return test_dataset