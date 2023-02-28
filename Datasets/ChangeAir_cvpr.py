import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from os import listdir
from .transformation import pos_quats2SEs, pose2motion, SEs2ses
from .utils_tartanvo import make_intrinsics_layer
import os.path
from .listdataset import TestListDataset

def Changeair_cvpr_Dataset(root, source_image_transform=None, target_image_transform=None, flow_transform=None,
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
    pose_root = '/home/jovyan/datasets/tartanair_cvpr/mono_gt'
    root = os.path.join(root, 'mono')
    dir_list = listdir(root)
    #dir_list = ['amusement', 'oldtown', 'neighborhood', 'soulcity', 'japanesealley', 'office', 'office2', 'seasidetown', 'abandonedfactory', 'hospital']
    #data_type_list = ['Easy', 'Hard']

    train_list = []
    test_list = []
    for dataset_name in dir_list:
        subpath_image_ref = os.path.join(root, dataset_name, 'ref', 'image')
        subpath_image_query = os.path.join(root, dataset_name, 'query', 'image')
        #subpath_flow = os.path.join(path, subdir, 'flow')
        #posefile = os.path.join(path, subdir, 'pose_left.txt')
        #subpath_flow = os.path.join(pose_root, dataset_name, dataset_name, data_type_name, subdir, 'flow')
        posefile = os.path.join(pose_root, dataset_name + '.txt')
        image_list_ref=[]
        image_list_query=[]
        #flow_list=[]
        #mask_list=[]
        pose_list = np.loadtxt(posefile).astype(np.float32)
        assert(pose_list.shape[1]==7) # position + quaternion
        poses = pos_quats2SEs(pose_list)
        matrix_list = pose2motion(poses)
        motion_list = SEs2ses(matrix_list).astype(np.float32)
        # self.motions = self.motions / self.pose_std

        for (roots, directories, files) in os.walk(subpath_image_ref):
            directories.sort()
            files.sort()
            for image_file in files:
                image_list_ref.append(os.path.join(roots.replace(root + '/', ''), image_file))
        for (roots, directories, files) in os.walk(subpath_image_query):
            directories.sort()
            files.sort()
            for image_file in files:
                image_list_query.append(os.path.join(roots.replace(root + '/', ''), image_file))
        '''
        for (roots, directories, files) in os.walk(subpath_flow):
            directories.sort()
            files.sort()
            for flow_file in files:
                if 'flow' in flow_file:
                    flow_list.append(os.path.join(roots.replace(pose_root + '/', ''), flow_file))
                elif 'mask' in flow_file:
                    mask_list.append(os.path.join(roots.replace(pose_root + '/', ''), flow_file))
        '''
        for i in range(len(image_list_ref)):
            #if dataset_name == 'abandonedfactory' or dataset_name == 'hospital':
            test_list.append([[image_list_ref[i], image_list_query[i]], motion_list[i]])


    print('Loading test dataset at {}'.format(root))

    test_dataset = TestListDataset(root, test_list, source_image_transform=source_image_transform,
                               target_image_transform=target_image_transform,
                               flow_transform=flow_transform, co_transform=co_transform, transform = valid_transform, 
                               focalx = focalx, focaly = focaly, centerx = centerx, centery = centery)


    return test_dataset