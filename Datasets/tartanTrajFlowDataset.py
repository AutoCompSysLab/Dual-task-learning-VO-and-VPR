import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from os import listdir
from .transformation import pos_quats2SEs, pose2motion, SEs2ses
from .utils_tartanvo import make_intrinsics_layer

class TrajFolderDataset(Dataset):
    """scene flow synthetic dataset. """

    def __init__(self, ref_imgfolder, query_imgfolder, posefile = None, transform = None, 
                    focalx = 320.0, focaly = 320.0, centerx = 320.0, centery = 240.0):
        
        ref_files = listdir(ref_imgfolder)
        query_files = listdir(query_imagefolder)
        self.ref_rgbfiles = [(imgfolder +'/'+ ff) for ff in files if (ff.endswith('.png') or ff.endswith('.jpg'))]
        self.ref_rgbfiles.sort()
        self.query_rgbfiles = [(imgfolder +'/'+ ff) for ff in files if (ff.endswith('.png') or ff.endswith('.jpg'))]
        self.query_rgbfiles.sort()
        self.ref_imgfolder = ref_imgfolder
        self.query_imgfolder = query_imgfolder

        print('Find {} image files in {}'.format(len(self.rgbfiles), imgfolder))

        if posefile is not None and posefile!="":
            poselist = np.loadtxt(posefile).astype(np.float32)
            assert(poselist.shape[1]==7) # position + quaternion
            poses = pos_quats2SEs(poselist)
            self.matrix = pose2motion(poses)
            self.motions     = SEs2ses(self.matrix).astype(np.float32)
            # self.motions = self.motions / self.pose_std
            assert(len(self.motions) == len(self.rgbfiles)) - 1
        else:
            self.motions = None

        self.N = len(self.ref_rgbfiles)

        # self.N = len(self.lines)
        self.transform = transform
        self.focalx = focalx
        self.focaly = focaly
        self.centerx = centerx
        self.centery = centery

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        imgfile1 = self.ref_rgbfiles[idx].strip()
        imgfile2 = self.query_rgbfiles[idx].strip()
        #import pdb; pdb.set_trace()
        img1 = cv2.imread(imgfile1)
        img2 = cv2.imread(imgfile2)

        res = {'img1': img1, 'img2': img2 }

        h, w, _ = img1.shape
        intrinsicLayer = make_intrinsics_layer(w, h, self.focalx, self.focaly, self.centerx, self.centery)
        res['intrinsic'] = intrinsicLayer

        if self.transform:
            res = self.transform(res)

        if self.motions is None:
            return res
        else:
            res['motion'] = self.motions[idx]
            return res


