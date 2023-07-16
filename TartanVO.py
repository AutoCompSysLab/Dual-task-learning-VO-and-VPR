# Software License Agreement (BSD License)
#
# Copyright (c) 2020, Wenshan Wang, Yaoyu Hu,  CMU
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of CMU nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import torch
import numpy as np
import time

np.set_printoptions(precision=4, suppress=True, threshold=10000)

import torch 
import torch.nn as nn
import torch.nn.functional as F
from Network.VONet import VONet


class TartanVO(nn.Module):
    def __init__(self):
        super(TartanVO, self).__init__()
        # import ipdb;ipdb.set_trace()
        self.vonet = VONet()
        #self.vonet.cuda()

        #self.test_count = 0
        self.pose_std = np.array([ 0.13,  0.13,  0.13,  0.013 ,  0.013,  0.013], dtype=np.float32) # the output scale factor
        self.flow_norm = 20 # scale factor for flow
    
    def forward(self, x):
        pose = self.vonet(x)

        return pose

    def test_batch(self, sample):
        self.test_count += 1
        
        # import ipdb;ipdb.set_trace()
        img0   = sample['img1'].cuda()
        img1   = sample['img2'].cuda()
        intrinsic = sample['intrinsic'].cuda()
        inputs = [img0, img1, intrinsic]

        self.vonet.eval()

        with torch.no_grad():
            starttime = time.time()
            flow, pose = self.vonet(inputs)
            inferencetime = time.time()-starttime
            # import ipdb;ipdb.set_trace()
            posenp = pose.data.cpu().numpy()
            posenp = posenp * self.pose_std # The output is normalized during training, now scale it back
            flownp = flow.data.cpu().numpy()
            flownp = flownp * self.flow_norm

        # calculate scale from GT posefile
        if 'motion' in sample:
            motions_gt = sample['motion']
            scale = np.linalg.norm(motions_gt[:,:3], axis=1)
            trans_est = posenp[:,:3]
            trans_est = trans_est/np.linalg.norm(trans_est,axis=1).reshape(-1,1)*scale.reshape(-1,1)
            posenp[:,:3] = trans_est 
        else:
            print('    scale is not given, using 1 as the default scale value..')

        print("{} Pose inference using {}s: \n{}".format(self.test_count, inferencetime, posenp))
        return posenp, flownp

