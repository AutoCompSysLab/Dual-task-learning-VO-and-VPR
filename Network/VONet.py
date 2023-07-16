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
import torch.nn as nn
import torch.nn.functional as F
from .PWC.PWCNet import PWCDCNet as FlowNet
from .VOFlowNet import VOFlowRes as FlowPoseNet

class VONet(nn.Module):
    def __init__(self):
        super(VONet, self).__init__()

        #self.flowNet     = FlowNet()
        self.flowPoseNet = FlowPoseNet()

    def forward(self, x):
        # import ipdb;ipdb.set_trace() 
        #flow2, flow3, flow4, flow5, flow6 = self.flowNet(x[0:2])
        #flow = [flow2, flow3, flow4, flow5, flow6]
        #flow_scale = 20.0
        #import pdb; pdb.set_trace()
        #flow = flow[0]
        #flow2 = flow2/flow_scale
        flow_input = torch.cat( ( x[3], x[2] ), dim=1 )        
        pose = self.flowPoseNet( flow_input )

        return pose

    def get_flow_loss(self, netoutput, target, criterion, mask=None, training = True, small_scale=False):
        '''
        small_scale: the target flow and mask are down scaled (when in forward_vo)
        '''
        # netoutput 1/4, 1/8, ..., 1/32 size flow
        if mask is not None:
            return self.flowNet.get_loss_w_mask(netoutput, target, criterion, mask, training, small_scale=small_scale)
        else:
            return self.flowNet.get_loss(netoutput, target, criterion, training, small_scale=small_scale)
    

