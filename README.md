## Dual-task learning with VO and VPR
Robust camera pose estimation for large displacement trajectories through dual-task learning of visual odometry and visual place recognition

## Abstract
Visual Odometry (VO) is a crucial task for estimating the current pose of intelligent agents---particularly in applications such as autonomous driving and Simultaneous Localization and Mapping (SLAM). 
However, accurately estimating the pose of intelligent agents becomes challenging when significant displacement occurs between consecutive image frames. Also, camera pose estimation and loop closure detection which are the main modules of the SLAM task have utilized separate networks in previous works---increasing computations for resource-constrained environments.
To overcome these limitations, we propose an architecture that robustly estimates relative pose between consecutive image frames with large displacement. By formulating dual-task learning of VO and VPR, the proposed architecture leverages both local and global contexts to handle large displacement---reducing the overall computation as well. Our proposed network demonstrates notable performance enhancement for substantial displacement trajectories in TartanAir and DynaKITTI benchmark datasets---showcasing its effectiveness and potential for applications in real-world scenarios, such as autonomous navigation and mapping.

## Train
After downloading TartanAir and GSVCites dataset, run following code:
```
python train.py --pre_loaded_training_dataset True --training_data_dir PATH/TO/DATASET --pretrained_posenet PATH/TO/PRETRAINED_TRIPLE_HEAD_NETWORK
```

## Evaluation on VO task
To test for TartanAir test dataset, first modify evaluation_data_dir, pretrained_flownet, and pretrained_posenet to appropriate path run follwing code:
```
sh test_all_seq.sh
```

To test for DynaKITTI test dataset, run following code:
```
python test_kitti.py --evaluation_data_dir --pretrained_flownet PATH/TO/PRETRAINED_FLOW_NETWORK  --pretrained_posenet PATH/TO/PRETRAINED_TRIPLE_HEAD_NETWORK --test_seq TEST_SEQUENCE_NUMBER
```

## Evaluation on VPR task

```
python test_vpr.py --pretrained_flownet PATH/TO/PRETRAINED_FLOW_NETWORK --pretrained_posenet PATH/TO/PRETRAINED_TRIPLE_HEAD_NETWORK
```

## Acknowledgements
Our code is implemented utillizing part of codes in:
[TartanVO](https://github.com/castacks/tartanvo.git)
[Compass](https://github.com/microsoft/COMPASS.git)
[Flowformer](https://github.com/drinkingcoder/FlowFormer-Official.git)
[MixVPR](https://github.com/amaralibey/MixVPR.git)

## License
This software is BSD licensed.

Copyright (c) 2020, Carnegie Mellon University All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
