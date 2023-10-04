SET=$(seq 0 7)
for i in $SET

do
    #python3 eval_cvpr.py --name_exp test_cvpr --evaluation_data_dir /home/jovyan/datasets/tartanair_cvpr  --pretrained_flownet ./snapshots/AddVPR_freezeVGG_nogroup_TartanAir_valtest_VPRloss0.1/flownet_epoch_18.pth --pretrained_posenet ./snapshots/AddVPR_freezeVGG_nogroup_TartanAir_valtest_VPRloss0.1/posenet_epoch_18.pth --batch-size 8 --test-seq MH00$i
    CUDA_VISIBLE_DEVICES=4,5,6,7 python3 test.py --name_exp tartanvo_test --evaluation_data_dir /home/main/storage/gpuserver00_storage/tartanair_cvpr  --pretrained_flownet /home/main/workspace/jeongwook/Trans_TartanVO_mixvpr_sum_with_VPRdata/snapshots/2023_09_22_11_18/flownet_epoch_5.pth --pretrained_posenet /home/main/workspace/jeongwook/Trans_TartanVO_mixvpr_sum_with_VPRdata/snapshots/2023_09_22_11_18/posenet_epoch_5.pth --batch-size 8 --test_seq MH00$i

    echo "Running loop seq "$i

    # some instructions

done