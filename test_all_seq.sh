SET=$(seq 0 7)
for i in $SET

do
    #python3 eval_cvpr.py --name_exp test_cvpr --evaluation_data_dir /home/jovyan/datasets/tartanair_cvpr  --pretrained_flownet ./snapshots/AddVPR_freezeVGG_nogroup_TartanAir_valtest_VPRloss0.1/flownet_epoch_18.pth --pretrained_posenet ./snapshots/AddVPR_freezeVGG_nogroup_TartanAir_valtest_VPRloss0.1/posenet_epoch_18.pth --batch-size 8 --test-seq MH00$i
    CUDA_VISIBLE_DEVICES=0,1,2,3 python3 test.py --name_exp tartanvo_test --evaluation_data_dir /home/main/storage/gpuserver00_storage/tartanair_cvpr  --pretrained_flownet /home/main/workspace/jeongwook/Trans_TartanVO_mixvpr_sum_with_VPRdata_new/snapshots/2023_12_24_20_22/flownet_epoch_1.pth --pretrained_posenet /home/main/workspace/jeongwook/Trans_TartanVO_mixvpr_sum_with_VPRdata_new/snapshots/2023_12_24_20_22/posenet_epoch_1.pth --batch-size 8 --test_seq MH00$i

    echo "Running loop seq "$i

    # some instructions

done