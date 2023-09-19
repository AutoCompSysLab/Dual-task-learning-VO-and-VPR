SET=$(seq 0 7)
for i in $SET

do
    #python3 eval_cvpr.py --name_exp test_cvpr --evaluation_data_dir /home/jovyan/datasets/tartanair_cvpr  --pretrained_flownet ./snapshots/AddVPR_freezeVGG_nogroup_TartanAir_valtest_VPRloss0.1/flownet_epoch_18.pth --pretrained_posenet ./snapshots/AddVPR_freezeVGG_nogroup_TartanAir_valtest_VPRloss0.1/posenet_epoch_18.pth --batch-size 8 --test-seq MH00$i
    CUDA_VISIBLE_DEVICES=2 python3 test.py --name_exp tartanvo_test --evaluation_data_dir /home/main/storage/gpuserver00_storage/tartanair_cvpr  --pretrained_flownet /home/main/workspace/jeongwook/TartanVO_flowformer_mixvpr_sum/snapshots/2023_09_12_15_57/flownet_epoch_6.pth --pretrained_posenet /home/main/workspace/jeongwook/TartanVO_flowformer_mixvpr_sum/snapshots/2023_09_12_15_57/posenet_epoch_6.pth --batch-size 8 --test_seq MH00$i

    echo "Running loop seq "$i

    # some instructions

done