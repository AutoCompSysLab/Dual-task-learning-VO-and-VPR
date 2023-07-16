SET=$(seq 0 24)
for i in $SET

do
    #python3 eval_cvpr.py --name_exp test_cvpr --evaluation_data_dir /home/jovyan/datasets/tartanair_cvpr  --pretrained_flownet ./snapshots/AddVPR_freezeVGG_nogroup_TartanAir_valtest_VPRloss0.1/flownet_epoch_18.pth --pretrained_posenet ./snapshots/AddVPR_freezeVGG_nogroup_TartanAir_valtest_VPRloss0.1/posenet_epoch_18.pth --batch-size 8 --test-seq MH00$i
    CUDA_VISIBLE_DEVICES=4 python3 test.py --name_exp tartanvo_test --evaluation_data_dir /home/main/storage/gpuserver00_storage/tartanair_cvpr  --pretrained_model /home/main/workspace/jeongwook/TartanVO/snapshots/TartanVO_train/epoch_$i.pth --batch-size 8 --test_seq MH007

    echo "Running loop seq "$i

    # some instructions

done