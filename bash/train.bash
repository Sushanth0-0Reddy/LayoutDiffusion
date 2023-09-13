
WORKSPACE="/content/gdrive/MyDrive/meronymnet/LayoutDiffusion"
cd ${WORKSPACE}



python -m torch.distributed.launch \
       --nproc_per_node 1 \
       scripts/image_train_for_layout.py \
       --config_file ./configs/MN_128x128/LayoutDiffusion_large.yaml
#"C:\GitHub\LayoutDiffusion\configs\MN_128x128\LayoutDiffusion_large.yaml"