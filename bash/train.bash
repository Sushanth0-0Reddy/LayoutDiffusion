
WORKSPACE="c:\GitHub\LayoutDiffusion"
cd ${WORKSPACE}

conda activate LayoutDiffusion

python -m torch.distributed.launch \
       --nproc_per_node 1 \
       scripts/image_train_for_layout.py \
       --config_file ./configs/COCO-stuff_256x256/LayoutDiffusion_small.yaml
#"C:\GitHub\LayoutDiffusion\configs\COCO-stuff_256x256\LayoutDiffusion_small.yaml"