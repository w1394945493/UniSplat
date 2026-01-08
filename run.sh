# todo UniSplat 推理
python demo.py --load_from /path/to/checkpoint.pth --data_path /path/to/demo_data

export CUDA_VISIBLE_DEVICES=2
python /home/lianghao/wangyushen/Projects/UniSplat/demo.py \
    --config /home/lianghao/wangyushen/Projects/UniSplat/configs/waymo.yaml \
    --work_dir_root /home/lianghao/wangyushen/data/wangyushen/Output/unisplat/outputs/demo \
    --load_from /home/lianghao/wangyushen/data/wangyushen/Weights/unisplat/model.safetensors \
    --data_path /home/lianghao/wangyushen/data/wangyushen/Datasets/unisplat/demo/data/waymo