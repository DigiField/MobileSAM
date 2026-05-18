CUDA_VISIBLE_DEVICES=0 python3 ./Inference.py  \
    --img_path './test_images/' \
    --output_dir './' \
    --encoder_type 'efficientvit_l2'
