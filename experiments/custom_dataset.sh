cd src
python train.py --exp_id mark --data_cfg '../src/lib/cfg/mark.json' --lr 5e-4 --batch_size 4 --wh_weight 0.5 --multi_loss 'fix' --arch 'yolo' --reid_dim 64
cd ..