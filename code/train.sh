nohup python -u train.py --gpu_list=0 --data_type=nucleus > nucleus_g0.txt &
nohup python -u train.py --gpu_list=1 --data_type=flureoscent_microsphere > fm_50_g1.txt &
nohup python -u train.py --gpu_list=2 --data_type=f-actin > fa_g2.txt &
nohup python -u train.py --gpu_list=3 --data_type=potato_tuber > pt_g3.txt &