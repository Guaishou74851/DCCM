nohup python -u train.py --gpu_list=0 --data_type=flureoscent_microsphere --gq=40 > fm_40_g0.txt &
nohup python -u train.py --gpu_list=1 --data_type=flureoscent_microsphere --gq=50 > fm_50_g1.txt &
nohup python -u train.py --gpu_list=2 --data_type=potato_tuber --gq=30 > pt_30_g2.txt &
nohup python -u train.py --gpu_list=3 --data_type=potato_tuber --gq=39 > pt_39_g3.txt &
