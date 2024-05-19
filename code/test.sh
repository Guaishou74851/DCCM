# python -u test.py --gpu_list=1 --data_type=20240127_flureoscent_microsphere --gq=25 > scalable_FM_20240204_25.txt &
# python -u test.py --gpu_list=2 --data_type=20240127_flureoscent_microsphere --gq=30 > scalable_FM_20240204_30.txt &
# python -u test.py --gpu_list=0 --data_type=20240127_potato_tuber --gq=30 > scalable_PT_20240204_30.txt &
# python -u test.py --gpu_list=3 --data_type=20240127_potato_tuber --gq=39 > scalable_PT_20240204_39.txt &
# python -u test.py --gpu_list=0 --data_type=20240127_potato_tuber --gq=30 > /dev/null &
python -u test.py --gpu_list=2 --data_type=20240127_potato_tuber --gq=30 > /dev/null &
python -u test.py --gpu_list=3 --data_type=20240127_potato_tuber --gq=39 > /dev/null &
