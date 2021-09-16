python write_to_filelist.py --split train --pose_config uncover
python write_to_filelist.py --split train --pose_config cover1
python write_to_filelist.py --split train --pose_config cover2

python write_to_filelist.py --split valid --pose_config cover1
python write_to_filelist.py --split valid --pose_config cover2

python write_to_filelist.py --split test1 --pose_config cover1
python write_to_filelist.py --split test1 --pose_config cover2

python combine_cover1_cover2.py
