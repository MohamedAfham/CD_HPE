# Cross Domain - Human Pose Estimation

VIPCUP 2021 - Team NFP_Undercover

## Pose Definition
We follow the Leeds Sports Pose Dataset pose definition with 14 joints labeling, namely, 

`Right ankle
Right knee
Right hip
Left hip
Left knee
Left ankle
Right wrist
Right elbow
Right shoulder
Left shoulder
Left elbow
Left wrist
Neck
Head top`

The lable matrix `joints_gt_<modality>.mat` has the format  `<x,y,if_occluded>` x n_joints x n_subjects 

## Dataset

Download dataset from codalab. Create a folder named data/SLP_VIPCup. Unzip the data inside data/SLP_VIPCup/
The directory should look like
`CD_HPE/
    data/
        SLP_VIPCup/
            train/
            test1/
            val/
`
In `filelists/write_to_filelist.py` change `DATA_DIR` based on where the dataset is stored. Run `python filelists/write_to_filelist.py` which will create `.json` dictionary files for the dataset.

## Train

Run `python main.py --adam --use_target_weight --model stacked_hg --print_freq 50 --batch_size 10 --wandb_run Name/of/wandb/run` to start the training.
