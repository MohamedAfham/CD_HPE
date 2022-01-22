# Towards Accurate Cross Domain In-Bed Human Pose Estimation

## Citation
If you find our work or this repository useful, please consider giving a star :star: and citation.
```bibtex
@article{CD_HPE,
      title={Towards Accurate Cross-Domain In-Bed Human Pose Estimation}, 
      author={Mohamed Afham and Udith Haputhanthri and Jathurshan Pradeepkumar and Mithunjha Anandakumar and Ashwin De Silva and Chamira Edussooriya},
      year={2021},
      journal={arXiv preprint arXiv:2110.03578},
}
```

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

Create a folder named data/SLP_VIPCup.
```
mkdir data
mkdir data/SLP_VIPCup
```

Download dataset from codalab. Unzip the data inside data/SLP_VIPCup/ by running:

```
source download_data.sh
```

The directory should look like

```
CD_HPE/ 
    data/
        SLP_VIPCup/ 
            train/ 
            test1/ 
            val/
```

Run the following to create the required `.json` files for the dataset

```
cd filelists
source create_filelist.sh
cd ..
```

## Train

### For Learning Stage 01 (Standard Supervision)

```
python train_supervised.py --adam --use_target_weight --model stacked_hg --print_freq 50 --batch_size 3 --wandb_run Name/of/wandb/run
```

### For Learning Stage 02 (Knowledge Distillation)

```
python train_distillation.py --adam --best_path path/to/best/model --model stacked_hg --print_freq 50 --batch_size 4 --lr 1e-4 --wandb_run Name/of/wandb/run
```

Kindly refer to [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MohamedAfham/CD_HPE/blob/master/Reproduce_NFP_Undercover.ipynb) for reproducing our results.
