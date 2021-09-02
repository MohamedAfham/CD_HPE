import os
import glob
import argparse
import scipy.io
import json

def parse_option():

    parser = argparse.ArgumentParser('argument for creating json files')

    parser.add_argument('--split', type=str, default = None, choices=['train', 'valid', 'test1'])
    parser.add_argument('--pose_config', type=str, default = None, choices=['uncover', 'cover1', 'cover2'])
    parser.add_argument('--save_dir', type=str, default='', help='path to save json file')
    
    args = parser.parse_args()
    
    return args

def sort_by_subject(x):
  return int(x[-5:])

def sort_by_img_name(x):
  return int(x[-10:-4])


args = parse_option()
DATA_DIR = "/home/mohamedafham/notebooks/projects/CD_HPE/data/SLP_VIPCup"

if args.split == 'train':
    if args.pose_config == 'uncover':
        ids = range(1, 31)
    elif args.pose_config == 'cover1':
        ids = range(31, 56)
    elif args.pose_config == 'cover2':
        ids = range(56, 81)
elif args.split == 'valid':
    if args.pose_config == 'cover1':
        ids = range(31, 86)
    elif args.pose_config == 'cover2':
        ids = range(86, 91)
elif args.split == 'test1':
    ids = [1, 2, 3, 4, 91, 92, 93, 94, 95, 96]

dir_list = []
for data_dir in glob.glob(os.path.join(DATA_DIR, args.split, '*')):
    if int(data_dir[-5:]) in ids:
      dir_list.append(data_dir)
dir_list_sorted = sorted(dir_list, key=sort_by_subject)

def get_filelist(train_uncover_dirs, args):
    img_dict_list = []
    for dir_path in dir_list_sorted:
        
        img_dirs = sorted(glob.glob(os.path.join(dir_path, "IR", args.pose_config, '*.png')), key = sort_by_img_name)
        
        if args.split == 'train':
            if args.pose_config == 'cover1' or args.pose_config == 'cover2':
                joints_dir = None
            else:
                joints_dir =  f'{dir_path}/joints_gt_IR.mat'              # (<x, y, is_occluded>, num_joints, num_images) 
                joints_all = scipy.io.loadmat(joints_dir)['joints_gt']
        elif args.split == 'test1':
            joints_dir = None
        else:
            joints_dir =  f'{dir_path}/joints_gt_IR.mat'              # (<x, y, is_occluded>, num_joints, num_images) 
            joints_all = scipy.io.loadmat(joints_dir)['joints_gt']

            
        for img_dir in img_dirs:             
            img_number = img_dir[-10:-4]
            if joints_dir != None:
                joints = joints_all[:,:,int(img_number)-1]

                new_x, new_y = joints[0] - 1 , joints[1] - 1
                joints[0] = new_x
                joints[1] = new_y

                joints = joints.T

                joints = joints.tolist()
            else:
                joints = None
                
            img_dict = {'file_name':img_dir , 'key_points':joints}
            img_dict_list.append(img_dict)
            
    return img_dict_list  

def write_to_json(img_dict_list, args):
    if args.split == 'test1':
        with open(os.path.join(args.save_dir,f'{args.split}.json'), 'w') as f:
            json.dump(img_dict_list,f)
    else:
        with open(os.path.join(args.save_dir,f'{args.split}_{args.pose_config}.json'), 'w') as f:
            json.dump(img_dict_list,f)
        
write_to_json(get_filelist(dir_list_sorted, args), args)
    
    