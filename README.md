# In-bed Multi-modal Human Dataset
This dataset recorded in-bed human subjects from multiple modality sensors including RGB, IR for pose estimation purpose. 
The data is collected in both a home setting and hospital setting environment named 'danaLab' and 'simLab' with 102 and 7 subjects respectively. 
For each setting, subjects are requested to give 15 poses as wish in 3 general categories as supine, left lying and right lying. For each pose,data is collected from 3 cover conditions as uncover, cover1 and cover2.  

Each subject is named with a numbered folder, with structure
`[subjNumber]
--IR
	-- uncover
	-- cover1
	-- cover2
-- RGB 
	...
-- joints_gt_IR.mat
-- joints_gt_RGB.mat
-- align_PTr_IR.npy
-- align_PTr_RGB.npy
`
Under each modality, there will be 3 cover conditions with corresponding same poses to reflect our physical parameter tuning strategy proposed in our paper. 
`joints_gt_<modality>.mat` is the pose label file
`align_PTr_<modality>.npy` saved the homography transformation matrix to reference frame. To get transformation between RGB and IR, we can simply use `inv(H_RGB) * H_IR`.   


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


## Contact
Sarah Ostadabbas, 
email: ostadabbas@ece.neu.edu
Shuanjun Liu,
email: shuliu@ece.neu.edu
