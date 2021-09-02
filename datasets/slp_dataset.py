from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import json


class SLP(Dataset):
    def __init__(self, data_file, transform, args):
        with open(data_file, 'r') as f:
          self.data = json.load(f)

        self.transform = transform
        self.args = args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]['file_name']
        image = Image.open(img_path)
        image = self.transform(image)
        keypoints = np.array(self.data[idx]['key_points'])
        joints_vis = np.absolute(keypoints[:, 2] - 1)

        joints_3d = np.zeros((14, 3), dtype=np.float)
        joints_3d_vis = np.zeros((14,  3), dtype=np.float)

        joints_3d[:, 0:2] = keypoints[:, 0:2]
        joints_3d_vis[:, 0] = joints_vis[:]
        joints_3d_vis[:, 1] = joints_vis[:]

        target, target_weight = self.generate_target(joints_3d, joints_3d_vis, self.args)

        return image, target, target_weight

    
    def generate_target(self, joints, joints_vis, args):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''

        num_joints = 14; sigma = args.sigma; heatmap_size = np.array(args.heatmap_size)
        image_size = np.array([120, 160])
        target_weight = np.ones((num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        target = np.zeros((num_joints,
                            heatmap_size[1],
                            heatmap_size[0]),
                          dtype=np.float32)

        tmp_size = sigma * 3

        for joint_id in range(num_joints):
            feat_stride = image_size / heatmap_size
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                    or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                target_weight[joint_id] = 0
                continue

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_weight


class SLPWeak(Dataset):
    def __init__(self, data_file, transform=None):
        with open(data_file, 'r') as f:
          self.data = json.load(f)

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]['file_name']
        image = Image.open(img_path)
        image = self.transform(image)
        
        return image