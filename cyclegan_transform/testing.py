from cyclegan_transform import cyclegan_transform, get_cyclegan_opt
import cv2
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image

### 1. for single_image

np_uncovered= cv2.cvtColor(plt.imread('/content/In-Bed-Human-Pose-Estimation(VIP-CUP)/train/00001/IR/uncover/image_000001.png'), cv2.COLOR_GRAY2RGB) # shape: (160, 120, 3)
torch_uncovered= torch.tensor(np_uncovered).permute(2, 0, 1)  # shape: (3, 160, 120)

out = cyclegan_transform(cyclegan_opt= get_cyclegan_opt())(torch_uncovered)
plt.imshow(out.permute(1,2,0).cpu())
plt.savefig('../cyclegan_transform_single_img.png')
plt.show()


### 2. Compose with other transforms

my_transform= transforms.Compose([transforms.ToTensor(), cyclegan_transform(cyclegan_opt= get_cyclegan_opt())])

x= (255*cv2.cvtColor(plt.imread('/content/In-Bed-Human-Pose-Estimation(VIP-CUP)/train/00001/IR/uncover/image_000001.png'), cv2.COLOR_GRAY2RGB)).astype('uint8') # shape: (160, 120, 3)

x= Image.fromarray(x)
y= my_transform(x)
plt.imshow(y.permute(1,2,0).cpu())
plt.savefig('../cyclegan_transform_compose.png')
plt.show()