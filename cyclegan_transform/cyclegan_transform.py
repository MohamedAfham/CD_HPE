
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from cyclegan_transform.models.test_model import TestModel
import torch

class get_cyclegan_opt:
    def __init__(self, checkpoint_dir= './cyclegan_transform/checkpoints', name= 'InbedPose_CyleGAN'):
        self.aspect_ratio=1.0
        self.batch_size=1
        self.checkpoints_dir=checkpoint_dir
        self.crop_size=256
        #self.dataroot='/content/In-Bed-Human-Pose-Estimation(VIP-CUP)/train/00001/IR/uncover'
        self.dataset_mode='single'
        self.direction='AtoB'
        self.display_id=-1
        self.display_winsize=256
        self.epoch='latest'
        self.eval=False
        self.gpu_ids=[0]
        self.init_gain=0.02
        self.init_type='normal'
        self.input_nc=3
        self.isTrain=False
        self.load_iter=0
        self.load_size=256
        self.max_dataset_size=np.inf
        self.model='test'
        self.model_suffix=''
        self.n_layers_D=3
        self.name=name
        self.ndf=64
        self.netD='basic'
        self.netG='resnet_9blocks'
        self.ngf=64
        self.no_dropout=True
        self.no_flip=True
        self.norm='instance'
        self.num_test=50
        self.num_threads=0
        self.output_nc=3
        self.phase='test'
        self.preprocess='resize_and_crop'
        #self.results_dir='/content/dummy'
        self.serial_batches=True
        self.suffix=''
        self.verbose=False

class cyclegan_transform:
    def __init__(self, cyclegan_opt):
        self.opt= cyclegan_opt
        
        model = TestModel(self.opt)
        model.load_networks('latest')
        model.eval()

        self.model= model

    def __call__(self, img): # torch.tensor: (n_channels, n, n), range: [0, 1]
        cycle_covered = self.uncover2cover(img)
        return cycle_covered

    def uncover2cover(self, img): # torch tensor,shape: (3, a, b): range: [0, 1]
        pil_img= Image.fromarray((img.permute(1,2,0).numpy()*255).astype('uint8'))
        img = self.pre_transform_for_cyclegan(self.opt)(pil_img) # do transformation needed for cycleGAN
        converted_img = self.model.netG(img.unsqueeze(dim=0)).detach() #torch.Size([1, 3, 256, 256]) -> range: [-1, 1]
        converted_img= converted_img[0] #shape: 3, 256, 256 , range: [-1, +1], dtype: float

        converted_img= (converted_img+1.)/2. # torch tensor,shape: (3, 256, 256): range: [0, 1]

        return converted_img.cpu()

    def pre_transform_for_cyclegan(self, opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
        transform_list = []
        if grayscale:
            transform_list.append(transforms.Grayscale(1))
        if 'resize' in opt.preprocess:
            osize = [opt.load_size, opt.load_size]
            transform_list.append(transforms.Resize(osize, method))
        elif 'scale_width' in opt.preprocess:
            transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, opt.crop_size, method)))

        if 'crop' in opt.preprocess:
            if params is None:
                transform_list.append(transforms.RandomCrop(opt.crop_size))
            else:
                transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

        if opt.preprocess == 'none':
            transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

        if not opt.no_flip:
            if params is None:
                transform_list.append(transforms.RandomHorizontalFlip())
            elif params['flip']:
                transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

        if convert:
            transform_list += [transforms.ToTensor()]
            if grayscale:
                transform_list += [transforms.Normalize((0.5,), (0.5,))]
            else:
                transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        return transforms.Compose(transform_list)
