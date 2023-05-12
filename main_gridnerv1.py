import os
import math
import warnings
warnings.filterwarnings("ignore")
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
torch.set_float32_matmul_precision('medium')
torch.cuda.empty_cache()
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.multiprocessing.set_sharing_strategy('file_system')
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import kornia
from kornia.augmentation import RandomAffine

from pytorch3d.renderer.implicit.utils import ray_bundle_to_ray_points, _validate_ray_bundle_variables, ray_bundle_variables_to_ray_points
from pytorch3d.renderer.cameras import FoVOrthographicCameras, FoVPerspectiveCameras, look_at_view_transform
from pytorch3d.renderer import NDCMultinomialRaysampler

from diffusers import UNet2DModel
from lightning_fabric.utilities.seed import seed_everything
from lightning import Trainer, LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import StochasticWeightAveraging
from lightning.pytorch.loggers import TensorBoardLogger
from argparse import ArgumentParser
from typing import Optional, Union, List
from monai.networks.nets import Unet, EfficientNetBN, Regressor
from monai.networks.layers.factories import Norm

from datamodule import UnpairedDataModule
from dvr.renderer import DirectVolumeFrontToBackRenderer, normalized, standardized

backbones = {
    "efficientnet-b0": (16, 24, 40, 112, 320),
    "efficientnet-b1": (16, 24, 40, 112, 320),
    "efficientnet-b2": (16, 24, 48, 120, 352),
    "efficientnet-b3": (24, 32, 48, 136, 384),
    "efficientnet-b4": (24, 32, 56, 160, 448),
    "efficientnet-b5": (24, 40, 64, 176, 512),
    "efficientnet-b6": (32, 40, 72, 200, 576),
    "efficientnet-b7": (32, 48, 80, 224, 640),
    "efficientnet-b8": (32, 56, 88, 248, 704),
    "efficientnet-l2": (72, 104, 176, 480, 1376),
}

class GridNeRVFrontToBackFrustumFeaturer(nn.Module):
    def __init__(self, in_channels=1, shape=256, out_channels=1, backbone="efficientnet-b7"):
        super().__init__()
        assert backbone in backbones.keys()
        self.model = EfficientNetBN(
            model_name=backbone, #(24, 32, 56, 160, 448)
            spatial_dims=2,
            in_channels=in_channels,
            num_classes=out_channels,
            # pretrained=True, 
            # adv_prop=True,
        )
        # self.model = Regressor(
        #     in_shape=(in_channels, shape, shape), 
        #     out_shape=(out_channels), 
        #     channels=backbones[backbone], 
        #     strides=(2, 2, 2, 2, 2), 
        #     norm=Norm.BATCH,
            # dropout=0.2,
        # )

    def forward(self, figures):
        camfeat = self.model.forward(figures)
        return camfeat

class GridNeRVFrontToBackInverseRenderer(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, img_shape=400, vol_shape=256, n_pts_per_ray=256, sh=0, pe=8, backbone="efficientnet-b7"):
        super().__init__()
        self.sh = sh
        self.pe = pe
        self.img_shape = img_shape
        self.vol_shape = vol_shape
        self.n_pts_per_ray = n_pts_per_ray
        assert backbone in backbones.keys()
        if self.pe>0:
            # Generate grid
            zs = torch.linspace(-1, 1, steps=self.vol_shape)
            ys = torch.linspace(-1, 1, steps=self.vol_shape)
            xs = torch.linspace(-1, 1, steps=self.vol_shape)
            z, y, x = torch.meshgrid(zs, ys, xs)
            zyx = torch.stack([z, y, x], dim=-1) # torch.Size([100, 100, 100, 3])
            from nerfstudio.field_components import encodings
            num_frequencies = self.pe
            min_freq_exp = 0
            max_freq_exp = 8
            encoder = encodings.NeRFEncoding(
                in_dim=self.pe, 
                num_frequencies=num_frequencies, 
                min_freq_exp=min_freq_exp, 
                max_freq_exp=max_freq_exp
            )
            pebasis = encoder(zyx.view(-1, 3))
            pebasis = pebasis.view(self.vol_shape, self.vol_shape, self.vol_shape, -1).permute(3, 0, 1, 2)
            self.register_buffer('pebasis', pebasis)

        if self.sh > 0:
            # Generate grid
            zs = torch.linspace(-1, 1, steps=self.vol_shape)
            ys = torch.linspace(-1, 1, steps=self.vol_shape)
            xs = torch.linspace(-1, 1, steps=self.vol_shape)
            z, y, x = torch.meshgrid(zs, ys, xs)
            zyx = torch.stack([z, y, x], dim=-1) # torch.Size([100, 100, 100, 3])
            
            from nerfstudio.field_components import encodings
            encoder = encodings.SHEncoding(self.sh)
            assert out_channels == self.sh**2 if self.sh>0 else 1
            shbasis = encoder(zyx.view(-1, 3))
            shbasis = shbasis.view(self.vol_shape, self.vol_shape, self.vol_shape, -1).permute(3, 0, 1, 2)
            self.register_buffer('shbasis', shbasis)
            
        self.clarity_net = UNet2DModel(
            sample_size=self.img_shape,  
            in_channels=1,  
            out_channels=self.n_pts_per_ray,
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=backbones[backbone],  # More channels -> more parameters
            norm_num_groups=8,
            down_block_types=(
                "DownBlock2D",  
                "DownBlock2D",  
                "DownBlock2D",
                "AttnDownBlock2D",  
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",    
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",    
            ),
            class_embed_type="timestep",
        )

        self.density_net = nn.Sequential(
            Unet(
                spatial_dims=3,
                in_channels=1+(2*3*self.pe),
                out_channels=1,
                channels=backbones[backbone],
                strides=(2, 2, 2, 2, 2),
                num_res_units=2,
                kernel_size=3,
                up_kernel_size=3,
                act=("LeakyReLU", {"inplace": True}),
                norm=Norm.BATCH,
                dropout=0.2,
            ),
        )

        self.mixture_net = nn.Sequential(
            Unet(
                spatial_dims=3,
                in_channels=2+(2*3*self.pe),
                out_channels=1,
                channels=backbones[backbone],
                strides=(2, 2, 2, 2, 2),
                num_res_units=2,
                kernel_size=3,
                up_kernel_size=3,
                act=("LeakyReLU", {"inplace": True}),
                norm=Norm.BATCH,
                dropout=0.2,
            ),
        )

        self.refiner_net = nn.Sequential(
            Unet(
                spatial_dims=3,
                in_channels=3+(2*3*self.pe),
                out_channels=out_channels,
                channels=backbones[backbone],
                strides=(2, 2, 2, 2, 2),
                num_res_units=2,
                kernel_size=3,
                up_kernel_size=3,
                act=("LeakyReLU", {"inplace": True}),
                norm=Norm.BATCH,
                dropout=0.2,
            ), 
        )

        self.raysampler = NDCMultinomialRaysampler(  
            image_width=self.img_shape,
            image_height=self.img_shape,
            n_pts_per_ray=self.n_pts_per_ray,  
            min_depth=8.0,
            max_depth=4.0,
        )        

        
    def forward(self, figures, azim, elev, n_views=2):
        clarity = self.clarity_net(figures, azim*900, elev*1800)[0].view(-1, 1, self.n_pts_per_ray, self.img_shape, self.img_shape)
        
        # Process (resample) the clarity from ray views to ndc
        _device = figures.device
        B = figures.shape[0]
        dist = 6.0 * torch.ones(B, device=_device)
        cameras = make_cameras(dist, elev, azim)
        
        # ray_bundle = self.raysampler.forward(cameras=cameras, n_pts_per_ray=self.n_pts_per_ray)
        # ray_points = ray_bundle_to_ray_points(ray_bundle).view(B, -1, 3) 
        
        # # Generate camera intrinsics and extrinsics
        # itransform = cameras.get_ndc_camera_transform().inverse()
        # ndc_coords = itransform.transform_points(ray_points)
        # # ray_values = torch.add(figures.unsqueeze(1), clarity)
        # ray_values = clarity
        # ndc_values = F.grid_sample(
        #     ray_values,
        #     ndc_coords.view(-1, self.vol_shape, self.vol_shape, self.vol_shape, 3),
        #     mode='bilinear', 
        #     padding_mode='zeros', 
        #     align_corners=True
        # )
        
        # Generate a grid of ndc coordinates that covers the entire ndc volume
        ndc_z = torch.linspace(-1, 1, steps=self.vol_shape, device=_device)
        ndc_y = torch.linspace(-1, 1, steps=self.vol_shape, device=_device)
        ndc_x = torch.linspace(-1, 1, steps=self.vol_shape, device=_device)
        ndc_coords = torch.stack(torch.meshgrid(ndc_x, ndc_y, ndc_z), dim=-1).view(-1, 3).unsqueeze(0).repeat(B, 1, 1)    
        v2w_coords = cameras.get_world_to_view_transform().inverse().transform_points(ndc_coords) # view to world
        # w2c_coords = cameras.transform_points(v2w_coords) # world to ndc
        # w2v_coords = cameras.get_world_to_view_transform().transform_points(ndc_coords) # world to view
        
        ray_values = torch.add(figures.unsqueeze(1), clarity)
        # ray_values = clarity
        ndc_values = F.grid_sample(
            ray_values,
            v2w_coords.view(B, self.vol_shape, self.vol_shape, self.vol_shape, 3),
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=True
        )
        
        # ray_bundle = self.raysampler.forward(cameras=cameras, n_pts_per_ray=self.n_pts_per_ray)
        # ray_points = ray_bundle_to_ray_points(ray_bundle).view(B, -1, 3) 
        # # Generate camera intrinsics and extrinsics
        # v2w_coords = cameras.get_world_to_view_transform().inverse().transform_points(ray_points) # view to world
        # w2c_coords = cameras.transform_points(v2w_coords) # world to ndc
        # ray_values = torch.add(figures.unsqueeze(1), clarity)
        # ndc_values = F.grid_sample(
        #     ray_values,
        #     w2c_coords.view(-1, self.vol_shape, self.vol_shape, self.vol_shape, 3),
        #     mode='bilinear', 
        #     padding_mode='zeros', 
        #     align_corners=True
        # )
        
        # Multiview can stack along batch dimension, last dimension is for X-ray
        clarity_ct, clarity_xr = torch.split(ndc_values, split_size_or_sections=n_views, dim=0)
        clarity_ct = clarity_ct.mean(dim=0, keepdim=True)
        clarity = torch.cat([clarity_ct, clarity_xr])

        if self.pe > 0:
            density = self.density_net(torch.cat([self.pebasis.repeat(clarity.shape[0], 1, 1, 1, 1), clarity], dim=1))
            mixture = self.mixture_net(torch.cat([self.pebasis.repeat(clarity.shape[0], 1, 1, 1, 1), clarity, density], dim=1))
            shcoeff = self.refiner_net(torch.cat([self.pebasis.repeat(clarity.shape[0], 1, 1, 1, 1), clarity, density, mixture], dim=1))
        else:
            density = self.density_net(torch.cat([clarity], dim=1))
            density = torch.add(density, clarity)
            mixture = self.mixture_net(torch.cat([clarity, density], dim=1))
            mixture = torch.add(mixture, clarity)
            shcoeff = self.refiner_net(torch.cat([clarity, density, mixture], dim=1))
            shcoeff = torch.add(shcoeff, clarity)
        if self.sh > 0:
            shcomps = torch.einsum('abcde,bcde->abcde', shcoeff, self.shbasis)
        else:
            shcomps = shcoeff 
        volumes = shcomps
        volumes_ct, volumes_xr = torch.split(volumes, 1)
        volumes_ct = volumes_ct.repeat(n_views, 1, 1, 1, 1)
        volumes = torch.cat([volumes_ct, volumes_xr])
        return volumes

def make_cameras(dist: torch.Tensor, elev: torch.Tensor, azim: torch.Tensor, seed=None):
    assert dist.device == elev.device == azim.device
    _device = dist.device
    R, T = look_at_view_transform(
        dist=dist.float(), 
        elev=elev.float() * 90, 
        azim=azim.float() * 180
    )
    return FoVPerspectiveCameras(R=R, T=T, fov=20, znear=4.0, zfar=8.0).to(_device)

def torch_distributions_uniform_or_zeros(shape=[1, 1], device=torch.device('cpu')):
    rng = torch.randint(low=0, high=10, size=(1, 1))
    if rng>0:
        return torch.distributions.uniform.Uniform(-1.0, 1.0).sample(shape).to(device)
    else:
        return torch.zeros(shape, device=device)

def init_weights(net, init_type='kaiming', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)
    # print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>
    
class GridNeRVLightningModule(LightningModule):
    def __init__(self, hparams, **kwargs):
        super().__init__()
        self.lr = hparams.lr
        self.stn = hparams.stn
        self.gan = hparams.gan
        self.cam = hparams.cam
        self.sup = hparams.sup
        self.ckpt = hparams.ckpt
        self.strict = hparams.strict
        
        self.img_shape = hparams.img_shape
        self.vol_shape = hparams.vol_shape
        self.alpha = hparams.alpha
        self.gamma = hparams.gamma
        self.delta = hparams.delta
        self.theta = hparams.theta
        self.omega = hparams.omega
        self.lambda_gp = hparams.lambda_gp
        self.clamp_val = hparams.clamp_val
       
        self.logsdir = hparams.logsdir
       
        self.sh = hparams.sh
        self.pe = hparams.pe
        
        self.n_pts_per_ray = hparams.n_pts_per_ray
        self.weight_decay = hparams.weight_decay
        self.batch_size = hparams.batch_size
        self.backbone = hparams.backbone
        self.devices = hparams.devices
        
        self.save_hyperparameters()

        self.fwd_renderer = DirectVolumeFrontToBackRenderer(
            image_width=self.img_shape, 
            image_height=self.img_shape, 
            n_pts_per_ray=self.n_pts_per_ray, 
            min_depth=4.0, 
            max_depth=8.0, 
        )
        
        self.inv_renderer = GridNeRVFrontToBackInverseRenderer(
            in_channels=2, 
            out_channels=self.sh**2 if self.sh>0 else 1, 
            vol_shape=self.vol_shape, 
            img_shape=self.img_shape, 
            sh=self.sh, 
            pe=self.pe,
            backbone=self.backbone,
        )
        # init_weights(self.inv_renderer)
        if self.ckpt:
            # load the checkpoint
            checkpoint = torch.load(self.ckpt, map_location=torch.device('cpu'))["state_dict"]
            # create a new state dict with the keys that exist in both the checkpoint and the model
            state_dict = {k: v for k, v in checkpoint.items() if k in self.state_dict()}
            # load the state dict into the model, ignoring non-existent keys
            self.load_state_dict(state_dict, strict=self.strict)
            
        if self.stn:
            self.stn_modifier = GridNeRVFrontToBackFrustumFeaturer(
                in_channels=1, 
                out_channels=6, 
                backbone=self.backbone,
            )
            self.stn_modifier.model._fc.weight.data.zero_()
            self.stn_modifier.model._fc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
            # affine_transform = torchvision.transforms.RandomAffine(degrees=(30, 30), translate=(0.1, 0.1), scale=(0.75, 0.75))
            self.affine_transform = RandomAffine(shear=(-10, 10, -10, 10),
                                                scale=(0.75, 1.25, 0.75, 1.25),
                                                degrees=(-10, 10), 
                                                translate=(0.1, 0.1), 
                                                p=1.0)
        if self.cam:
            self.cam_settings = GridNeRVFrontToBackFrustumFeaturer(
                in_channels=1, 
                out_channels=2, 
                backbone=self.backbone,
            )
            init_weights(self.cam_settings)
            torch.nn.init.trunc_normal_(self.cam_settings.model._fc.weight.data, mean=0.0, std=0.05, a=-0.05, b=0.05)
            torch.nn.init.trunc_normal_(self.cam_settings.model._fc.bias.data, mean=0.0, std=0.05, a=-0.05, b=0.05)
            # self.cam_settings.model._fc.weight.data.random_()
            # self.cam_settings.model._fc.bias.data.random_()
            # self.cam_settings.model._fc.weight.data.copy_(torch.tensor([0.2, 0.2], dtype=torch.float))
            # # self.cam_settings.model._fc.bias.data.copy_(torch.tensor([0.2, 0.2], dtype=torch.float))

        self.train_step_outputs = []
        self.validation_step_outputs = []
        # self.loss = nn.SmoothL1Loss(reduction="mean", beta=0.1)
        self.loss = nn.L1Loss(reduction="mean")
        # self.pips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True)
        
    # Spatial transformer network forward function
    def forward_affine(self, x):
        theta = self.stn_modifier(x * 2.0 - 1.0)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        xs = F.grid_sample(x, grid)
        return xs
    
    def forward_screen(self, image3d, cameras):   
        return self.fwd_renderer(image3d, cameras) 

    def forward_volume(self, image2d, azim, elev, n_views=2):      
        return self.inv_renderer(image2d * 2.0 - 1.0, azim.squeeze(1), elev.squeeze(1), n_views) #* 0.5 + 0.5

    def forward_camera(self, image2d):
        return self.cam_settings(image2d * 2.0 - 1.0)

    def forward_critic(self, image2d):
        return self.critic_model(image2d * 2.0 - 1.0)
    
    def _common_step(self, batch, batch_idx, optimizer_idx, stage: Optional[str] = 'evaluation'):
        _device = batch["image3d"].device
        image3d = batch["image3d"]
        image2d = batch["image2d"]
            
        # Construct the random cameras
        src_azim_random = torch.distributions.uniform.Uniform(-1.0, 1.0).sample([self.batch_size]).to(_device)
        src_elev_random = torch.distributions.uniform.Uniform(-1.0, 1.0).sample([self.batch_size]).to(_device)
        src_dist_random = 6.0 * torch.ones(self.batch_size, device=_device)
        camera_random = make_cameras(src_dist_random, src_elev_random, src_azim_random)
        
        src_azim_locked = torch.distributions.uniform.Uniform(-1.0, 1.0).sample([self.batch_size]).to(_device)
        src_elev_locked = torch.distributions.uniform.Uniform(-1.0, 1.0).sample([self.batch_size]).to(_device)
        src_dist_locked = 6.0 * torch.ones(self.batch_size, device=_device)
        camera_locked = make_cameras(src_dist_locked, src_elev_locked, src_azim_locked)

        est_figure_ct_random = self.forward_screen(image3d=image3d, cameras=camera_random)
        est_figure_ct_locked = self.forward_screen(image3d=image3d, cameras=camera_locked)
        
        # XR pathway
        if self.stn: 
            src_figure_xr_hidden = self.forward_affine(image2d).detach()      
        else:
            src_figure_xr_hidden = image2d

        est_dist_random = 6.0 * torch.ones(self.batch_size, device=_device)
        est_dist_locked = 6.0 * torch.ones(self.batch_size, device=_device)
        est_dist_hidden = 6.0 * torch.ones(self.batch_size, device=_device)

        if self.cam:        
            # Reconstruct the cameras
            est_feat_random, \
            est_feat_locked, \
            est_feat_hidden = torch.split(
                self.forward_camera(
                    image2d=torch.cat([est_figure_ct_random, est_figure_ct_locked, src_figure_xr_hidden])
                ), self.batch_size
            )

            est_azim_random, est_elev_random = torch.split(est_feat_random, 1, dim=1)
            est_azim_locked, est_elev_locked = torch.split(est_feat_locked, 1, dim=1)
            est_azim_hidden, est_elev_hidden = torch.split(est_feat_hidden, 1, dim=1)
        else:
            est_azim_random, est_elev_random = src_azim_random, src_elev_random 
            est_azim_locked, est_elev_locked = src_azim_locked, src_elev_locked 
            est_azim_hidden, est_elev_hidden = torch.zeros(self.batch_size, device=_device), torch.zeros(self.batch_size, device=_device)

        if self.sup:
            camera_random = make_cameras(src_dist_random, src_elev_random, src_azim_random)
            camera_locked = make_cameras(src_dist_locked, src_elev_locked, src_azim_locked)
            camera_hidden = make_cameras(est_dist_hidden, est_elev_hidden, est_azim_hidden)
        else:
            camera_random = make_cameras(est_dist_random, est_elev_random, est_azim_random)
            camera_locked = make_cameras(est_dist_locked, est_elev_locked, est_azim_locked)
            camera_hidden = make_cameras(est_dist_hidden, est_elev_hidden, est_azim_hidden)

        if self.stn:
            est_figure_ct_hidden = self.forward_screen(image3d=image3d, cameras=camera_hidden)
            est_figure_ct_affine = self.affine_transform(est_figure_ct_hidden).detach()
            est_figure_ct_warped = self.forward_affine(est_figure_ct_affine)
    
        cam_view = [self.batch_size, 1]     
        if self.sup:  
            est_volume_ct_random, \
            est_volume_ct_locked, \
            est_volume_xr_hidden = torch.split(
                self.forward_volume(
                    image2d=torch.cat([est_figure_ct_random, est_figure_ct_locked, src_figure_xr_hidden]),
                    azim=torch.cat([src_azim_random.view(cam_view), src_azim_locked.view(cam_view), est_azim_hidden.view(cam_view)]),
                    elev=torch.cat([src_elev_random.view(cam_view), src_elev_locked.view(cam_view), est_elev_hidden.view(cam_view)]),
                    n_views=2,
                ), self.batch_size
            )  
        else:
            est_volume_ct_random, \
            est_volume_ct_locked, \
            est_volume_xr_hidden = torch.split(
                self.forward_volume(
                    image2d=torch.cat([est_figure_ct_random, est_figure_ct_locked, src_figure_xr_hidden]),
                    azim=torch.cat([est_azim_random.view(cam_view), est_azim_locked.view(cam_view), est_azim_hidden.view(cam_view)]),
                    elev=torch.cat([est_elev_random.view(cam_view), est_elev_locked.view(cam_view), est_elev_hidden.view(cam_view)]),
                    n_views=2,
                ), self.batch_size
            )  
            
        # Reconstruct the appropriate XR
        rec_figure_ct_random = self.forward_screen(image3d=est_volume_ct_random, cameras=camera_random)
        rec_figure_ct_locked = self.forward_screen(image3d=est_volume_ct_locked, cameras=camera_locked)
        est_figure_xr_hidden = self.forward_screen(image3d=est_volume_xr_hidden, cameras=camera_hidden)

        # Perform Post activation like DVGO      
        est_volume_ct_random = est_volume_ct_random.sum(dim=1, keepdim=True)
        est_volume_ct_locked = est_volume_ct_locked.sum(dim=1, keepdim=True)
        est_volume_xr_hidden = est_volume_xr_hidden.sum(dim=1, keepdim=True)

        # Compute the loss
        # Per-pixel_loss
        im2d_loss_ct_random = self.loss(est_figure_ct_random, rec_figure_ct_random) 
        im2d_loss_ct_locked = self.loss(est_figure_ct_locked, rec_figure_ct_locked) 
        im2d_loss_xr_hidden = self.loss(src_figure_xr_hidden, est_figure_xr_hidden) 
        
        im3d_loss_ct_random = self.loss(image3d, est_volume_ct_random) #+ self.loss(image3d, mid_volume_ct_random) 
        im3d_loss_ct_locked = self.loss(image3d, est_volume_ct_locked) #+ self.loss(image3d, mid_volume_ct_locked) 

        if self.stn:
            im2d_loss_ct_hidden = self.loss(est_figure_ct_hidden, est_figure_ct_warped) 
            im2d_loss_ct = im2d_loss_ct_random + im2d_loss_ct_locked + im2d_loss_ct_hidden   
        else:
            im2d_loss_ct = im2d_loss_ct_random + im2d_loss_ct_locked 
            
        im2d_loss_xr = im2d_loss_xr_hidden 
        im2d_loss = im2d_loss_ct + im2d_loss_xr
        
        im3d_loss_ct = im3d_loss_ct_random + im3d_loss_ct_locked
        im3d_loss = im3d_loss_ct
        
        # pips_loss_ct_random = self.pips(est_figure_ct_random.repeat(1,3,1,1), rec_figure_ct_random.repeat(1,3,1,1)) 
        # pips_loss_ct_locked = self.pips(est_figure_ct_locked.repeat(1,3,1,1), rec_figure_ct_locked.repeat(1,3,1,1)) 
        # pips_loss_xr_hidden = self.pips(src_figure_xr_hidden.repeat(1,3,1,1), est_figure_xr_hidden.repeat(1,3,1,1)) 
        # pips_loss = pips_loss_ct_random + pips_loss_ct_locked + pips_loss_xr_hidden
        
        self.log(f'{stage}_im2d_loss', im2d_loss, on_step=(stage=='train'), prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)
        self.log(f'{stage}_im3d_loss', im3d_loss, on_step=(stage=='train'), prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)
        # self.log(f'{stage}_pips_loss', pips_loss, on_step=(stage=='train'), prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)

        p_loss = self.gamma*im2d_loss + self.alpha*im3d_loss #+ self.delta*pips_loss
        
        if self.cam:
            view_loss_ct_random = self.loss(torch.cat([src_azim_random, src_elev_random]), 
                                            torch.cat([est_azim_random, est_elev_random]))
            view_loss_ct_locked = self.loss(torch.cat([src_azim_locked, src_elev_locked]), 
                                            torch.cat([est_azim_locked, est_elev_locked]))
            view_loss_ct = view_loss_ct_random + view_loss_ct_locked

            view_cond_xr = self.loss(torch.cat([torch.zeros_like(est_azim_hidden), torch.zeros_like(est_elev_hidden)]), 
                                     torch.cat([est_azim_hidden, est_elev_hidden]))
                         
            # view_cond_xr = self.loss(est_azim_hidden, torch.zeros_like(est_azim_hidden)) \
            #              + self.loss(est_elev_hidden, torch.zeros_like(est_elev_hidden))
    
            view_loss = view_loss_ct 
            view_cond = view_cond_xr

            self.log(f'{stage}_view_loss', view_loss, on_step=(stage=='train'), prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)
            c_loss = self.gamma*im2d_loss + self.theta*view_loss + self.omega*view_cond

        if batch_idx==0:
            viz2d = torch.cat([
                        torch.cat([est_figure_ct_random,
                                   est_figure_ct_locked,
                                   image2d, 
                                   ], dim=-2).transpose(2, 3),
                        torch.cat([rec_figure_ct_random,
                                   rec_figure_ct_locked,
                                   est_figure_xr_hidden,
                                   ], dim=-2).transpose(2, 3),
                        
                    ], dim=-2)
            viz3d = torch.cat([
                        torch.cat([image3d[..., self.vol_shape//2, :], 
                                   est_volume_ct_locked[..., self.vol_shape//2, :],
                                   est_volume_xr_hidden[..., self.vol_shape//2, :],
                                   ], dim=-2).transpose(2, 3),
                    ], dim=-2)
            if self.stn:
                viz2d = torch.cat([
                            torch.cat([image3d[..., self.shape//2, :], 
                                    est_figure_ct_random,
                                    est_figure_ct_locked,
                                    est_figure_ct_affine
                                    ], dim=-2).transpose(2, 3),
                            torch.cat([est_volume_ct_locked[..., self.shape//2, :],
                                    rec_figure_ct_random,
                                    rec_figure_ct_locked,
                                    est_figure_ct_warped
                                    ], dim=-2).transpose(2, 3),
                            torch.cat([image2d, 
                                    src_figure_xr_hidden,
                                    est_volume_xr_hidden[..., self.shape//2, :],
                                    est_figure_xr_hidden,
                                    ], dim=-2).transpose(2, 3),
                        ], dim=-2)
            grid2d = torchvision.utils.make_grid(viz2d, normalize=False, scale_each=False, nrow=1, padding=0)
            grid3d = torchvision.utils.make_grid(viz3d, normalize=False, scale_each=False, nrow=1, padding=0)
            tensorboard = self.logger.experiment
            if self.img_shape==self.vol_shape:
                grid = torch.cat([grid2d, grid3d], dim=-2)
                tensorboard.add_image(f'{stage}_samples', grid.clamp(0., 1.), self.current_epoch*self.batch_size + batch_idx)    
            else:
                tensorboard.add_image(f'{stage}_2d_samples', grid2d.clamp(0., 1.), self.current_epoch*self.batch_size + batch_idx)
                tensorboard.add_image(f'{stage}_3d_samples', grid3d.clamp(0., 1.), self.current_epoch*self.batch_size + batch_idx)
        

        if not self.cam and not self.gan:
            return p_loss
        elif self.cam:
            return p_loss + c_loss
        elif self.gan:
            optimizer_g, optimizer_d = self.optimizers()
            # generator loss
            self.toggle_optimizer(optimizer_g)
            fake_images = torch.cat([rec_figure_ct_random, rec_figure_ct_locked, est_figure_xr_hidden])
            fake_scores = self.forward_critic(fake_images)
            g_loss = torch.mean(-fake_scores)
            loss = p_loss + g_loss + c_loss
            self.log(f'{stage}_g_loss', g_loss, on_step=(stage=='train'), prog_bar=False, logger=True, sync_dist=True, batch_size=self.batch_size)
            self.manual_backward(loss)
            optimizer_g.step()
            optimizer_g.zero_grad()
            self.untoggle_optimizer(optimizer_g)
            
            # discriminator
            for p in self.critic_model.parameters():
                p.data.clamp_(-self.clamp_val, self.clamp_val)
            self.toggle_optimizer(optimizer_d)
            real_images = torch.cat([est_figure_ct_random, est_figure_ct_locked, src_figure_xr_hidden])
            real_scores = self.forward_critic(real_images)
            fake_images = torch.cat([rec_figure_ct_random, rec_figure_ct_locked, est_figure_xr_hidden])
            fake_scores = self.forward_critic(fake_images.detach())
            d_loss = torch.mean(-real_scores) + torch.mean(+fake_scores)
            loss = d_loss 
            self.log(f'{stage}_d_loss', d_loss, on_step=(stage=='train'), prog_bar=False, logger=True, sync_dist=True, batch_size=self.batch_size)
            self.manual_backward(loss)
            optimizer_d.step()
            optimizer_d.zero_grad()
            self.untoggle_optimizer(optimizer_d)
            return p_loss + c_loss

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        loss = self._common_step(batch, batch_idx, optimizer_idx, stage='train')
        self.train_step_outputs.append(loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, optimizer_idx=-1, stage='validation')
        self.validation_step_outputs.append(loss)
        return loss

    def on_train_epoch_end(self):
        loss = torch.stack(self.train_step_outputs).mean()
        self.log(f'train_loss_epoch', loss, on_step=False, prog_bar=True, logger=True, sync_dist=True)
        self.train_step_outputs.clear()  # free memory
        
    def on_validation_epoch_end(self):
        loss = torch.stack(self.validation_step_outputs).mean()
        self.log(f'validation_loss_epoch', loss, on_step=False, prog_bar=True, logger=True, sync_dist=True)
        self.validation_step_outputs.clear()  # free memory
        
    def configure_optimizers(self):
        if self.gan:
            # If --gan is set, optimize Unprojector, Camera as generator, and Discriminator with 2 optimizers
            optimizer_g = torch.optim.AdamW(list(self.inv_renderer.parameters()) 
                                          + list(self.cam_settings.parameters()), lr=self.lr, betas=(0.9, 0.999))
            optimizer_d = torch.optim.AdamW(self.critic_model.parameters(), lr=self.lr * 4, betas=(0.9, 0.999))
            scheduler_g = torch.optim.lr_scheduler.MultiStepLR(optimizer_g, milestones=[100, 200], gamma=0.1)
            scheduler_d = torch.optim.lr_scheduler.MultiStepLR(optimizer_d, milestones=[100, 200], gamma=0.1)
            return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d] 
        else:
            # 
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.999))
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)
            return [optimizer], [scheduler]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--conda_env", type=str, default="Unet")
    parser.add_argument("--notification_email", type=str, default="quantm88@gmail.com")
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--devices", default=None)
    
    # Model arguments
    parser.add_argument("--n_pts_per_ray", type=int, default=400, help="Sampling points per ray")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--img_shape", type=int, default=256, help="spatial size of the tensor")
    parser.add_argument("--vol_shape", type=int, default=256, help="spatial size of the tensor")
    parser.add_argument("--epochs", type=int, default=301, help="number of epochs")
    parser.add_argument("--train_samples", type=int, default=1000, help="training samples")
    parser.add_argument("--val_samples", type=int, default=400, help="validation samples")
    parser.add_argument("--test_samples", type=int, default=400, help="test samples")
    parser.add_argument("--st", type=int, default=1, help="with spatial transformer network")
    parser.add_argument("--sh", type=int, default=0, help="degree of spherical harmonic (2, 3)")
    parser.add_argument("--pe", type=int, default=0, help="positional encoding (0 - 8)")
    
    parser.add_argument("--stn", action="store_true", help="whether to train with spatial transformer")
    parser.add_argument("--gan", action="store_true", help="whether to train with GAN")
    parser.add_argument("--cam", action="store_true", help="train cam locked or hidden")
    parser.add_argument("--sup", action="store_true", help="train cam ct or not")
    parser.add_argument("--amp", action="store_true", help="train with mixed precision or not")
    parser.add_argument("--strict", action="store_true", help="checkpoint loading")
    
    parser.add_argument("--alpha", type=float, default=1., help="vol loss")
    parser.add_argument("--gamma", type=float, default=1., help="img loss")
    parser.add_argument("--delta", type=float, default=1., help="vgg loss")
    parser.add_argument("--theta", type=float, default=1., help="cam loss")
    parser.add_argument("--omega", type=float, default=1., help="cam cond")
    parser.add_argument("--lambda_gp", type=float, default=10, help="gradient penalty")
    parser.add_argument("--clamp_val", type=float, default=.1, help="gradient discrim clamp value")
    
    parser.add_argument("--lr", type=float, default=1e-3, help="adam: learning rate")
    parser.add_argument("--ckpt", type=str, default=None, help="path to checkpoint")
    parser.add_argument("--logsdir", type=str, default='logs', help="logging directory")
    parser.add_argument("--datadir", type=str, default='data', help="data directory")
    parser.add_argument("--backbone", type=str, default='efficientnet-b7', help="Backbone for network")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    
    # parser = Trainer.add_argparse_args(parser)

    # Collect the hyper parameters
    hparams = parser.parse_args()

    # Seed the application
    seed_everything(42)

    # Callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{hparams.logsdir}_sh{hparams.sh}_pe{hparams.pe}_cam{int(hparams.cam)}_gan{int(hparams.gan)}_sup{int(hparams.sup)}",
        # filename='epoch={epoch}-validation_loss={validation_loss_epoch:.2f}',
        monitor="validation_loss_epoch",
        auto_insert_metric_name=True, 
        save_top_k=-1,
        save_last=True,
        every_n_epochs=10,
    )
    lr_callback = LearningRateMonitor(logging_interval='step')

    # Logger
    tensorboard_logger = TensorBoardLogger(
        save_dir=f"{hparams.logsdir}_sh{hparams.sh}_pe{hparams.pe}_cam{int(hparams.cam)}_sup{int(hparams.sup)}", 
        log_graph=True
    )
    swa_callback = StochasticWeightAveraging(swa_lrs=1e-2)
    # Init model with callbacks
    trainer = Trainer(
        accelerator=hparams.accelerator,
        devices=hparams.devices,
        max_epochs=hparams.epochs,
        logger=[tensorboard_logger],
        callbacks=[
            lr_callback,
            checkpoint_callback,
            swa_callback if not hparams.gan else None,
        ],
        accumulate_grad_batches=4 if not hparams.gan else 1,
        strategy="auto", 
        precision=16 if hparams.amp else 32,
        # gradient_clip_val=0.01, 
        # gradient_clip_algorithm="value"
        # stochastic_weight_avg=True if not hparams.gan else False,
        # deterministic=False,
        profiler="advanced"
    )

    # Create data module
    train_image3d_folders = [
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/NSCLC/processed/train/images'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-0'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-1'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-2'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-3'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-4'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Imagenglab/processed/train/images'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MELA2022/raw/train/images'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MELA2022/raw/val/images'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/AMOS2022/raw/train/images'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/AMOS2022/raw/val/images'),

        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2019/raw/train/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2020/raw/train/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2019/raw/val/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2020/raw/val/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2019/raw/test/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2020/raw/test/rawdata/'),

        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/UWSpine/processed/train/images'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/UWSpine/processed/test/images/'),
    ]

    train_label3d_folders = [
    ]

    train_image2d_folders = [
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/JSRT/processed/images/'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/ChinaSet/processed/images/'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Montgomery/processed/images/'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/train/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/test/images/'),

        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62020/20200501/raw/images'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62021/20211101/raw/images'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/train/images/'),
        # # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/test/images/'),
    ]

    train_label2d_folders = [
    ]

    val_image3d_folders = [
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/NSCLC/processed/train/images'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-0'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-1'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-2'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-3'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-4'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Imagenglab/processed/train/images'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MELA2022/raw/train/images'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MELA2022/raw/val/images'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/AMOS2022/raw/train/images'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/AMOS2022/raw/val/images'),

        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2019/raw/train/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2020/raw/train/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2019/raw/val/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2020/raw/val/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2019/raw/test/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2020/raw/test/rawdata/'),

        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/UWSpine/processed/train/images'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/UWSpine/processed/test/images/'),
    ]

    val_image2d_folders = [
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/JSRT/processed/images/'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/ChinaSet/processed/images/'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Montgomery/processed/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/train/images/'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/test/images/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62020/20200501/raw/images'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62021/20211101/raw/images'),
        # # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/train/images/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/test/images/'),
    ]

    test_image3d_folders = val_image3d_folders
    test_image2d_folders = val_image2d_folders

    datamodule = UnpairedDataModule(
        train_image3d_folders=train_image3d_folders,
        train_image2d_folders=train_image2d_folders,
        val_image3d_folders=val_image3d_folders,
        val_image2d_folders=val_image2d_folders,
        test_image3d_folders=test_image3d_folders,
        test_image2d_folders=test_image2d_folders,
        train_samples=hparams.train_samples,
        val_samples=hparams.val_samples,
        test_samples=hparams.test_samples,
        batch_size=hparams.batch_size,
        img_shape=hparams.img_shape,
        vol_shape=hparams.vol_shape
    )
    datamodule.setup()

    ####### Test camera mu and bandwidth ########
    # test_random_uniform_cameras(hparams, datamodule)
    #############################################

    model = GridNeRVLightningModule(
        hparams=hparams
    )
    # model = model.load_from_checkpoint(hparams.ckpt, strict=False) if hparams.ckpt is not None else model
    # compiled_model = torch.compile(model, mode="reduce-overhead")
    trainer.fit(
        model,
        # compiled_model,
        train_dataloaders=datamodule.train_dataloader(), 
        val_dataloaders=datamodule.val_dataloader(),
        # datamodule=datamodule,
        ckpt_path=hparams.ckpt if hparams.ckpt is not None and hparams.strict else None, # "some/path/to/my_checkpoint.ckpt"
    )

    # test

    # serve