from pytorch3d.implicitron.models.generic_model import GenericModel
import os
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

from pytorch3d.renderer.cameras import FoVPerspectiveCameras, look_at_view_transform
from pytorch3d.renderer.camera_utils import join_cameras_as_batch
from pytorch3d.renderer.implicit.utils import ray_bundle_to_ray_points
from pytorch3d.renderer import NDCMultinomialRaysampler

from pytorch3d.implicitron.models.renderer.base import (
    BaseRenderer,
    EvaluationMode,
    ImplicitFunctionWrapper,
    ImplicitronRayBundle,
    RendererOutput,
    RenderSamplingMode,
)

from diffusers import UNet2DModel
from lightning_fabric.utilities.seed import seed_everything
from lightning import Trainer, LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import StochasticWeightAveraging
from lightning.pytorch.loggers import TensorBoardLogger
from argparse import ArgumentParser
from typing import Optional

from pytorch3d.implicitron.models.renderer.base import EvaluationMode
from pytorch3d.implicitron.models.feature_extractor.resnet_feature_extractor import ResNetFeatureExtractor

from datamodule import UnpairedDataModule
from dvr.renderer import DirectVolumeFrontToBackRenderer
# from dvr.raymarcher import EmissionAbsorptionFrontToBackRaymarcher
# from pixelnerf.renderer import PixelNeRFFrontToBackRenderer

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


def make_cameras(dist: torch.Tensor, elev: torch.Tensor, azim: torch.Tensor):
    assert dist.device == elev.device == azim.device
    _device = dist.device
    R, T = look_at_view_transform(
        dist=dist.float(),
        elev=elev.float() * 90,
        azim=azim.float() * 180
    )
    return FoVPerspectiveCameras(R=R, T=T, fov=45, aspect_ratio=1).to(_device)

CONSTRUCT_MODEL_FROM_CONFIG = True

# class PixelNeRFFrontToBackInverseRenderer(GenericModel):
#     # renderer_class_type="PixelNeRFFrontToBackRenderer"
#     # ---- renderer configs
#     renderer_class_type: str = "PixelNeRFFrontToBackRenderer"
#     renderer: BaseRenderer
#     # implicit_function_class_type="NeuralRadianceFieldImplicitFunction",
#     # render_image_height=256,
#     # render_image_width=256,
#     # loss_weights={"loss_rgb_huber": 0.0, "loss_rgb_mse": 1.0},
#     # tqdm_trigger_threshold=1000,
#     # raysampler_class_type="AdaptiveRaySampler",
#     # raysampler_AdaptiveRaySampler_args = {"scene_extent": 4.0},
#     # image_feature_extractor_class_type="ResNetFeatureExtractor",
#     # image_feature_extractor_ResNetFeatureExtractor_args = {"name": "resnet101", "add_masks": False},
#     # chunk_size_grid=4096,
#     # render_features_dimensions=3
        
class GridNeRVLightningModule(LightningModule):
    def __init__(self, hparams, **kwargs):
        super().__init__()
        self.lr = hparams.lr
        self.stn = hparams.stn
        self.gan = hparams.gan
        self.cam = hparams.cam
        self.shape = hparams.shape
        self.alpha = hparams.alpha
        self.gamma = hparams.gamma
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
            image_width=self.shape, 
            image_height=self.shape, 
            n_pts_per_ray=self.n_pts_per_ray, 
            min_depth=2.0, 
            max_depth=6.0, 
        )
        
        self.inv_renderer = GenericModel(
            implicit_function_class_type="NeuralRadianceFieldImplicitFunction",
            render_image_height=self.shape,
            render_image_width=self.shape,
            loss_weights={"loss_rgb_huber": 0.0, "loss_rgb_mse": 1.0},
            tqdm_trigger_threshold=1000,
            raysampler_class_type="AdaptiveRaySampler",
            raysampler_AdaptiveRaySampler_args = {"scene_extent": 4.0},
            view_pooler_enabled=True,
            # renderer_class_type="PixelNeRFFrontToBackRenderer",
            # renderer_args={},
            renderer_class_type="MultiPassEmissionAbsorptionRenderer",
            renderer_MultiPassEmissionAbsorptionRenderer_args={
                "raymarcher_class_type": "MultiPassEmissionAbsorptionFrontToBackRaymarcher",
            },
            image_feature_extractor_class_type="ResNetFeatureExtractor",
            image_feature_extractor_ResNetFeatureExtractor_args = {"name": "resnet101", "add_masks": False},
            chunk_size_grid=4096,
            render_features_dimensions=3
        )    
        # In this case we can get the equivalent DictConfig cfg object to the way gm is configured as follows
        from omegaconf import OmegaConf
        from pprint import pprint
        cfg = OmegaConf.structured(self.inv_renderer)    
        pprint(cfg)
        # self.inv_renderer = GenericModel(
        #     # # renderer_class=PixelNeRFFrontToBackRenderer,
        #     # # renderer_class_type="PixelNeRFFrontToBackRenderer",
        #     # # renderer_PixelNeRFFrontToBackRenderer_args={
        #     # #     "raymarcher_class_type": "EmissionAbsorptionFrontToBackRaymarcher"
        #     # # },
        #     implicit_function_class_type="NeuralRadianceFieldImplicitFunction",
        #     render_image_height=self.shape,
        #     render_image_width=self.shape,
        #     loss_weights={"loss_rgb_huber": 0.0, "loss_rgb_mse": 1.0},
        #     tqdm_trigger_threshold=1000,
        #     raysampler_class_type="AdaptiveRaySampler",
        #     raysampler_AdaptiveRaySampler_args = {"scene_extent": 4.0},
        #     renderer_class_type="MultiPassEmissionAbsorptionRenderer",
        #     image_feature_extractor_class_type="ResNetFeatureExtractor",
        #     image_feature_extractor_ResNetFeatureExtractor_args = {"name": "resnet101", "add_masks": False},
        #     chunk_size_grid=4096,
        #     render_features_dimensions=1
        # )
        
        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.loss = nn.L1Loss(reduction="mean")
        
    def forward_screen(self, image3d, cameras):   
        return self.fwd_renderer(image3d, cameras) 
                                 
    def _common_step(self, batch, batch_idx, optimizer_idx, stage: Optional[str] = 'evaluation'):
        _device = batch["image3d"].device
        image3d = batch["image3d"]
        image2d = batch["image2d"]
        
        n_views = 1
        src_azim_random = torch.distributions.uniform.Uniform(-0.5, 0.5).sample([n_views]).to(_device) 
        src_elev_random = torch.distributions.uniform.Uniform(-0.5, 0.5).sample([n_views]).to(_device) 
        src_dist_random = 4.0 * torch.ones(n_views, device=_device)
        cam_view_random = make_cameras(src_dist_random, src_elev_random, src_azim_random)
        est_cmpt_random = self.forward_screen(image3d=image3d, cameras=cam_view_random)
        out_cmpt_random = self.inv_renderer.forward(
            image_rgb=est_cmpt_random.repeat(1, 3, 1, 1),
            camera=cam_view_random,  # type: ignore
            evaluation_mode=EvaluationMode.TRAINING if stage == "train" else EvaluationMode.EVALUATION,
            # evaluation_mode=EvaluationMode.EVALUATION, 
            sequence_name=["111"],
        )
        # print(out_cmpt_random)
        
        loss = out_cmpt_random["loss_rgb_mse"]

        if batch_idx==0 and stage=="validation":
            viz2d = torch.cat([
                        torch.cat([image3d[..., self.shape//2, :], 
                                   est_cmpt_random,
                                   image2d, 
                                   out_cmpt_random["images_render"].mean(dim=1, keepdim=True)
                                   ], dim=-2).transpose(2, 3),
                    ], dim=-2)
            grid = torchvision.utils.make_grid(viz2d, normalize=False, scale_each=False, nrow=1, padding=0)
            tensorboard = self.logger.experiment
            tensorboard.add_image(f'{stage}_samples', grid.clamp(0., 1.), self.current_epoch*self.batch_size + batch_idx)
            
        return loss

        
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
        if not self.cam and not self.gan:
            # If neither --cam nor --gan are set, use one optimizer to optimize Unprojector model
            optimizer = torch.optim.AdamW(self.trainer.model.parameters(), lr=self.lr, betas=(0.5, 0.999))
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)
            return [optimizer], [scheduler]
        elif self.cam:
            # If --cam is set, optimize Unprojector and Camera model using 2 optimizers
            optimizer = torch.optim.AdamW(self.trainer.model.parameters(), lr=self.lr, betas=(0.5, 0.999))
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)
            return [optimizer], [scheduler]
        elif self.gan:
            # If --gan is set, optimize Unprojector, Camera as generator, and Discriminator with 2 optimizers
            optimizer_g = torch.optim.AdamW(list(self.inv_renderer.parameters()) 
                                          + list(self.cam_settings.parameters()), lr=self.lr, betas=(0.5, 0.999))
            optimizer_d = torch.optim.AdamW(self.critic_model.parameters(), lr=self.lr * 4, betas=(0.5, 0.999))
            scheduler_g = torch.optim.lr_scheduler.MultiStepLR(optimizer_g, milestones=[100, 200], gamma=0.1)
            scheduler_d = torch.optim.lr_scheduler.MultiStepLR(optimizer_d, milestones=[100, 200], gamma=0.1)
            return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d] 

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--conda_env", type=str, default="Unet")
    parser.add_argument("--notification_email", type=str, default="quantm88@gmail.com")
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--devices", default=None)
    
    # Model arguments
    parser.add_argument("--n_pts_per_ray", type=int, default=512, help="Sampling points per ray")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--shape", type=int, default=256, help="spatial size of the tensor")
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
    parser.add_argument("--amp", action="store_true", help="train with mixed precision or not")
    
    parser.add_argument("--alpha", type=float, default=1., help="vol loss")
    parser.add_argument("--gamma", type=float, default=1., help="img loss")
    parser.add_argument("--theta", type=float, default=1., help="cam loss")
    parser.add_argument("--omega", type=float, default=1., help="cam cond")
    parser.add_argument("--lambda_gp", type=float, default=10, help="gradient penalty")
    parser.add_argument("--clamp_val", type=float, default=.1, help="gradient discrim clamp value")
    
    parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
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
        dirpath=f"{hparams.logsdir}_sh{hparams.sh}_pe{hparams.pe}_cam{int(hparams.cam)}_gan{int(hparams.gan)}_stn{int(hparams.stn)}",
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
        save_dir=f"{hparams.logsdir}_sh{hparams.sh}_pe{hparams.pe}_cam{int(hparams.cam)}_gan{int(hparams.gan)}_stn{int(hparams.stn)}", 
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
            # swa_callback
        ],
        accumulate_grad_batches=4 if not hparams.cam and not hparams.gan else 1,
        strategy="auto", 
        precision=16 if hparams.amp else 32,
        # gradient_clip_val=0.01, 
        # gradient_clip_algorithm="value"
        # stochastic_weight_avg=True,
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
        img_shape=hparams.shape,
        vol_shape=hparams.shape
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
        ckpt_path=hparams.ckpt if hparams.ckpt is not None else None, # "some/path/to/my_checkpoint.ckpt"
    )

    # test

    # serve