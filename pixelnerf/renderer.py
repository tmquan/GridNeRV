
from typing import List

import torch
from pytorch3d.implicitron.tools.config import (
    Configurable,
    ReplaceableBase,
    expand_args_fields,
    get_default_args,
    registry,
    run_auto_creation,
)

from pytorch3d.implicitron.models.renderer.base import (
    BaseRenderer,
    EvaluationMode,
    ImplicitFunctionWrapper,
    ImplicitronRayBundle,
    RendererOutput,
    RenderSamplingMode,
)

from pytorch3d.implicitron.models.renderer.ray_point_refiner import RayPointRefiner
from pytorch3d.implicitron.models.renderer.raymarcher import RaymarcherBase

from pytorch3d.implicitron.models.renderer.multipass_ea import MultiPassEmissionAbsorptionRenderer

from dvr.raymarcher import EmissionAbsorptionFrontToBackRaymarcher

@registry.register
class PixelNeRFFrontToBackRenderer(MultiPassEmissionAbsorptionRenderer, Configurable, ReplaceableBase):
    raymarcher_class_type: str = "EmissionAbsorptionFrontToBackRaymarcher"
    raymarcher: EmissionAbsorptionFrontToBackRaymarcher
    
    n_pts_per_ray_fine_training: int = 64
    n_pts_per_ray_fine_evaluation: int = 64
    stratified_sampling_coarse_training: bool = True
    stratified_sampling_coarse_evaluation: bool = False
    append_coarse_samples_to_fine: bool = True
    density_noise_std_train: float = 0.0
    return_weights: bool = False

    def __post_init__(self):
        self._refiners = {
            EvaluationMode.TRAINING: RayPointRefiner(
                n_pts_per_ray=self.n_pts_per_ray_fine_training,
                random_sampling=self.stratified_sampling_coarse_training,
                add_input_samples=self.append_coarse_samples_to_fine,
            ),
            EvaluationMode.EVALUATION: RayPointRefiner(
                n_pts_per_ray=self.n_pts_per_ray_fine_evaluation,
                random_sampling=self.stratified_sampling_coarse_evaluation,
                add_input_samples=self.append_coarse_samples_to_fine,
            ),
        }
        run_auto_creation(self)