import torch
from pytorch3d.renderer import EmissionAbsorptionRaymarcher
from pytorch3d.renderer.implicit.raymarching import (
    _check_density_bounds,
    _check_raymarcher_inputs,
    _shifted_cumprod,
)


class EmissionAbsorptionFrontToBackRaymarcher(EmissionAbsorptionRaymarcher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        rays_densities: torch.Tensor,
        rays_features: torch.Tensor,
        eps: float = 1e-10,
        **kwargs,
    ) -> torch.Tensor:
        _check_raymarcher_inputs(
            rays_densities,
            rays_features,
            None,
            z_can_be_none=True,
            features_can_be_none=False,
            density_1d=True,
        )
        _check_density_bounds(rays_densities)
        rays_densities = rays_densities[..., 0]
        # print(rays_densities.shape)
        # absorption = _shifted_cumprod(
        #     (1.0 + eps) - rays_densities, shift=self.surface_thickness
        # )
        # weights = rays_densities * absorption
        # features = (weights[..., None] * rays_features).sum(dim=-2)
        # opacities = 1.0 - torch.prod(1.0 - rays_densities, dim=-1, keepdim=True)
        absorption = _shifted_cumprod(
            (1.0 + eps) - rays_densities.flip(dims=(-1,)), shift=-self.surface_thickness
        ).flip(dims=(-1,))  # Reverse the direction of the absorption to match X-ray detector
        weights = rays_densities * absorption
        features = (weights[..., None] * rays_features).sum(dim=-2)
        opacities = 1.0 - \
            torch.prod(1.0 - rays_densities, dim=-1, keepdim=True)
        return torch.cat((features, opacities), dim=-1)

from pytorch3d.implicitron.tools.config import (
    Configurable,
    ReplaceableBase,
    expand_args_fields,
    get_default_args,
    registry,
    run_auto_creation,
)


from pytorch3d.implicitron.models.renderer.raymarcher import EmissionAbsorptionRaymarcher as MultiPassEmissionAbsorptionRaymarcher
from typing import Any, Callable, Dict, Tuple

import torch
from pytorch3d.implicitron.models.renderer.base import RendererOutput
from pytorch3d.implicitron.tools.config import registry, ReplaceableBase
from pytorch3d.renderer.implicit.raymarching import _check_raymarcher_inputs


_TTensor = torch.Tensor

@registry.register
class MultiPassEmissionAbsorptionFrontToBackRaymarcher(MultiPassEmissionAbsorptionRaymarcher):
    def forward(
        self,
        rays_densities: torch.Tensor,
        rays_features: torch.Tensor,
        aux: Dict[str, Any],
        ray_lengths: torch.Tensor,
        density_noise_std: float = 0.0,
        **kwargs,
    ) -> RendererOutput:
        """
        Args:
            rays_densities: Per-ray density values represented with a tensor
                of shape `(..., n_points_per_ray, 1)`.
            rays_features: Per-ray feature values represented with a tensor
                of shape `(..., n_points_per_ray, feature_dim)`.
            aux: a dictionary with extra information.
            ray_lengths: Per-ray depth values represented with a tensor
                of shape `(..., n_points_per_ray, feature_dim)`.
            density_noise_std: the magnitude of the noise added to densities.
        Returns:
            features: A tensor of shape `(..., feature_dim)` containing
                the rendered features for each ray.
            depth: A tensor of shape `(..., 1)` containing estimated depth.
            opacities: A tensor of shape `(..., 1)` containing rendered opacities.
            weights: A tensor of shape `(..., n_points_per_ray)` containing
                the ray-specific non-negative opacity weights. In general, they
                don't sum to 1 but do not overcome it, i.e.
                `(weights.sum(dim=-1) <= 1.0).all()` holds.
        """
        _check_raymarcher_inputs(
            rays_densities,
            rays_features,
            ray_lengths,
            z_can_be_none=True,
            features_can_be_none=False,
            density_1d=True,
        )

        ray_lengths_diffs = ray_lengths[..., 1:] - ray_lengths[..., :-1]
        if self.replicate_last_interval:
            last_interval = ray_lengths_diffs[..., -1:]
        else:
            last_interval = torch.full_like(
                ray_lengths[..., :1], self.background_opacity
            )
        deltas = torch.cat((ray_lengths_diffs, last_interval), dim=-1)

        rays_densities = rays_densities[..., 0]

        if density_noise_std > 0.0:
            noise: _TTensor = torch.randn_like(rays_densities).mul(density_noise_std)
            rays_densities = rays_densities + noise
        if self.density_relu:
            rays_densities = torch.relu(rays_densities)

        weighted_densities = deltas * rays_densities.flip(dims=(-1,))
        capped_densities = self._capping_function(weighted_densities)  # pyre-ignore: 29

        rays_opacities = self._capping_function(  # pyre-ignore: 29
            torch.cumsum(weighted_densities, dim=-1)
        )
        opacities = rays_opacities[..., -1:]
        absorption_shifted = (-rays_opacities + 1.0).roll(
            self.surface_thickness, dims=-1
        )
        absorption_shifted[..., : self.surface_thickness] = 1.0

        weights = self._weight_function(  # pyre-ignore: 29
            capped_densities, absorption_shifted.flip(dims=(-1,))
        )
        
        # absorption = _shifted_cumprod(
        #     (1.0 + eps) - rays_densities.flip(dims=(-1,)), shift=-self.surface_thickness
        # ).flip(dims=(-1,))  # Reverse the direction of the absorption to match X-ray detector
        # weights = rays_densities * absorption
        # features = (weights[..., None] * rays_features).sum(dim=-2)
        # opacities = 1.0 - \
        #     torch.prod(1.0 - rays_densities, dim=-1, keepdim=True)
        
        features = (weights[..., None] * rays_features).sum(dim=-2)
        depth = (weights * ray_lengths)[..., None].sum(dim=-2)

        alpha = opacities if self.blend_output else 1
        if self._bg_color.shape[-1] not in [1, features.shape[-1]]:
            raise ValueError("Wrong number of background color channels.")
        features = alpha * features + (1 - opacities) * self._bg_color

        return RendererOutput(
            features=features,
            depths=depth,
            masks=opacities,
            weights=weights,
            aux=aux,
        )