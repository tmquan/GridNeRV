from typing import Callable, Tuple, Union, List

import torch
import torch.nn as nn

from pytorch3d.ops.utils import eyes
from pytorch3d.structures import Volumes
from pytorch3d.transforms import Transform3d
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.renderer.implicit.raysampling import HeterogeneousRayBundle, RayBundle
from pytorch3d.renderer.implicit.utils import _validate_ray_bundle_variables, ray_bundle_variables_to_ray_points

        
class FrustumSampler(nn.Module):
    """
    A module to backproject a batch of ray densities into volumes `Volumes`
    at 3D points sampled along projection rays.
    """

    def __init__(self, volumes: Volumes, sample_mode: str = "bilinear") -> None:
        """
        Args:
            volumes: An instance of the `Volumes` class representing a
                batch of volumes that are being rendered (dummy).
            sample_mode: Defines the algorithm used to sample the volumetric
                voxel grid. Can be either "bilinear" or "nearest".
        """
        super().__init__()
        if not isinstance(volumes, Volumes):
            raise ValueError("'volumes' have to be an instance of the 'Volumes' class.")
        self._volumes = volumes
        self._sample_mode = sample_mode

    # def inverse_grid_sample(self, 
    #                         rays_features_or_densities, 
    #                         rays_points_local_flat,
    #                         align_corners=False,
    #                         mode="bilinear"):
    #     # TODO: implement the inverse grid sampling operation
    #     raise NotImplementedError
    
    def forward(
        self, 
        ray_bundle: Union[RayBundle, HeterogeneousRayBundle], 
        rays_features_or_densities: torch.Tensor, 
    ) -> torch.Tensor:
        """
        Given an input ray parametrization and ray densities, the forward function backprojects
        `rays_features_or_densities` into `self._volumes` at the respective 3D ray-points.

        Args:
            ray_bundle: A RayBundle or HeterogeneousRayBundle object with the following fields:
                rays_origins_world: A tensor of shape `(minibatch, ..., 3)` denoting the
                    origins of the sampling rays in world coords.
                rays_directions_world: A tensor of shape `(minibatch, ..., 3)`
                    containing the direction vectors of sampling rays in world coords.
                rays_lengths: A tensor of shape `(minibatch, ..., num_points_per_ray)`
                    containing the lengths at which the rays are sampled.
            rays_features_or_densities: A tensor of shape
                `(minibatch, ..., num_points_per_ray, opacity_dim)` containing the
                density vectors sampled from the volume at the locations of
                the ray points.

        Returns:
            volumes_features_or_density: A tensor of shape
                `(minibatch, ..., volume_dim)` containing the
                updated volumes after backprojecting the ray features_or_densities.
        """

        # TODO: implement the forward operation of FrustumSampler using inverse_grid_sample
        # raise NotImplementedError
        # take out the interesting parts of ray_bundle
        rays_origins_world = ray_bundle.origins
        rays_directions_world = ray_bundle.directions
        rays_lengths = ray_bundle.lengths

        # validate the inputs
        _validate_ray_bundle_variables(
            rays_origins_world, rays_directions_world, rays_lengths
        )

        rays_points_world = ray_bundle_variables_to_ray_points(
            rays_origins_world, rays_directions_world, rays_lengths
        )

        ########################
        # 3 sample the frustum #
        ########################
        # reshape to a size which grid_sample likes 
        rays_points_world_flat = rays_points_world.view(
            rays_points_world.shape[0], -1, 1, 1, 3
        )

        # # run the grid sampler on the volumes features or densities
        # rays_densities = torch.nn.functional.grid_sample(
        #     volumes_densities,
        #     rays_points_world_flat,
        #     align_corners=False,
        #     mode=self._sample_mode,
        # )
        # run the grid sampler on the ray densities
        volumes_features_or_density = self.inverse_grid_sample(
            rays_features_or_densities,
            rays_points_local_flat,
            align_corners=False,
            mode=self._sample_mode,
        )

        # # permute the dimensions & reshape densities after sampling
        # rays_densities = rays_densities.permute(0, 2, 3, 4, 1).view(
        #     *rays_points_local.shape[:-1], volumes_densities.shape[1]
        # )
        return volumes_features_or_density
    
    
