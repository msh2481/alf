# Copyright (c) 2026 Horizon Robotics and ALF Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for parameter-space perturbations."""

from __future__ import annotations

import torch


@torch.no_grad()
def perturb_params_l2_sphere(params, alpha: float):
    """Perturb parameters on an L2 sphere using an MCMC-style random walk.

    Treats the entire parameter set as one vector.
    """
    if alpha is None or alpha == 0:
        return

    from torch.nn.utils import parameters_to_vector, vector_to_parameters

    params = list(params)
    if not params:
        return

    vec = parameters_to_vector(params)
    old_norm = vec.norm(p=2)
    if old_norm == 0:
        return

    noise = torch.randn_like(vec) * (alpha * old_norm)
    new_vec = vec + noise
    new_norm = new_vec.norm(p=2)
    if new_norm == 0:
        return

    new_vec = new_vec * (old_norm / new_norm)
    vector_to_parameters(new_vec, params)


@torch.no_grad()
def perturb_module_params_l2_sphere_per_layer(module: torch.nn.Module,
                                              alpha: float):
    """Perturb each layer (module) independently on its own L2 sphere.

    A \"layer\" here means a `torch.nn.Module` whose own parameters are returned by
    `module.parameters(recurse=False)`. This avoids coupling between layers via a
    single global renormalization.
    """
    if alpha is None or alpha == 0:
        return

    from torch.nn.utils import parameters_to_vector, vector_to_parameters

    for m in module.modules():
        params = list(m.parameters(recurse=False))
        if not params:
            continue

        vec = parameters_to_vector(params)
        old_norm = vec.norm(p=2)
        if old_norm == 0:
            continue

        noise = torch.randn_like(vec) * (alpha * old_norm)
        new_vec = vec + noise
        new_norm = new_vec.norm(p=2)
        if new_norm == 0:
            continue

        new_vec = new_vec * (old_norm / new_norm)
        vector_to_parameters(new_vec, params)
