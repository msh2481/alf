# Copyright (c) 2024 ALF Contributors. All Rights Reserved.
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
"""SAC with gradient synchronization regularization (Option D2).

Keeps standard SAC Q-networks and adds a regularization loss pushing
dQ/da toward the dynamics-implied gradient:
    ||dQ/da - d(r(s,a) + gamma * V_target(f(s,a)))/da||^2
Applied per Q replica.
"""

import torch

import alf
from alf.algorithms.sac_dyn_algorithm import SacDynAlgorithm, SacDynInfo
from alf.data_structures import StepType
from alf.utils import math_ops


@alf.configurable
class SacGradAlgorithm(SacDynAlgorithm):
    """SAC + gradient synchronization regularization.

    Adds a loss that pushes each Q replica's action-gradient toward the
    gradient implied by the learned dynamics model:
        d/da [r(s,a) + gamma * V_target(f(s,a))]
    One-sided push: model gradient is detached, only Q params are updated.
    """

    def __init__(self,
                 observation_spec,
                 action_spec,
                 grad_sync_weight=1.0,
                 **kwargs):
        super().__init__(observation_spec=observation_spec,
                         action_spec=action_spec,
                         **kwargs)
        self._grad_sync_weight = grad_sync_weight

    def calc_loss(self, info: SacDynInfo):
        sac_dyn_loss = super().calc_loss(info)
        sync_loss = self._calc_grad_sync_loss(info)
        total = math_ops.add_ignore_empty(sac_dyn_loss.loss,
                                          self._grad_sync_weight * sync_loss)
        return sac_dyn_loss._replace(loss=total)

    def _calc_grad_sync_loss(self, info):
        """Compute gradient sync loss between Q-network and model-implied Q.

        info fields are [T, B, ...]. We flatten to [T*B, ...] for network
        calls, then reshape back.
        """
        obs_tb = info.observation  # [T, B, obs_dim]
        T, B = obs_tb.shape[:2]
        obs = obs_tb.reshape(T * B, -1)  # [T*B, obs_dim]

        # Sample fresh action from current policy (detached leaf for autograd)
        action_dist, _ = self._actor_network(obs)
        a = action_dist.rsample()
        a = a.detach().requires_grad_(True)

        # --- Model-implied dQ/da ---
        s_next, r_hat = self.predict_next(obs, a)

        # V_target(s_next): a_next detached, target net forward with s_next
        with torch.no_grad():
            a_next_dist, _ = self._actor_network(s_next)
            a_next = a_next_dist.rsample()
        v_next, _ = self._compute_critics(self._target_critic_networks, s_next,
                                          a_next, ())

        gamma = self._critic_losses[0].gamma
        model_Q = r_hat + gamma * v_next  # [T*B]
        model_dQ_da = torch.autograd.grad(model_Q.sum(), a,
                                          create_graph=False)[0]
        model_dQ_da = model_dQ_da.detach()

        # --- Per-replica actual dQ/da and sync loss ---
        critics, _ = self._critic_networks((obs, a),
                                           state=())  # [T*B, n_replicas]

        total_sync_loss = torch.zeros(T * B, device=obs.device)
        for i in range(self._num_critic_replicas):
            q_i = critics[..., i]
            actual_dQ_da_i = torch.autograd.grad(q_i.sum(),
                                                 a,
                                                 create_graph=True,
                                                 retain_graph=True)[0]
            sync_loss_i = ((actual_dQ_da_i - model_dQ_da)**2).sum(dim=-1)
            total_sync_loss = total_sync_loss + sync_loss_i

        total_sync_loss = total_sync_loss / self._num_critic_replicas

        # Reshape back to [T, B]
        total_sync_loss = total_sync_loss.reshape(T, B)

        # Mask LAST steps
        valid = (info.step_type != StepType.LAST).float()
        return total_sync_loss * valid
