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
"""SAC with learned dynamics and reward models.

SacDynAlgorithm extends SacAlgorithm with dynamics f(s,a)->s' and reward
r(s,a)->scalar networks trained via MSE. The models are NOT used for
actor/critic updates — subclasses (SacVAlgorithm, SacGradAlgorithm) override
methods to use them.
"""

import functools

import torch

import alf
from alf.algorithms.sac_algorithm import (SacAlgorithm, SacInfo, SacState,
                                          SacLossInfo, ActionType)
from alf.networks.actor_distribution_networks import ActorDistributionNetwork
from alf.networks.critic_networks import CriticNetwork
from alf.networks.q_networks import QNetwork
from alf.data_structures import (TimeStep, AlgStep, LossInfo, StepType,
                                 namedtuple)
from alf.networks.encoding_networks import EncodingNetwork
from alf.nest.utils import NestConcat
from alf.tensor_specs import TensorSpec
from alf.utils import math_ops

SacDynInfo = namedtuple("SacDynInfo", [
    *SacInfo._fields,
    "observation",
],
                        default_value=())

SacDynLossInfo = namedtuple(
    "SacDynLossInfo",
    (
        *SacLossInfo._fields,
        "model_loss_mean",
        "model_loss_min",
        "model_loss_max",
        "model_loss_nonfinite_frac",
    ),
    default_value=())


def _finite_stats(x: torch.Tensor):
    """Return (mean, min, max, nonfinite_frac) as scalar tensors."""
    if not isinstance(x, torch.Tensor):
        return (), (), (), ()
    xf = x.detach()
    finite = torch.isfinite(xf)
    nonfinite_frac = (1.0 - finite.to(torch.float32).mean())
    if finite.any():
        vals = xf[finite]
        return vals.mean(), vals.min(), vals.max(), nonfinite_frac

    nan = torch.tensor(float("nan"), device=xf.device, dtype=xf.dtype)
    return nan, nan, nan, nonfinite_frac


@alf.configurable
class SacDynAlgorithm(SacAlgorithm):
    """SAC with learned dynamics and reward models.

    Adds dynamics (predicting state deltas) and reward networks trained via MSE.
    By default these models are not used for actor/critic updates — subclasses
    override methods to use them.
    """

    def __init__(self,
                 observation_spec,
                 action_spec,
                 dynamics_hidden=(256, 256),
                 reward_hidden=(256, 256),
                 model_loss_weight=1.0,
                 reward_spec=TensorSpec(()),
                 actor_network_cls=ActorDistributionNetwork,
                 critic_network_cls=CriticNetwork,
                 q_network_cls=QNetwork,
                 repr_alg_ctor=None,
                 reward_weights=None,
                 train_eps_greedy=1.0,
                 epsilon_greedy=None,
                 use_entropy_reward=True,
                 use_mc_return=False,
                 normalize_entropy_reward=False,
                 calculate_priority=False,
                 num_critic_replicas=2,
                 env=None,
                 config=None,
                 critic_loss_ctor=None,
                 target_entropy=None,
                 prior_actor_ctor=None,
                 target_kld_per_dim=3.,
                 initial_log_alpha=0.0,
                 max_log_alpha=None,
                 target_update_tau=0.05,
                 target_update_period=1,
                 parameter_reset_period=-1,
                 dqda_clipping=None,
                 actor_optimizer=None,
                 critic_optimizer=None,
                 alpha_optimizer=None,
                 num_actor_updates=None,
                 checkpoint=None,
                 debug_summaries=False,
                 reproduce_locomotion=False,
                 name="SacDynAlgorithm"):
        """
        Args:
            dynamics_hidden: FC layer sizes for dynamics network.
            reward_hidden: FC layer sizes for reward network.
            model_loss_weight: weight for dynamics + reward MSE loss.
            All other args: forwarded to SacAlgorithm.
        """
        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            actor_network_cls=actor_network_cls,
            critic_network_cls=critic_network_cls,
            q_network_cls=q_network_cls,
            repr_alg_ctor=repr_alg_ctor,
            reward_weights=reward_weights,
            train_eps_greedy=train_eps_greedy,
            epsilon_greedy=epsilon_greedy,
            use_entropy_reward=use_entropy_reward,
            use_mc_return=use_mc_return,
            normalize_entropy_reward=normalize_entropy_reward,
            calculate_priority=calculate_priority,
            num_critic_replicas=num_critic_replicas,
            env=env,
            config=config,
            critic_loss_ctor=critic_loss_ctor,
            target_entropy=target_entropy,
            prior_actor_ctor=prior_actor_ctor,
            target_kld_per_dim=target_kld_per_dim,
            initial_log_alpha=initial_log_alpha,
            max_log_alpha=max_log_alpha,
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            parameter_reset_period=parameter_reset_period,
            dqda_clipping=dqda_clipping,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            alpha_optimizer=alpha_optimizer,
            num_actor_updates=num_actor_updates,
            checkpoint=checkpoint,
            debug_summaries=debug_summaries,
            reproduce_locomotion=reproduce_locomotion,
            name=name,
        )

        self._model_loss_weight = model_loss_weight

        # Flatten obs spec to get the shape for delta prediction.
        # For simple vector obs this is just the spec itself.
        obs_spec = observation_spec
        act_spec = action_spec

        _dyn_last_init = functools.partial(torch.nn.init.uniform_,
                                           a=-0.03,
                                           b=0.03)
        _rew_last_init = functools.partial(torch.nn.init.normal_, std=1e-3)

        self._dynamics_net = EncodingNetwork(
            input_tensor_spec=(obs_spec, act_spec),
            preprocessing_combiner=NestConcat(),
            fc_layer_params=dynamics_hidden,
            last_layer_size=obs_spec.shape[0],
            last_activation=math_ops.identity,
            last_kernel_initializer=_dyn_last_init)

        self._reward_net = EncodingNetwork(
            input_tensor_spec=(obs_spec, act_spec),
            preprocessing_combiner=NestConcat(),
            fc_layer_params=reward_hidden,
            last_layer_size=1,
            last_activation=math_ops.identity,
            last_kernel_initializer=_rew_last_init)

        from absl import logging as _logging
        _logging.info(
            f"[SacDynAlgorithm] dynamics_hidden={dynamics_hidden!r} reward_hidden={reward_hidden!r}"
        )
        _logging.info(f"[SacDynAlgorithm] dynamics_net:\n{self._dynamics_net}")
        _logging.info(f"[SacDynAlgorithm] reward_net:\n{self._reward_net}")

    def predict_next(self, obs, action):
        """Predict next state and reward using learned models.

        Args:
            obs: observation tensor [..., obs_dim]
            action: action tensor [..., act_dim]
        Returns:
            s_next_pred: predicted next state [..., obs_dim]
            r_pred: predicted reward [...]
        """
        delta, _ = self._dynamics_net((obs, action))
        s_next_pred = obs + delta
        r_pred, _ = self._reward_net((obs, action))
        r_pred = r_pred.squeeze(-1)
        return s_next_pred, r_pred

    def train_step(self, inputs: TimeStep, state: SacState,
                   rollout_info: SacInfo):
        alg_step = super().train_step(inputs, state, rollout_info)
        # Wrap SacInfo into SacDynInfo with observation attached.
        sac_info = alg_step.info
        dyn_info = SacDynInfo(**{
            f: getattr(sac_info, f)
            for f in SacInfo._fields
        },
                              observation=inputs.observation)
        return alg_step._replace(info=dyn_info)

    def calc_loss(self, info: SacDynInfo):
        sac_loss = super().calc_loss(info)
        model_loss = self._calc_model_loss(info)
        total = math_ops.add_ignore_empty(sac_loss.loss,
                                          self._model_loss_weight * model_loss)
        (model_loss_mean, model_loss_min, model_loss_max,
         model_loss_nonfinite_frac) = _finite_stats(model_loss)

        extra = sac_loss.extra
        if extra == ():
            extra = SacDynLossInfo(model_loss_mean=model_loss_mean,
                                   model_loss_min=model_loss_min,
                                   model_loss_max=model_loss_max,
                                   model_loss_nonfinite_frac=
                                   model_loss_nonfinite_frac)
        else:
            extra_fields = {
                field: getattr(extra, field, ())
                for field in SacDynLossInfo._fields
            }
            extra_fields.update(model_loss_mean=model_loss_mean,
                                model_loss_min=model_loss_min,
                                model_loss_max=model_loss_max,
                                model_loss_nonfinite_frac=
                                model_loss_nonfinite_frac)
            extra = SacDynLossInfo(**extra_fields)

        return sac_loss._replace(loss=total, extra=extra)

    def _calc_model_loss(self, info):
        """Compute dynamics + reward MSE loss using time-shifted observations.

        Batch is time-major [T, B, ...]. We use obs[:-1] as current and
        obs[1:] as next, same pattern as one_step_discounted_return in td_loss.
        """
        obs_all = info.observation  # [T, B, obs_dim]
        obs = obs_all[:-1]  # [T-1, B, obs_dim]
        next_obs = obs_all[1:]  # [T-1, B, obs_dim]
        action = info.action[:-1]  # [T-1, B, act_dim]
        reward = info.reward[1:]  # [T-1, B] — reward at next step

        # Dynamics loss
        delta_pred, _ = self._dynamics_net((obs, action))
        s_next_pred = obs + delta_pred
        dyn_loss = ((s_next_pred - next_obs.detach())**2).mean(dim=-1)

        # Reward loss
        r_pred, _ = self._reward_net((obs, action))
        r_pred = r_pred.squeeze(-1)
        rew_loss = (r_pred - reward.detach())**2

        # Mask out transitions crossing episode boundaries:
        # invalid if current step is LAST (no meaningful next)
        step_type = info.step_type[:-1]  # [T-1, B]
        valid = (step_type != StepType.LAST).float()
        loss = (dyn_loss + rew_loss) * valid

        # Pad back to [T, B] so it aligns with SAC's loss shape.
        # Append zeros for the last time step.
        pad = torch.zeros_like(loss[:1])
        loss = torch.cat([loss, pad], dim=0)
        return loss
