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
"""SAC with V-network replacing Q-network (Option A).

Q(s,a) := r(s,a) + gamma * V(f(s,a)) — full gradient synchronization.
The Q-function is never a free-standing network; it is defined entirely
through the learned dynamics model and a V(s) network.

Reference: SAC-SVG (Amos et al., 2021) arXiv:2008.12775
"""

import torch

import alf
from alf.algorithms.sac_algorithm import (SacCriticInfo, SacCriticState,
                                          ActionType)
from alf.algorithms.sac_dyn_algorithm import SacDynAlgorithm, SacDynInfo
from alf.data_structures import LossInfo, StepType
from alf.nest import nest
from alf.networks.actor_distribution_networks import ActorDistributionNetwork
from alf.networks.critic_networks import CriticNetwork
from alf.networks.q_networks import QNetwork
from alf.networks.value_networks import ValueNetwork
from alf.tensor_specs import TensorSpec
from alf.utils import losses, math_ops, dist_utils


@alf.configurable
class SacVAlgorithm(SacDynAlgorithm):
    """SAC with V-network instead of Q-network.

    The critic is a V(s) network. _compute_critics returns the model-implied
    Q(s,a) = r(s,a) + gamma * V(f(s,a)), so the actor loss works unchanged.
    The critic loss trains V via a separate target computation.
    """

    def __init__(self,
                 observation_spec,
                 action_spec,
                 value_network_cls=ValueNetwork,
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
                 name="SacVAlgorithm"):
        # Must set before super().__init__() because _make_networks is called there
        self._value_network_cls = value_network_cls
        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            dynamics_hidden=dynamics_hidden,
            reward_hidden=reward_hidden,
            model_loss_weight=model_loss_weight,
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

    def _make_networks(self, observation_spec, action_spec, reward_spec,
                       continuous_actor_network_cls, critic_network_cls,
                       q_network_cls):
        """Create actor + parallel V-networks instead of Q-networks."""
        actor_network = continuous_actor_network_cls(
            input_tensor_spec=observation_spec, action_spec=action_spec)
        value_network = self._value_network_cls(
            input_tensor_spec=observation_spec)
        critic_networks = value_network.make_parallel(
            self._num_critic_replicas * reward_spec.numel)
        return critic_networks, actor_network, ActionType.Continuous

    def _compute_v(self, v_net, observation, state, replica_min=True):
        """Compute V(s) from a parallel V-network.

        Args:
            v_net: parallel value network
            observation: [B, obs_dim]
            state: network state
            replica_min: if True, return min over replicas
        Returns:
            values: [B] if replica_min else [B, n_replicas]
            state: new network state
        """
        values, state = v_net(observation, state=state)
        # values shape: [B, n_replicas]
        if replica_min:
            values = values.min(dim=-1)[0]
        return values, state

    def _compute_critics(self,
                         critic_net,
                         observation,
                         action,
                         critics_state,
                         replica_min=True,
                         apply_reward_weights=True):
        """Compute Q(s,a) = r(s,a) + gamma * V(f(s,a)).

        When action is None (e.g. during rollout state maintenance), just
        returns V(s) as a fallback.
        """
        if action is None:
            return self._compute_v(critic_net,
                                   observation,
                                   critics_state,
                                   replica_min=replica_min)

        # Compute model-implied Q
        s_next, r_hat = self.predict_next(observation, action)
        v_next, new_state = self._compute_v(critic_net,
                                            s_next,
                                            critics_state,
                                            replica_min=replica_min)

        gamma = self._critic_losses[0].gamma
        if replica_min:
            q = r_hat + gamma * v_next  # [B]
        else:
            # r_hat is [B], v_next is [B, n_replicas]
            q = r_hat.unsqueeze(-1) + gamma * v_next  # [B, n_replicas]
        return q, new_state

    def _critic_train_step(self, observation, target_observation, state,
                           rollout_info, action, action_distribution):
        """Compute V(s) predictions and model-implied V target.

        V_target(s) = r(s, a') + gamma * V_target(f(s, a')) - alpha * log_pi(a'|s)
        where a' ~ pi(.|s) is freshly sampled.
        """
        # V(s) for all replicas — this is what we train
        v_all, critics_state = self._compute_v(self._critic_networks,
                                               observation,
                                               state.critics,
                                               replica_min=False)

        with torch.no_grad():
            # Sample fresh action
            action_dist, _ = self._actor_network(observation)
            a_fresh = dist_utils.rsample_action_distribution(action_dist)
            log_pi = action_dist.log_prob(a_fresh)
            if isinstance(log_pi, (list, tuple)):
                log_pi = sum(nest.flatten(log_pi))

            # Model-implied target
            s_next, r_hat = self.predict_next(observation, a_fresh)
            v_target_next, target_critics_state = self._compute_v(
                self._target_critic_networks, s_next, state.target_critics)

            gamma = self._critic_losses[0].gamma
            if self._use_entropy_reward:
                alpha = torch.exp(self._log_alpha).detach()
                v_target = r_hat + gamma * v_target_next - alpha * log_pi
            else:
                v_target = r_hat + gamma * v_target_next

        state = SacCriticState(critics=critics_state,
                               target_critics=target_critics_state)
        info = SacCriticInfo(critics=v_all, target_critic=v_target)
        return state, info

    def _calc_critic_loss(self, info):
        """Simple MSE between V predictions and model-implied targets.

        Entropy is already in the target via -alpha * log_pi, so we skip
        the parent's entropy reward addition.
        """
        critic_info = info.critic
        target = critic_info.target_critic.detach()

        critic_losses = []
        for i in range(self._num_critic_replicas):
            v_i = critic_info.critics[..., i]
            loss_i = losses.element_wise_squared_loss(target, v_i)
            critic_losses.append(loss_i)

        critic_loss = math_ops.add_n(critic_losses)

        # Mask LAST steps
        valid = (info.step_type != StepType.LAST).float()
        critic_loss = critic_loss * valid

        return LossInfo(loss=critic_loss,
                        extra=critic_loss / float(self._num_critic_replicas))
