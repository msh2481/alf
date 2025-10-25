I need to design an architecture for a concurrent RL algorithm, that would fit into this RL library.

## Interfaces used in the library

### RLAlgorithm & Algorithm
Short summary of implementation from `alf/algorithms/rl_algorithm.py` and `alf/algorithms/algorithm.py`, with example shapes from `alf/examples/toy_conf.py`:

```python
def train_iter(self):
    # Off-policy case
    self.unroll(self._config.unroll_length)
    steps = self.train_from_replay_buffer(update_global_counter=True)
    return steps
```

```python
def unroll(self, unroll_length: int):
    time_step = self._current_time_step
    policy_state = self._current_policy_state
    experience_list = []
    for _ in range(unroll_length):
        policy_step = self.rollout_step(time_step, policy_state)
        """
        time_step.observation: [num_envs, obs_dim] = [2, 4]
        policy_step.output (action): [num_envs] = [2]
        policy_step.info.action_distribution.logits: [num_envs, num_actions] = [2, 2]
        """
        action = common.detach(policy_step.output)

        exp = Experience(
            time_step=time_step,
            action=policy_step.output,
            rollout_info=policy_step.info,
            state=(),
        )
        self._set_replay_buffer(exp)

        exp_for_training = Experience(
            time_step=time_step,
            action=action,
            rollout_info=dist_utils.distributions_to_params(policy_step.info),
            state=policy_state)
        experience_list.append(exp_for_training)

        time_step = self._env.step(action)
        policy_state = policy_step.state

    experience = alf.nest.utils.stack_nests(experience_list)
    """
    After stacking:
    experience.time_step.observation: [unroll_length, num_envs, obs_dim] = [1, 2, 4]
    experience.action: [unroll_length, num_envs] = [1, 2]
    """
    experience = experience._replace(
        rollout_info=dist_utils.params_to_distributions(experience.rollout_info, self._rollout_info_spec)
    )
    self._current_time_step = time_step
    self._current_policy_state = common.detach(policy_state)
    return experience
```

```python
def train_from_replay_buffer(self, update_global_counter=False):
    config: TrainerConfig = self._config

    experience, batch_info = self._replay_buffer.get_batch(
        batch_size=(config.mini_batch_size *
                    config.num_updates_per_train_iter),
        batch_length=config.mini_batch_length)
    """
    After sampling from replay buffer:
    experience.time_step.observation: [mini_batch_size, mini_batch_length, obs_dim] = [7, 3, 4]
    experience.action: [mini_batch_size, mini_batch_length] = [7, 3]
    experience.rollout_info.action_distribution.logits: [mini_batch_size, mini_batch_length, num_actions] = [7, 3, 2]
    batch_info.env_ids: [mini_batch_size] = [7]
    batch_info.positions: [mini_batch_size] = [7]
    """

    experience = dist_utils.params_to_distributions(experience, experience_spec)
    processed_exp_spec = dist_utils.extract_spec(experience, from_dim=2)
    experience = dist_utils.distributions_to_params(experience)

    assert mini_batch_length == alf.nest.get_nest_size(experience, dim=1)
    experience = alf.nest.map_structure(lambda x: x.reshape(-1, mini_batch_length, *x.shape[2:]), experience)
    """
    After reshape (same shape in this case since batch_size=7, mini_batch_size=7):
    experience.time_step.observation: [mini_batch_size, mini_batch_length, obs_dim] = [7, 3, 4]
    """
    batch_size = alf.nest.get_nest_batch_size(experience)

    indices = torch.randperm(batch_size)
    for b in range(0, batch_size, mini_batch_size):
        mini_batch_list, mini_batch_info_list = self._extract_mini_batch_and_info_from_experience(
            indices, [experience], [batch_info], batch_size, b, mini_batch_size)

        exp, train_info, loss_info, params = self._update(
            mini_batch_list[0], mini_batch_info_list[0]
        )
    train_steps = batch_size * mini_batch_length
    return train_steps


def _extract_mini_batch_and_info_from_experience(self,
                                                    indices,
                                                    experience_list,
                                                    batch_info_list,
                                                    batch_size,
                                                    start,
                                                    size):
        batch_indices = indices[start : start + size]

        def _make_time_major(nest):
            return alf.nest.map_structure(lambda x: x.transpose(0, 1), nest)

        mini_batch_list, mini_batch_info_list = [], []
        for experience, batch_info in zip(experience_list, batch_info_list):
            batch = alf.nest.map_structure(lambda x: x[batch_indices], experience)
            binfo = alf.nest.map_structure(lambda x: x[batch_indices] if isinstance(x, torch.Tensor) else x, batch_info)
            batch = _make_time_major(batch)
            """
            After transpose (time-major):
            batch.time_step.observation: [mini_batch_length, mini_batch_size, obs_dim] = [3, 7, 4]
            batch.action: [mini_batch_length, mini_batch_size] = [3, 7]
            """
            batch = alf.nest.utils.convert_device(batch)

            mini_batch_list.append(batch)
            mini_batch_info_list.append(binfo)
        return mini_batch_list, mini_batch_info_list

def _update(self, experience, batch_info):
    train_info = self._collect_train_info_sequentially(experience)
    loss_info = train_info # or calc_loss(train_info), optionally
    loss_info, params = self.update_with_gradient(loss_info, batch_info)
    return experience, train_info, loss_info, params

def _collect_train_info_sequentially(self, experience):
    """
    Input experience is time-major:
    experience.time_step.observation: [mini_batch_length, mini_batch_size, obs_dim] = [3, 7, 4]
    """
    batch_size = alf.nest.get_nest_size(experience, dim=1)
    policy_state = self.get_initial_train_state(batch_size)

    info_list = []
    for counter in range(alf.nest.get_nest_size(experience, dim=0)):
        exp = alf.nest.map_structure(lambda ta: ta[counter], experience)
        exp = dist_utils.params_to_distributions(exp, self.processed_experience_spec)

        policy_step = self.train_step(exp.time_step, policy_state, exp.rollout_info)
        info_list.append(dist_utils.distributions_to_params(policy_step.info))
        policy_state = policy_step.state

    info = alf.nest.utils.stack_nests(info_list)
    info = dist_utils.params_to_distributions(info, self.train_info_spec)
    return info

def update_with_gradient(self, loss_info):
    """
    Input loss_info (after calc_loss):
    loss_info.loss: [mini_batch_length, mini_batch_size] = [3, 7]
    loss_info.extra.rl.critic: [mini_batch_length, mini_batch_size] = [3, 7]
    loss_info.extra.rl.alpha: [mini_batch_length, mini_batch_size] = [3, 7]
    """
    optimizers = self.optimizers()
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)

    all_params = []
    for optimizer in optimizers:
        params = []
        for param_group in optimizer.param_groups:
            params.extend(param_group['params'])
        all_params.extend(params)

    loss_info.loss.mean().backward()

    for optimizer in optimizers:
        optimizer.step()

    all_params = [(self._param_to_name[p], p) for p in all_params]
    loss_info = alf.nest.map_structure(torch.mean, loss_info)
    return loss_info, all_params
```

```python
def train_step(self, inputs, state, rollout_info):
    """Perform one step of training computation.

    It is called to calculate output for every time step for a batch of
    experience from replay buffer. It also needs to generate necessary
    information for ``calc_loss()``.

    Args:
        inputs (nested Tensor): inputs for train.
        state (nested Tensor): consistent with ``train_state_spec``.
        rollout_info (nested Tensor): info from ``rollout_step()``. It is
            retrieved from replay buffer.
    Returns:
        AlgStep:
        - output (nested Tensor): prediction result.
        - state (nested Tensor): should match ``train_state_spec``.
        - info (nested Tensor): information for training. It will temporally
            batched and passed as ``info`` for calc_loss(). If this is
            ``LossInfo``, ``calc_loss()`` in ``Algorithm`` can be used.
            Otherwise, the user needs to override ``calc_loss()`` to
            calculate loss or override ``update_with_gradient()`` to do
            customized training.
    """
    raise NotImplementedError()
```
