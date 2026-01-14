# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
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

import gym

gym.register(
    id='CheckValue-v0',
    entry_point='alf.environments.simple.check_value:CheckValue',
)

gym.register(
    id='CheckPolicy-v0',
    entry_point='alf.environments.simple.check_policy:CheckPolicy',
)

gym.register(
    id='BipolarChain-v0',
    entry_point='alf.environments.simple.bipolar_chain:BipolarChain',
)

for continuous_name, continuous in [('discrete', False), ('continuous', True)]:
    for size_name, k in [('small', 6), ('medium', 12), ('big', 24)]:
        for reward_name, dense in [('sparse', False), ('dense', True)]:
            for obs_name, factored in [('onehot', False), ('factored', True)]:
                env_id = f'BipolarChain-{size_name}-{reward_name}-{obs_name}-{continuous_name}-v0'
                gym.register(
                    id=env_id,
                    entry_point=
                    'alf.environments.simple.bipolar_chain:BipolarChain',
                    kwargs={
                        'k': k,
                        'dense': dense,
                        'factored': factored,
                        'continuous': continuous
                    })

gym.register(
    id='ParallelChains-v0',
    entry_point='alf.environments.simple.parallel_chains:ParallelChains',
)

gym.register(
    id='Bezier-v0',
    entry_point='alf.environments.simple.bezier:Bezier',
)

gym.register(
    id='LongHorizon-v0',
    entry_point='alf.environments.simple.long_horizon:LongHorizon',
)

gym.register(
    id='Rotator-v0',
    entry_point='alf.environments.simple.rotator:Rotator',
    max_episode_steps=1000,
)
