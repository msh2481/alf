# Copyright (c) 2025 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
from absl import logging
import itertools
import json
import os
import shutil
import subprocess
import sys
from typing import Dict, List
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import alf
from alf.utils import common

CONF_FILE = "alf/examples/seed_sac_conf.py"
ROOT_DIR = "/tmp/experiments"
OUTPUT_DIR = "/tmp/results"

GRID_CONFIG = {
    'SeedSacAlgorithm.reward_noise_std': [1e-3, 1e-4],
    'SeedSacAlgorithm.parameter_target_std': [1.0, 0.5],
    'SeedSacAlgorithm.parameter_target_alpha': [0.001],
    'ActionRepulsionAlgorithm.repulsion_alpha': [1e-4, 1e-5],
}


def generate_param_configs(grid_config: Dict) -> List[Dict]:
    param_keys = []
    param_values = []

    for key, value in grid_config.items():
        if isinstance(value, list):
            param_keys.append(key)
            param_values.append(value)
        else:
            param_keys.append(key)
            param_values.append([value])

    param_configs = []
    for idx, combination in enumerate(itertools.product(*param_values),
                                      start=1):
        config = dict(zip(param_keys, combination))
        config['run_id'] = f'run_{idx:03d}'
        param_configs.append(config)

    return param_configs


PARAM_CONFIGS = generate_param_configs(GRID_CONFIG)


def run_training(conf_file: str, root_dir: str, params: Dict) -> str:
    run_id = params.pop('run_id')
    run_dir = os.path.join(root_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    logging.info(f"Running experiment {run_id} with params: {params}")

    pre_configs = dict(params)
    pre_configs['TrainerConfig.confirm_checkpoint_upon_crash'] = False

    conf_params = [f"{k}={v}" for k, v in pre_configs.items()]

    cmd = [
        sys.executable,
        '-m',
        'alf.bin.train',
        '--root_dir',
        run_dir,
        '--conf',
        conf_file,
    ]
    for param in conf_params:
        cmd.extend(['--conf_param', param])

    logging.info(f"Running command: {' '.join(cmd)}")

    result = subprocess.run(cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True)

    if result.returncode != 0:
        logging.error(f"Training failed for {run_id}")
        logging.error(f"STDOUT: {result.stdout}")
        logging.error(f"STDERR: {result.stderr}")
        raise RuntimeError(f"Training failed for {run_id}")

    logging.info(f"Training completed for {run_id}")
    return run_dir


def extract_average_return_curve(run_dir: str) -> Dict:
    train_dir = os.path.join(run_dir, 'train')
    if not os.path.exists(train_dir):
        logging.warning(f"Train directory not found: {train_dir}")
        return None

    event_file = None
    for root, dirs, files in os.walk(train_dir):
        for file_name in files:
            if "events" in file_name and 'profile' not in file_name:
                event_file = os.path.join(root, file_name)
                break

    if event_file is None:
        logging.warning(f"No event file found in {train_dir}")
        return None

    logging.info(f"Parsing event file: {event_file}")
    event_acc = EventAccumulator(event_file)
    event_acc.Reload()

    metric_names = ['Metrics/AverageReturn']

    for metric_name in metric_names:
        print("Saving metric: ", metric_name)
        if metric_name in event_acc.Tags()['scalars']:
            logging.info(f"Found metric: {metric_name}")
            scalar_events = event_acc.Scalars(metric_name)
            steps = [e.step for e in scalar_events]
            values = [e.value for e in scalar_events]
            return {
                'steps': steps,
                'values': values,
                'metric_name': metric_name,
            }

    logging.warning(f"AverageReturn metric not found in event file")
    return None


def run_experiment(conf_file: str, root_dir: str, output_dir: str,
                   params: Dict) -> Dict:
    run_id = params['run_id']
    logging.info(f"Starting experiment: {run_id}")

    run_dir = run_training(conf_file, root_dir, params.copy())
    curve = extract_average_return_curve(run_dir)

    result = {
        'run_id': run_id,
        'parameters': params,
        'run_dir': run_dir,
        'average_return_curve': curve,
    }

    output_file = os.path.join(output_dir, f"{run_id}.json")
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    logging.info(f"Saved results to {output_file}")
    return result


def main():
    conf_file = common.abs_path(CONF_FILE)
    if not os.path.exists(conf_file):
        logging.error(f"Config file not found: {conf_file}")
        sys.exit(1)

    root_dir = common.abs_path(ROOT_DIR)
    if os.path.exists(root_dir):
        response = input(
            f"Root directory {root_dir} already exists. Delete it? (yes/no): ")
        if response.lower() in ('yes', 'y'):
            logging.info(f"Deleting {root_dir}")
            shutil.rmtree(root_dir)
        else:
            logging.info("Exiting without deleting.")
            sys.exit(0)

    os.makedirs(root_dir, exist_ok=True)

    output_dir = common.abs_path(OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"Running {len(PARAM_CONFIGS)} experiments")
    logging.info(f"Config file: {conf_file}")
    logging.info(f"Root directory: {root_dir}")
    logging.info(f"Output directory: {output_dir}")

    for i, params in enumerate(PARAM_CONFIGS):
        logging.info(f"\n{'='*60}")
        logging.info(
            f"Experiment {i+1}/{len(PARAM_CONFIGS)}: {params['run_id']}")
        logging.info(f"{'='*60}")

        try:
            result = run_experiment(conf_file, root_dir, output_dir, params)
            logging.info(f"Successfully completed {params['run_id']}")
        except Exception as e:
            logging.error(f"Failed to run {params['run_id']}: {e}")
            continue

    logging.info(
        f"\nAll experiments completed. Results saved in: {output_dir}")


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    main()
