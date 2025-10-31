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

from absl import app
from absl import flags
from absl import logging
import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table

FLAGS = flags.FLAGS

START_STEP = 100
EMA_SPAN = 3


def _define_flags():
    flags.DEFINE_string('results_dir', None,
                        'Path to results directory containing JSON files.')
    flags.DEFINE_string('param1', None, 'First parameter name to group by.')
    flags.DEFINE_string('param2', None, 'Second parameter name to group by.')
    flags.DEFINE_string(
        'output_file', None,
        'Output file path for the plot. If None, displays interactively.')


def ema_smooth(values, span):
    if len(values) == 0:
        return values
    if span >= len(values):
        span = len(values)
    alpha = 2.0 / (span + 1.0)
    smoothed = [values[0]]
    for i in range(1, len(values)):
        smoothed.append(alpha * values[i] + (1 - alpha) * smoothed[-1])
    return smoothed


def load_results(results_dir):
    results = []
    json_files = glob.glob(os.path.join(results_dir, '*.json'))

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                if 'average_return_curve' in data and data[
                        'average_return_curve'] is not None:
                    results.append(data)
        except Exception as e:
            logging.warning(f"Failed to load {json_file}: {e}")

    return results


def filter_and_smooth_curve(curve, start_step, ema_span):
    if curve is None:
        return None, None

    steps = np.array(curve['steps'])
    values = np.array(curve['values'])

    mask = steps >= start_step
    filtered_steps = steps[mask]
    filtered_values = values[mask]

    if len(filtered_steps) == 0:
        return None, None

    smoothed_values = ema_smooth(filtered_values.tolist(), ema_span)

    return filtered_steps, np.array(smoothed_values)


def group_by_params(results, param1, param2):
    groups = {}

    for result in results:
        params = result.get('parameters', {})
        val1 = params.get(param1)
        val2 = params.get(param2)

        if val1 is None or val2 is None:
            logging.warning(
                f"Missing params for {result.get('run_id')}: {param1}={val1}, {param2}={val2}"
            )
            continue

        key = (val1, val2)
        if key not in groups:
            groups[key] = []
        groups[key].append(result)

    return groups


def get_final_reward(curve, last_n=5):
    if curve is None or 'values' not in curve or len(curve['values']) == 0:
        return None
    values = np.array(curve['values'])
    steps = np.array(curve['steps'])
    mask = steps >= START_STEP
    filtered_values = values[mask]
    if len(filtered_values) == 0:
        return None
    last_values = filtered_values[-min(last_n, len(filtered_values)):]
    return np.mean(last_values)


def print_results_table(results):
    if not results:
        return

    all_param_names = set()
    for result in results:
        params = result.get('parameters', {})
        for key in params.keys():
            if key != 'run_id':
                all_param_names.add(key)

    all_param_names = sorted(list(all_param_names))

    console = Console()
    table = Table(title="Experiment Results")
    table.add_column("Run ID", style="cyan")
    for param_name in all_param_names:
        table.add_column(param_name, style="magenta")
    table.add_column("Final Reward", style="green", justify="right")

    for result in sorted(results, key=lambda x: x.get('run_id', '')):
        run_id = result.get('run_id', 'unknown')
        params = result.get('parameters', {})
        curve = result.get('average_return_curve')
        final_reward = get_final_reward(curve)

        row = [run_id]
        for param_name in all_param_names:
            val = params.get(param_name, 'N/A')
            row.append(str(val))
        row.append(
            f"{final_reward:.3f}" if final_reward is not None else "N/A")
        table.add_row(*row)

    console.print(table)


def plot_results(results_dir, param1, param2, output_file):
    results = load_results(results_dir)
    logging.info(f"Loaded {len(results)} results")

    print_results_table(results)

    groups = group_by_params(results, param1, param2)
    logging.info(f"Grouped into {len(groups)} parameter combinations")

    if len(groups) == 0:
        logging.error("No valid groups found. Check parameter names.")
        return

    unique_val1 = sorted(set(k[0] for k in groups.keys()))
    unique_val2 = sorted(set(k[1] for k in groups.keys()))

    n_rows = len(unique_val2)
    n_cols = len(unique_val1)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for row_idx, val2 in enumerate(unique_val2):
        for col_idx, val1 in enumerate(unique_val1):
            ax = axes[row_idx, col_idx]
            key = (val1, val2)

            if key not in groups:
                ax.set_title(f"{val1}, {val2}\nNo data")
                ax.axis('off')
                continue

            group_results = groups[key]
            final_rewards = []

            for result in group_results:
                curve = result.get('average_return_curve')
                steps, values = filter_and_smooth_curve(
                    curve, START_STEP, EMA_SPAN)

                if steps is not None and len(steps) > 0:
                    run_id = result.get('run_id', 'unknown')
                    run_num = run_id.replace('run_', '#')
                    ax.plot(steps,
                            values,
                            alpha=0.5,
                            linewidth=1,
                            label=run_num)
                    final_rewards.append(values[-1])

            if len(final_rewards) > 0:
                avg_final = np.mean(final_rewards)
                ax.set_title(
                    f"{val1}, {val2}\nAvg final reward: {avg_final:.2f}")
                ax.legend()
            else:
                ax.set_title(f"{val1}, {val2}\nNo valid curves")

            ax.set_xlabel('Step')
            ax.set_ylabel('Average Return')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        logging.info(f"Saved plot to {output_file}")
    else:
        plt.show()


def main(_):
    if FLAGS.results_dir is None:
        logging.error("--results_dir is required")
        return

    if FLAGS.param1 is None or FLAGS.param2 is None:
        logging.error("--param1 and --param2 are required")
        return

    results_dir = os.path.abspath(FLAGS.results_dir)
    if not os.path.exists(results_dir):
        logging.error(f"Results directory not found: {results_dir}")
        return

    plot_results(results_dir, FLAGS.param1, FLAGS.param2, FLAGS.output_file)


if __name__ == '__main__':
    _define_flags()
    logging.set_verbosity(logging.INFO)
    app.run(main)
