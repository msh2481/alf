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
"""ASCII metric plotting utilities using plotille."""

from collections import defaultdict
import re
import plotille


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    return re.sub(r'\x1b\[[0-9;]*m', '', text)


class AsciiMetricPlotter:
    """Plots training metrics as ASCII charts with scatter points and smoothed lines."""

    def __init__(
        self,
        metrics_to_plot: list[str] | None = None,
        smoothing_fraction: float = 0.1,
        width: int = 120,
        height: int = 40,
    ):
        if metrics_to_plot is None:
            metrics_to_plot = ["AverageReturn", "AverageEpisodeLength"]
        self._metrics_to_plot = metrics_to_plot
        self._smoothing_fraction = smoothing_fraction
        self._width = width
        self._height = height
        self._history: dict[str, list[tuple[int, float]]] = defaultdict(list)

    def update(self, metrics, env_steps: int):
        """Update history with current metric values.
        
        Args:
            metrics: list of StepMetric objects from algorithm.get_metrics()
            env_steps: current environment step count
        """
        for metric in metrics:
            if metric.name in self._metrics_to_plot:
                value = metric.result()
                if hasattr(value, "item"):
                    value = value.item()
                self._history[metric.name].append((env_steps, float(value)))

    def set_history(self, metric_name: str, data: list[tuple[int, float]]):
        """Set the full history for a metric (replaces any existing data)."""
        self._history[metric_name] = data
        if metric_name not in self._metrics_to_plot:
            self._metrics_to_plot.append(metric_name)

    def get_metric_names(self) -> list[str]:
        """Get list of all metric names with data."""
        return list(self._history.keys())

    def _smooth(self, values: list[float]) -> list[float]:
        """Apply smoothing with constant fraction window.
        
        At index i, averages the last ceil((i+1) * smoothing_fraction) points.
        E.g., with 10% smoothing: point 100 uses last 10, point 1000 uses last 100.
        """
        smoothed = []
        for i in range(len(values)):
            window_size = max(1, int((i + 1) * self._smoothing_fraction + 0.5))
            start = max(0, i - window_size + 1)
            window = values[start:i + 1]
            smoothed.append(sum(window) / len(window))
        return smoothed

    def get_plot_string(self,
                        metric_name: str,
                        use_colors: bool = True) -> str:
        """Get ASCII plot string for a single metric.
        
        Args:
            metric_name: name of the metric to plot
            use_colors: if True, include ANSI color codes; if False, strip them
        
        Returns a string containing the ASCII chart with:
        - Scatter points showing raw data
        - Smoothed line overlay
        """
        if metric_name not in self._history or len(
                self._history[metric_name]) == 0:
            return f"No data for {metric_name}"

        data = self._history[metric_name]
        steps = [d[0] for d in data]
        values = [d[1] for d in data]

        if len(values) < 2:
            return f"{metric_name}: only {len(values)} point(s), need at least 2"

        smoothed = self._smooth(values)

        y_min, y_max = min(values), max(values)
        y_margin = (y_max - y_min) * 0.05 if y_max > y_min else 1.0

        fig = plotille.Figure()
        fig.width = self._width
        fig.height = self._height
        fig.set_x_limits(min_=min(steps), max_=max(steps))
        fig.set_y_limits(min_=y_min - y_margin, max_=y_max + y_margin)
        fig.x_label = "env_steps"
        fig.y_label = metric_name

        fig.color_mode = "names"
        fig.scatter(steps, values, lc="cyan", label="raw")
        fig.plot(steps,
                 smoothed,
                 lc="red",
                 label=f"smooth({self._smoothing_fraction:.0%})")

        header = f"=== {metric_name} (last={values[-1]:.3f}, smooth={smoothed[-1]:.3f}) ==="
        result = header + "\n" + fig.show(legend=True)

        if not use_colors:
            result = _strip_ansi(result)
        return result

    def get_all_plot_strings(self, use_colors: bool = True) -> dict[str, str]:
        """Get plot strings for all tracked metrics."""
        return {
            name: self.get_plot_string(name, use_colors=use_colors)
            for name in self._metrics_to_plot
        }

    def clear(self):
        """Clear all stored history."""
        self._history.clear()


def _save_plot(content: str, filename: str, output_dir: str) -> str:
    """Save plot content to file and return the path."""
    import os
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        f.write(content)
    return path


def run_tests():
    """Run all tests and save outputs to temporary directory."""
    import math
    import os
    import random
    import tempfile

    output_dir = tempfile.mkdtemp(prefix="ascii_plotter_test_")
    print(f"Saving test outputs to: {output_dir}\n")
    saved_files = []

    # Test 1: Basic upward trend
    random.seed(42)
    plotter = AsciiMetricPlotter(
        metrics_to_plot=["TestReturn"],
        smoothing_fraction=0.1,
        width=80,
        height=25,
    )
    for i in range(100):
        env_steps = i * 100
        value = 10 + i * 0.5 + random.gauss(0, 5)
        plotter._history["TestReturn"].append((env_steps, value))
    path = _save_plot(plotter.get_plot_string("TestReturn"), "test1_basic.txt",
                      output_dir)
    saved_files.append(path)

    # Test 2: Different smoothing fractions
    random.seed(123)
    for frac in [0.05, 0.1, 0.2]:
        plotter = AsciiMetricPlotter(
            metrics_to_plot=["SineReturn"],
            smoothing_fraction=frac,
            width=80,
            height=20,
        )
        for i in range(200):
            env_steps = i * 50
            value = 50 + math.sin(i * 0.05) * 20 + i * 0.1 + random.gauss(0, 5)
            plotter._history["SineReturn"].append((env_steps, value))
        path = _save_plot(plotter.get_plot_string("SineReturn"),
                          f"test2_smooth_{int(frac*100):02d}pct.txt",
                          output_dir)
        saved_files.append(path)

    # Test 3: Large RL-like learning curve (default size)
    random.seed(999)
    plotter = AsciiMetricPlotter(
        metrics_to_plot=["AverageReturn"],
        smoothing_fraction=0.1,
    )
    for i in range(500):
        env_steps = i * 1000
        progress = i / 500
        base = 100 * (1 - (1 - progress)**2)
        noise = random.gauss(0, 10 * (1 - progress) + 2)
        value = base + noise
        plotter._history["AverageReturn"].append((env_steps, value))
    path = _save_plot(plotter.get_plot_string("AverageReturn"),
                      "test3_rl_curve.txt", output_dir)
    saved_files.append(path)

    # Test 4: Per-algorithm returns (simulating ConcurrentAlgorithm)
    random.seed(777)
    plotter = AsciiMetricPlotter(
        metrics_to_plot=[],
        smoothing_fraction=0.1,
        width=100,
        height=30,
    )
    num_algs = 4
    per_alg_returns: dict[int, list[tuple[int, float]]] = {
        i: []
        for i in range(num_algs)
    }
    for ep in range(200):
        alg_idx = ep % num_algs
        env_steps = ep * 500
        base_return = 20 + ep * 0.3 + alg_idx * 5
        episode_return = base_return + random.gauss(0, 8)
        per_alg_returns[alg_idx].append((env_steps, episode_return))
    # This is how ConcurrentAlgorithm data connects to plotter:
    for alg_idx, returns in per_alg_returns.items():
        plotter.set_history(f"EpisodeReturn/alg_{alg_idx}", returns)
    for alg_idx in range(num_algs):
        metric_name = f"EpisodeReturn/alg_{alg_idx}"
        path = _save_plot(plotter.get_plot_string(metric_name),
                          f"test4_per_alg_{alg_idx}.txt", output_dir)
        saved_files.append(path)

    print("Saved files:")
    for f in saved_files:
        print(f"  {f}")


if __name__ == "__main__":
    run_tests()
