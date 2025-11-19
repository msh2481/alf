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
"""Simple profiler that logs timing immediately with running average."""

import time
from contextlib import contextmanager
from typing import Dict


class SimpleProfiler:
    """Simple profiler that prints timing immediately with running average.

    Uses exponential moving average (EMA) for efficient averaging without
    keeping lists of measurements.

    Usage:
        profiler = SimpleProfiler()
        profiler.start("my_operation")
        # ... do work ...
        profiler.end("my_operation")  # Prints: "my_operation: 15ms, avg 15ms"

    Or use context manager:
        with profiler.profile("my_operation"):
            # ... do work ...
    """

    def __init__(self, ema_period: int = 8):
        """Initialize profiler.

        Args:
            ema_period: Period for exponential moving average (e.g., 1000 means
                approximately averaging over last 1000 measurements).
        """
        self._starts: Dict[str, float] = {}
        self._ema_value: Dict[str, float] = {}  # EMA of measurements
        self._ema_weight: Dict[str,
                               float] = {}  # EMA of 1s (for bias correction)
        self._alpha = 1.0 / ema_period

    def start(self, tag: str):
        """Start timing for a tag.

        Args:
            tag: Identifier for this timing measurement.
        """
        indent = "\t" * len(self._starts)
        print(f"{indent}Starting {tag} " +
              "=" * max(1, 100 - len(tag) - len(indent) * 8))
        self._starts[tag] = time.time()

    def end(self, tag: str):
        """End timing and print: 'tag: Xms, avg Yms'.

        Args:
            tag: Identifier for this timing measurement (must match start() call).
        """
        if tag not in self._starts:
            print(f"⚠️  SimpleProfiler: start() not called for tag: {tag}")
            return

        duration_ms = (time.time() - self._starts[tag]) * 1000

        # Bias-corrected EMA: track EMA of values and EMA of 1s
        if tag not in self._ema_value:
            self._ema_value[tag] = 0.0
            self._ema_weight[tag] = 0.0

        # Update EMAs
        self._ema_value[tag] = (
            1 - self._alpha) * self._ema_value[tag] + self._alpha * duration_ms
        self._ema_weight[tag] = (
            1 - self._alpha) * self._ema_weight[tag] + self._alpha * 1.0

        # Adjusted average: divide value EMA by weight EMA
        avg_ms = self._ema_value[tag] / self._ema_weight[tag]

        # Indent based on depth (before removing from _starts)
        indent = "\t" * (len(self._starts) - 1)
        print(f"{indent}{tag}: {int(duration_ms)}ms, avg {int(avg_ms)}ms")
        del self._starts[tag]

    @contextmanager
    def profile(self, tag: str):
        """Context manager for profiling a code block.

        Args:
            tag: Identifier for this timing measurement.

        Example:
            with profiler.profile("my_operation"):
                # ... do work ...
        """
        self.start(tag)
        try:
            yield
        finally:
            self.end(tag)


# Global profiler instance
_global_profiler = SimpleProfiler()


def profile_start(tag: str):
    """Start timing for a tag using the global profiler.

    Args:
        tag: Identifier for this timing measurement.
    """
    _global_profiler.start(tag)


def profile_end(tag: str):
    """End timing and print result using the global profiler.

    Args:
        tag: Identifier for this timing measurement.
    """
    _global_profiler.end(tag)


@contextmanager
def profile(tag: str):
    """Context manager for profiling using the global profiler.

    Args:
        tag: Identifier for this timing measurement.

    Example:
        with profile("my_operation"):
            # ... do work ...
    """
    with _global_profiler.profile(tag):
        yield
