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
"""
Debug logging utility for exploring nested data structures.

Only logs the i-th call for each name when i is a power of two (1, 2, 4, 8, 16, ...)
"""
import os
from typing import Any, Dict


class DebugLogger:

    def __init__(self, log_file: str = "log.txt"):
        self.log_file = log_file
        self.call_counts: Dict[str, int] = {}

        # Clear the log file on initialization
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    def log(self, name: str, content: Any):
        if name not in self.call_counts:
            self.call_counts[name] = 0
        self.call_counts[name] += 1

        count = self.call_counts[name]

        if count in [1, 10]:
            with open(self.log_file, 'a') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"[{name}] - Call #{count}\n")
                f.write(f"{'='*80}\n")
                f.write(str(content))
                f.write(f"\n{'='*80}\n\n")

    def reset(self):
        self.call_counts.clear()


# Global logger instance
_logger = DebugLogger()


def log(name: str, content: Any):
    """
    Convenience function to log using the global logger.

    Args:
        name: The name/category for this log entry
        content: The content to log
    """
    _logger.log(name, content)


def reset_logger():
    """Reset the global logger."""
    _logger.reset()


def get_call_count(name: str) -> int:
    """Get the call count for a given name."""
    return _logger.call_counts.get(name, 0)
