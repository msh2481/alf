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

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


def extract_number(filename):
    match = re.search(r'(\d+)\.png$', filename)
    return int(match.group(1)) if match else 0


def create_video(logs_dir='logs', output_file='output.mp4', fps=1):
    logs_path = Path(logs_dir)
    png_files = sorted([f for f in logs_path.glob('*.png')],
                       key=lambda x: extract_number(x.name))

    if not png_files:
        print(f"No PNG files found in {logs_dir}")
        return

    # Build a clean, sequential image list for ffmpeg.
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        for idx, png_file in enumerate(png_files):
            target = tmp_path / f"{idx:06d}.png"
            try:
                os.symlink(png_file.resolve(), target)
            except OSError:
                shutil.copy(png_file, target)

        # Baseline: mimic the old OpenCV mp4v output, directly via ffmpeg.
        cmd = [
            'ffmpeg', '-y', '-framerate',
            str(fps), '-i',
            str(tmp_path / '%06d.png'), '-c:v', 'mpeg4', '-vtag', 'mp4v',
            '-qscale:v', '2', '-pix_fmt', 'yuv420p', output_file
        ]

        try:
            result = subprocess.run(cmd,
                                    check=True,
                                    capture_output=True,
                                    text=True)
            if result.stderr:
                print('\n'.join(result.stderr.strip().splitlines()[-5:]))
            print(f"Video created: {output_file}")
        except FileNotFoundError:
            print(
                "ffmpeg not found. Please install ffmpeg to enable compressed video output."
            )
        except subprocess.CalledProcessError as exc:
            print("ffmpeg failed:\n" + exc.stderr)


if __name__ == '__main__':
    create_video()
