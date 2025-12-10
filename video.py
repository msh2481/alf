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
from pathlib import Path
import cv2


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

    first_image = cv2.imread(str(png_files[0]))
    if first_image is None:
        print(f"Failed to read {png_files[0]}")
        return

    height, width, _ = first_image.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for png_file in png_files:
        img = cv2.imread(str(png_file))
        if img is not None:
            video_writer.write(img)
            print(f"Added {png_file.name}")
        else:
            print(f"Warning: Failed to read {png_file.name}")

    video_writer.release()
    print(f"Video created: {output_file}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Create video from PNG files in logs directory')
    parser.add_argument('--logs-dir',
                        default='logs',
                        help='Directory containing PNG files')
    parser.add_argument('--output',
                        default='output.mp4',
                        help='Output video file')
    parser.add_argument('--fps',
                        type=float,
                        default=1.0,
                        help='Frames per second (default: 1.0)')
    args = parser.parse_args()

    create_video(args.logs_dir, args.output, args.fps)
