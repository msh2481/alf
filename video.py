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

import re
import subprocess
import tempfile
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def extract_number(filename):
    match = re.search(r'(\d+)\.png$', filename)
    return int(match.group(1)) if match else 0


def create_video(run_name,
                 logs_root='logs',
                 output_file='output.mp4',
                 fps=1,
                 font_scale=1.0):
    logs_path = Path(logs_root) / run_name / '0'
    png_files = sorted([f for f in logs_path.glob('*.png')],
                       key=lambda x: extract_number(x.name))

    if not png_files:
        print(f"No PNG files found in {logs_path}")
        return

    # Build a clean, sequential image list for ffmpeg.
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        font_size = max(1, int(12 * font_scale))
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=font_size)
        except OSError:
            font = ImageFont.load_default()
            if font_scale != 1.0:
                print(
                    "Warning: scalable font unavailable; using default font size."
                )

        for idx, png_file in enumerate(png_files):
            target = tmp_path / f"{idx:06d}.png"
            timestep = str(extract_number(png_file.name))
            try:
                with Image.open(png_file).convert('RGB') as image:
                    draw = ImageDraw.Draw(image)
                    text_bbox = draw.textbbox((0, 0), timestep, font=font)
                    text_height = text_bbox[3] - text_bbox[1]
                    draw.text((10, image.height - text_height - 10),
                              timestep,
                              fill='black',
                              font=font)
                    image.save(target)
            except Exception as exc:
                print(f"Frame annotation failed for {png_file}: {exc}")
                return

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
    parser = argparse.ArgumentParser(
        description='Create a video from logs/<x>/0 PNG frames.')
    parser.add_argument('x',
                        help='Subdirectory name under logs/, e.g. with_prior')
    parser.add_argument('--logs-root',
                        default='logs',
                        help='Root logs directory (default: logs)')
    parser.add_argument('--output',
                        default='output.mp4',
                        help='Output mp4 filename (default: output.mp4)')
    parser.add_argument('--fps',
                        type=int,
                        default=1,
                        help='Video FPS (default: 1)')
    parser.add_argument(
        '--font-scale',
        type=float,
        default=5.0,
        help='Scale factor for timestep text size (default: 5.0)')
    args = parser.parse_args()
    create_video(args.x,
                 logs_root=args.logs_root,
                 output_file=args.output,
                 fps=args.fps,
                 font_scale=args.font_scale)
