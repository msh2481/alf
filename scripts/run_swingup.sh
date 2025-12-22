ROOT_DIR="/tmp/swingup/$(date +%Y%m%d_%H%M%S)"
python -m alf.bin.train --conf=experiments/swingup_conf.py --root_dir="$ROOT_DIR"
