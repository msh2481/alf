ROOT_DIR="/tmp/bipolar/$(date +%Y%m%d_%H%M%S)"
python -m alf.bin.train --conf=experiments/bipolar_conf.py --root_dir="$ROOT_DIR"
