# Experiment Management

## Remote Server
```
ssh Mikhail.Budnikov@34.40.32.81
```

## Launching Experiments

Each experiment runs in a named tmux session:

```bash
tmux new-session -d -s <name> "cd ~/alf && source .venv/bin/activate && bash scripts/run_cartpole_swingup.sh <PARAMS> NAME='<name>'"
```

Example:
```bash
tmux new-session -d -s utd_2 "cd ~/alf && source .venv/bin/activate && bash scripts/run_cartpole_swingup.sh UTD=2 NAME='utd_2'"
```

## Tracking Running Experiments

### experiments.txt
Located at `~/alf/experiments.txt`. Format:
```
# tmux_session | experiment_command
utd_1 | UTD=1 NAME="utd_1"
```

### List tmux sessions
```bash
tmux list-sessions
```

### Attach to a session (see live output)
```bash
tmux attach -t <name>
# Detach: Ctrl+B, then D
```

## Checking Results

Results are stored in `/tmp/cartpole_swingup/<name>/`

### Check episode returns
```bash
cat /tmp/cartpole_swingup/<name>/plots/episode_returns.txt
```

### Check all metrics (use cat or tail -100 for full plots)
```bash
cat /tmp/cartpole_swingup/<name>/plots/episode_returns.txt
cat /tmp/cartpole_swingup/<name>/plots/actor_losses.txt
cat /tmp/cartpole_swingup/<name>/plots/critic_losses.txt
```

### Check training log
```bash
tail -100 /tmp/cartpole_swingup/<name>/py_train.INFO
```

## Killing Experiments

```bash
tmux kill-session -t <name>
```
