import subprocess

ENV = "BipolarChain-medium-sparse-onehot-discrete-v0"
FOLDER = f"/tmp/bipolar/{ENV}/"

base = {
    "CONF": "experiments/bipolar_conf.py",
    "ENV": ENV,
    "LR": "0.1",
    "WD": "1e-5",
    "GAMMA": "0.95",
    "ALPHA": "0",
    "TAU": "0.05",
    "UTD": "8",
    "RESET_PERIOD": "24",
    "NUM_AGENTS": "4",
    "NUM_ENVS": "4",
    "ASYNC": "True",
    "ENTROPY_REWARD": "False",
    "N_COMPONENTS": "500",
    "SEEDS": "8",
    "BASE_DIR": "",
    "EXTRA_ARGS": "",
}
runs = {
    "with_prior": {
        "PRIOR_SCALE": "2.0",
    },
    "without_prior": {
        "PRIOR_SCALE": "0.0",
    },
}
NAMES = list(runs.keys())
EPISODE_INDEX_BASE_AGENTS = 4


def pueue_add(command: str, after: list[str] | None = None) -> str:
    args = ["pueue", "add", "-p"]
    for task_id in (after or []):
        args += ["-a", task_id]
    args += ["--", command]
    return subprocess.run(args, check=True, capture_output=True,
                          text=True).stdout.strip()


def main():
    task_ids: list[str] = []
    for name, overrides in runs.items():
        p = {**base, **overrides}
        args = " ".join(f'{k}="{v}"' for k, v in p.items())
        cmd = f'bash scripts/run_bipolar.sh {args} NAME="{name}"'
        task_ids.append(pueue_add(cmd))

    plot_cmd = ("python tools/custom_plot.py "
                f"--folder {FOLDER} "
                f"--episode_index_base_agents {EPISODE_INDEX_BASE_AGENTS} "
                f"--names {' '.join(NAMES)}")
    pueue_add(plot_cmd, after=task_ids)


if __name__ == "__main__":
    main()
