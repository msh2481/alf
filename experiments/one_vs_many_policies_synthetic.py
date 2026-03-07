import subprocess

ENV = "BipolarChain-medium-sparse-onehot-discrete-v0"
FOLDER = f"/tmp/bipolar/{ENV}/"

N = 4

base = {
    "CONF": "experiments/bipolar_conf.py",
    "ENV": ENV,
    "LR": "0.1",
    "WD": "1e-5",
    "GAMMA": "0.95",
    "ALPHA": "0",
    "TAU": "0.05",
    "UTD": "8",
    "RESET_PERIOD": "500",
    "NUM_AGENTS": "1",
    "NUM_ENVS": f"{N}",
    "OWN_ROLLOUT_FRACTION": "0.75",
    "ASYNC": "True",
    "ENTROPY_REWARD": "False",
    "N_COMPONENTS": "500",
    "SEEDS": "16",
    "BASE_DIR": "",
}
runs = {
    # 1) N parallel envs, 1 agent, no prior
    f"a1_e{N}_prior0": {
        "NUM_AGENTS": "1",
        "PRIOR_SCALE": "0.0",
    },
    # 2) N parallel envs, 1 agent, with prior
    f"a1_e{N}_prior1.0": {
        "NUM_AGENTS": "1",
        "PRIOR_SCALE": "1.0",
    },
    # 3) N parallel envs, N agents, no prior
    f"a{N}_e{N}_prior0.0": {
        "NUM_AGENTS": f"{N}",
        "PRIOR_SCALE": "0.0",
    },
    # 4) N parallel envs, N agents, with prior
    f"a{N}_e{N}_prior1.0": {
        "NUM_AGENTS": f"{N}",
        "PRIOR_SCALE": "1.0",
    },
}
NAMES = list(runs.keys())


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
        task_ids.append(
            pueue_add(f'bash scripts/run_bipolar.sh {args} NAME="{name}"'))

    plot_cmd = ("python tools/custom_plot.py "
                f"--folder {FOLDER} "
                f"--episode_index_base_agents {N} "
                f"--names {' '.join(NAMES)}")
    pueue_add(plot_cmd, after=task_ids)


if __name__ == "__main__":
    main()
