from dm_control import suite
import subprocess

base = {
    "LR": "1e-3",
    "WD": "1e-5",
    "GAMMA": "0.99",
    "ALPHA": "0",
    "LN": "True",
    "UTD": "1",
    "RESET_PERIOD": "1e9",
    "NUM_AGENTS": "1",
    "NUM_ENVS": "32",
    "SCALE_BATCH": "True",
    "UNROLL_LENGTH": "0.25",
    "SHUFFLE": "False",
    "TAU": "0.01",
    "N_CRITICS": "1",
    "ASYNC": "True",
    "SHARE_ACTOR": "False",
    "SHARE_CRITIC": "False",
    "OWN_ROLLOUT_FRACTION": "-1",
    "SEEDS": "",
    "BASE_DIR": "",
}
runs = {
    "a1_prior0": {
        "PRIOR_SCALE": "0.0"
    },
    "a1_prior_default": {
        "PRIOR_SCALE": "0.01"
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


def main() -> None:
    task_ids: list[str] = []
    for env in [f"{a}:{b}" for a, b in suite.BENCHMARKING]:
        for name, overrides in runs.items():
            p = {**base, "ENV": env, **overrides}
            args = " ".join(f'{k}="{v}"' for k, v in p.items())
            task_ids.append(
                pueue_add(f'bash scripts/run_dmc.sh {args} NAME="{name}"'))

    plot_cmd = f"python tools/custom_plot.py --names {' '.join(NAMES)}"
    pueue_add(plot_cmd, after=task_ids)


if __name__ == "__main__":
    main()
