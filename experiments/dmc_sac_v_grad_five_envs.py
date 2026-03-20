"""DMC sweep: five tasks × (sac_grad @ 1e2/1e3, sac_v @ prior 0.01/0.1), 16 seeds each.

Base hyperparameters match ``prior_vs_no_prior_dmc_four_agents_soft_reset.py``;
training uses ``scripts/run_sac_model.sh`` / ``sac_model_conf.py``. Default
``PRIOR_SCALE`` in ``base`` applies to ``sac_grad``; ``sac_v`` runs override it.
"""
import subprocess

ENVS = [
    "cartpole:swingup",
    "cartpole:swingup_sparse",
    "cheetah:run",
    "fish:swim",
    "hopper:hop",
]

base = {
    "LR": "1e-3",
    "WD": "1e-5",
    "GAMMA": "0.99",
    "PRIOR_SCALE": "0.01",
    "ALPHA": "5e-4",
    "LN": "True",
    "UTD": "1",
    "RESET_PERIOD": "1",
    "NUM_AGENTS": "4",
    "NUM_ENVS": "4",
    "SCALE_BATCH": "True",
    "UNROLL_LENGTH": "1",
    "SHUFFLE": "False",
    "TAU": "0.01",
    "N_CRITICS": "1",
    "ASYNC": "False",
    "SHARE_ACTOR": "False",
    "SHARE_CRITIC": "False",
    "OWN_ROLLOUT_FRACTION": "0.75",
    "NUM_LAYERS": "2",
    "MODEL_LOSS_WEIGHT": "1.0",
    "GRAD_SYNC_WEIGHT": "1.0",
    "DYNAMICS_HIDDEN": "(256,256)",
    "REWARD_HIDDEN": "(256,256)",
    "SEEDS": "16",
    "BASE_DIR": "",
}

runs = {
    "sac_grad_1e2": {
        "ALGO": "sac_grad",
        "GRAD_SYNC_WEIGHT": "1e2",
    },
    "sac_grad_1e3": {
        "ALGO": "sac_grad",
        "GRAD_SYNC_WEIGHT": "1e3",
    },
    "sac_v_prior0.01": {
        "ALGO": "sac_v",
        "PRIOR_SCALE": "0.01",
    },
    "sac_v_prior0.1": {
        "ALGO": "sac_v",
        "PRIOR_SCALE": "0.1",
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


def main() -> None:
    task_ids: list[str] = []
    for env in ENVS:
        for name, overrides in runs.items():
            p = {**base, "ENV": env, **overrides}
            args = " ".join(f'{k}="{v}"' for k, v in p.items())
            task_ids.append(
                pueue_add(
                    f'bash scripts/run_sac_model.sh {args} NAME="{name}"'))

    plot_cmd = ("python tools/custom_plot.py "
                f"--episode_index_base_agents {EPISODE_INDEX_BASE_AGENTS} "
                f"--names {' '.join(NAMES)}")
    pueue_add(plot_cmd, after=task_ids)


if __name__ == "__main__":
    main()
