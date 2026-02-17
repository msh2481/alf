import subprocess

base = {
    "CONF": "experiments/bipolar_conf.py",
    "ENV": "BipolarChain-medium-sparse-onehot-discrete-v0",
    "LR": "0.1",
    "WD": "1e-5",
    "GAMMA": "0.95",
    "ALPHA": "0",
    "TAU": "0.05",
    "UTD": "8",
    "RESET_PERIOD": "24",
    "NUM_AGENTS": "4",
    "ASYNC": "True",
    "ENTROPY_REWARD": "False",
    "N_COMPONENTS": "500",
    "SEEDS": "",
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


def main():
    for name, overrides in runs.items():
        p = {**base, **overrides}
        args = " ".join(f'{k}="{v}"' for k, v in p.items())
        cmd = f'bash scripts/run_bipolar.sh {args} NAME="{name}"'
        subprocess.run(["pueue", "add", "--", cmd], check=True)


if __name__ == "__main__":
    main()
