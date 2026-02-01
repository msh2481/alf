from dm_control import suite

names = [f"{a}:{b}" for a, b in suite.BENCHMARKING]
print(names)

for n in names:
    commands = [
        f'scripts/run_dmc.sh SEEDS="8" NUM_AGENTS=1 ENV="{n}" NAME="a1"',
        f'scripts/run_dmc.sh SEEDS="8" NUM_AGENTS=4 ENV="{n}" NAME="a4"',
    ]
    for c in commands:
        print(f"pueue add -- '{c}'")

