"""
Entry point.   uv sync  then:

    uv run main.py            -> play the best trained agent (terminal log + 3D viewer)
    uv run main.py train      -> runs only the experiments that have no saved result yet (never deletes or overwrites saved results)
    uv run main.py evaluate   -> make plots + generalisation test from logs/
"""
import sys


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "play"
    rest = sys.argv[2:]
    if cmd.startswith("-"):             # e.g. `uv run main.py --algo ppo`
        cmd, rest = "play", sys.argv[1:]
    sys.argv = [sys.argv[0]] + rest
    if cmd == "train":
        from training.sweeps import run_algo
        for a in ["dqn", "reinforce", "ppo", "a2c"]:
            run_algo(a)
    elif cmd == "evaluate":
        from evaluate import main as ev
        ev()
    else:
        from play import main as play
        play()


if __name__ == "__main__":
    main()
