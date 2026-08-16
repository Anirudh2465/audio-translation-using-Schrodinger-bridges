"""Quick-launch wrapper: python scripts/run_train.py"""
import subprocess, sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
subprocess.run([sys.executable, "-m", "training.train_sb",
                "--config", str(root/"configs/default.yaml")], cwd=root)
