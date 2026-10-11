"""Quick-launch wrapper: python scripts/run_explore.py"""
import subprocess, sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
subprocess.run([sys.executable, "-m", "data.explore",
                "--zh-tar",  str(root/"../Dataset/Chinese(China)/zh-CN.tar"),
                "--bn-tar",  str(root/"../Dataset/Bengali/bn.tar"),
                "--out-dir", str(root/"outputs/exploration")], cwd=root)
