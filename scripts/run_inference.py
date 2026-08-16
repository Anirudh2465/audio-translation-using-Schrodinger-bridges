"""Quick-launch wrapper: python scripts/run_inference.py --input <audio> --checkpoint <ckpt>"""
import subprocess, sys, pathlib, argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="morphed.wav")
    args, unknown = parser.parse_known_args()

    root = pathlib.Path(__file__).resolve().parents[1]
    cmd = [sys.executable, "inference.py", "--input", args.input, 
           "--checkpoint", args.checkpoint, "--output", args.output] + unknown
           
    subprocess.run(cmd, cwd=root)

if __name__ == "__main__":
    main()
