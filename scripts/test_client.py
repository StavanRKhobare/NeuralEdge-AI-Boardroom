"""End-to-end client ↔ server smoke test.

Spawns the FastAPI server in a background subprocess, polls /health until
ready, then connects via the BoardSimEnv client and runs one full episode.

Run: `python scripts/test_client.py`
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
import time

import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENV_DIR = os.path.join(ROOT, "envs", "board_sim_env")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "envs"))  # makes `import board_sim_env` work
sys.path.insert(0, ENV_DIR)

PORT = 8011
HOST = "127.0.0.1"
BASE = f"http://{HOST}:{PORT}"


def wait_healthy(timeout_s: float = 20.0) -> bool:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            r = requests.get(f"{BASE}/health", timeout=1.0)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(0.5)
    return False


def main() -> int:
    cmd = [
        sys.executable, "-m", "uvicorn",
        "server.app:app",
        "--host", HOST, "--port", str(PORT),
        "--log-level", "warning",
    ]
    print(f"Starting server: {' '.join(cmd)}  (cwd={ENV_DIR})")
    proc = subprocess.Popen(cmd, cwd=ENV_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        if not wait_healthy():
            print("Server failed to become healthy in 20s. Server output:")
            try:
                out = proc.stdout.read(4000) if proc.stdout else b""
                print(out.decode("utf-8", errors="replace"))
            finally:
                pass
            return 1
        print(f"Server healthy at {BASE}")

        from board_sim_env import BoardSimAction, BoardSimEnv  # noqa

        with BoardSimEnv(base_url=BASE).sync() as env:
            result = env.reset(seed=7)
            obs = result.observation
            print(f"Reset OK. round={obs.round}  options={obs.options}")
            n = 0
            while not result.done and n < 12:
                action = BoardSimAction(decision=random.choice(obs.options))
                result = env.step(action)
                obs = result.observation
                print(
                    f"  R{obs.round-1}: reward={result.reward:+.2f}  "
                    f"score={obs.state['profitability_score']:.1f}  "
                    f"runway={obs.state['runway_months']:.1f}mo"
                )
                n += 1
        print("CLIENT TEST PASSED.")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
