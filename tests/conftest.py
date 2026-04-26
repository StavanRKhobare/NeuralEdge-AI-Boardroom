import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "envs"))
sys.path.insert(0, os.path.join(ROOT, "envs", "board_sim_env"))

os.environ.setdefault("BOARDSIM_PITCH_BACKEND", "tfidf")
