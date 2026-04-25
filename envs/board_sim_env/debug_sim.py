import sys
import os

# Add current dir to path
sys.path.append(os.getcwd())

try:
    from models import BoardSimAction, BoardSimObservation
    from server.board_sim_env_environment import BoardSimEnvironment

    env = BoardSimEnvironment()
    print("Environment initialized.")
    
    # Try a step
    action = BoardSimAction(decision="differentiate", coalition_pitch="test")
    print(f"Action created: {action}")
    
    obs = env.step(action)
    print(f"Step successful. Round: {obs.round}")

except Exception as e:
    import traceback
    traceback.print_exc()
