import sys
import os

# Add the server directory to path so we can import things
sys.path.append(os.path.join(os.getcwd(), 'envs', 'board_sim_env'))

try:
    from envs.board_sim_env.models import BoardSimAction, BoardSimObservation
    from envs.board_sim_env.server.board_sim_env_environment import BoardSimEnvironment

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
