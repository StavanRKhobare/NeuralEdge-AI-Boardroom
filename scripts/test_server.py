import os
import sys

# add envs to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "envs")))

from board_sim_env.server.app import app
from board_sim_env.models import BoardSimAction
from fastapi.testclient import TestClient

client = TestClient(app)

def test_api():
    print("Testing /reset endpoint...")
    response = client.post("/reset")
    assert response.status_code == 200, f"Reset failed: {response.text}"
    reset_data = response.json()
    print("/reset succeeded!")
    
    print("Testing /schema endpoint...")
    schema_response = client.get("/schema")
    assert schema_response.status_code == 200, f"Schema failed: {schema_response.text}"
    print("/schema succeeded!")
    
    print("Testing /state endpoint...")
    state_response = client.get("/state")
    assert state_response.status_code == 200, f"State failed: {state_response.text}"
    print("/state succeeded!")

    print("Testing /step endpoint...")
    # Get options from reset observation
    obs = reset_data.get("observation", {})
    options = obs.get("options", [])
    if not options:
        print("Warning: No options found in observation.")
        return
        
    action_payload = {
        "action": {
            "decision": options[0],
            "coalition_pitch": "Let's align on this for maximum profit!"
        }
    }
    
    step_response = client.post("/step", json=action_payload)
    assert step_response.status_code == 200, f"Step failed: {step_response.text}"
    print("/step succeeded!")
    print("All endpoints passed successfully!")

if __name__ == "__main__":
    test_api()
