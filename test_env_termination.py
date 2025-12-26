"""
Test script to check if MATE environment terminates correctly
"""
import numpy as np
from mate.environment import MultiAgentTracking

# Create environment
env = MultiAgentTracking(
    config='mate/assets/MATE-4v4-9.yaml',
    render_mode=None
)

print(f"Environment created: {env}")
print(f"Num cameras: {env.num_cameras}")
print(f"Num targets: {env.num_targets}")

# Test episode
obs, info = env.reset()
camera_obs, target_obs = obs

max_steps = 2000
step_count = 0
done = False

print(f"\nStarting episode test (max {max_steps} steps)...")

while not done and step_count < max_steps:
    # Random actions
    cam_actions = np.random.randn(env.num_cameras, 2)
    tar_actions = np.random.randn(env.num_targets, 2)
    
    actions = (cam_actions, tar_actions)
    next_obs, rewards, terminated, truncated, info = env.step(actions)
    
    step_count += 1
    done = terminated or truncated
    
    if step_count % 100 == 0:
        print(f"Step {step_count}: terminated={terminated}, truncated={truncated}, done={done}")
    
    if done:
        print(f"\nEpisode ended at step {step_count}")
        print(f"  Terminated: {terminated}")
        print(f"  Truncated: {truncated}")
        print(f"  Info: {info}")
        break

if not done:
    print(f"\nEpisode did NOT terminate after {step_count} steps!")
    print("This is the problem - environment never returns done=True")

env.close()
print("\nTest complete!")
