"""
Simple test to isolate the training loop issue
"""
import numpy as np
import time
from mate.environment import MultiAgentTracking

print("Creating environment...")
env = MultiAgentTracking(
    config='mate/assets/MATE-4v4-9.yaml',
    render_mode='rgb_array'  # Use rgb_array instead of None
)

print(f"Environment: {env}")
print(f"Cameras: {env.num_cameras}, Targets: {env.num_targets}")

# Reset
obs, info = env.reset()
camera_obs, target_obs = obs

max_episode_steps = 2000
current_episode_steps = 0
episode_count = 0
total_steps = 0
max_total_steps = 5000

print(f"\nStarting training loop (max {max_total_steps} steps)...")
start_time = time.time()

for t in range(max_total_steps):
    # Random actions
    cam_actions = np.random.randn(env.num_cameras, 2)
    tar_actions = np.random.randn(env.num_targets, 2)
    
    actions = (cam_actions, tar_actions)
    next_obs, rewards, terminated, truncated, info = env.step(actions)
    
    done = terminated or truncated
    current_episode_steps += 1
    
    # Force truncation
    if current_episode_steps >= max_episode_steps:
        done = True
        print(f"  FORCE TRUNCATION at step {current_episode_steps}")
    
    if done:
        episode_count += 1
        reason = "terminated" if terminated else "truncated" if truncated else "forced"
        print(f"Episode {episode_count} ended ({reason}): {current_episode_steps} steps")
        
        # Reset
        obs, info = env.reset()
        camera_obs, target_obs = obs
        current_episode_steps = 0
    else:
        camera_obs, target_obs = next_obs
    
    # Log every 1000 steps
    if (t + 1) % 1000 == 0:
        elapsed = time.time() - start_time
        speed = (t + 1) / elapsed
        print(f"\nTimestep {t+1}/{max_total_steps} | Speed: {speed:.1f} steps/s | Episodes: {episode_count}")
        print(f"  Current episode: {current_episode_steps} steps")

print(f"\nTest complete! Total episodes: {episode_count}")
env.close()
