import os
from typing import List, Optional

import imageio.v2 as imageio
import numpy as np


def write_video(frames: List[np.ndarray], out_path: str, fps: int = 30):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with imageio.get_writer(out_path, fps=fps, codec='libx264', quality=8) as writer:
        for frame in frames:
            writer.append_data(frame)


def collect_episode_frames(env, max_steps: int = 2000) -> List[np.ndarray]:
    """Attempt to collect RGB frames from the environment with best-effort API coverage."""
    frames: List[np.ndarray] = []
    step = 0
    while step < max_steps:
        frame = None
        # Try common render APIs
        if hasattr(env, 'render'):
            try:
                frame = env.render(mode='rgb_array')  # gym-like
            except TypeError:
                try:
                    frame = env.render(return_image=True)  # custom API
                except Exception:
                    try:
                        frame = env.render()  # might already return array
                    except Exception:
                        frame = None
        if frame is not None and isinstance(frame, np.ndarray):
            frames.append(frame)
        else:
            # If no frame available, stop collecting
            break
        step += 1
    return frames

