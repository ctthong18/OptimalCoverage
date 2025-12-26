from typing import List, Dict, Any, Optional
import random
import copy
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity: int = 200000):
        self.capacity = int(capacity)
        self.storage: List[Dict[str, Any]] = []
        self.pos = 0
    
    def add(self, states: Dict[int, np.ndarray], actions: Dict[int, np.ndarray],
        obs: Dict[int, np.ndarray], rewards: Dict[int, float],
        next_states: Dict[int, np.ndarray], next_obs: Dict[int, np.ndarray],
        done: bool
    ):
        transition = {
            "states": {int(k): np.asarray(v, dtype=np.float32) for k, v in states.items()},
            "actions": {int(k): np.asarray(v, dtype=np.float32) for k, v in actions.items()},
            "obs": {int(k): np.asarray(v, dtype=np.float32) for k, v in obs.items()},
            "rewards": {int(k): float(v) for k, v in rewards.items()},
            "next_states": {int(k): np.asarray(v, dtype=np.float32) for k, v in next_states.items()},
            "next_obs": {int(k): np.asarray(v, dtype=np.float32) for k, v in next_obs.items()},
            "done": bool(done)
        }

        if len(self.storage) < self.capacity:
            self.storage.append(transition)
        else:
            self.storage[self.pos] = transition
            self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size: int):
        batch = random.sample(self.storage, int(batch_size))
        return batch
    
    def __len__(self) -> int:
        return len(self.storage)