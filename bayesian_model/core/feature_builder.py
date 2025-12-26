from typing import Dict, Iterable, List, Sequence, Tuple
import numpy as np

def build_node_features(
    states: Dict[int, Sequence[float]],
    actions: Dict[int, Sequence[float]],
    ordering: Sequence[int],
    use_action: bool = True,
) -> np.ndarray:
    feat_list = []
    for j in ordering:
        s = np.asarray(states[int(j)], dtype=np.float32)
        if use_action:
            a = np.asarray(actions[int(j)], dtype=np.float32)
            f = np.concatenate([s, a], axis=0)
        else:
            f = s
        feat_list.append(f)
    
    # Handle empty ordering case - return empty array with correct shape
    if len(feat_list) == 0:
        # Return empty array with shape (0, feature_dim)
        # We need at least one sample to determine feature dimension
        return np.empty((0, 0), dtype=np.float32)
    
    return np.stack(feat_list, axis=0)

def build_agent_critic_input(
    i: int,
    I_Q_i: Iterable[int],
    global_states: Dict[int, Sequence[float]],
    global_actions: Dict[int, Sequence[float]],
    use_action: bool = True,
) -> Tuple[np.ndarray, List[int]]:
    ordering = sorted(list(I_Q_i))
    node_features = build_node_features(global_states, global_actions, ordering, use_action=use_action)
    
    return node_features, ordering
