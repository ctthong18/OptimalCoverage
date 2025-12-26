from typing import Iterable, Tuple, List, Sequence
import numpy as np

def edge_list_to_edge_index(edge_list: Iterable[Tuple[int, int]]) -> np.ndarray:
    edges = list(edge_list)
    if len(edges) == 0:
        return np.zeros((2, 0), dtype=np.int64)
    src = np.array([int(u) for u, v in edges], dtype=np.int64)
    dst = np.array([int(v) for u, v in edges], dtype=np.int64)
    
    return np.stack([src, dst], axis=0)

def induced_edge_index(edge_list: Iterable[Tuple[int, int]], ordering: Sequence[int]) -> np.ndarray:
    ordering_set = set(int(x) for x in ordering)
    g2l = {int(g): i for i, g in enumerate(ordering)}
    edges = []
    for u, v in edge_list:
        if int(u) in ordering_set and int(v) in ordering_set:
            edges.append((g2l[int(u)], g2l[int(v)]))
            
    if len(edges) == 0:
        return np.zeros((2, 0), dtype=np.int64)
    
    src = np.array([e[0] for e in edges], dtype=np.int64)
    dst = np.array([e[1] for e in edges], dtype=np.int64)
    
    return np.stack([src, dst], axis=0)

        