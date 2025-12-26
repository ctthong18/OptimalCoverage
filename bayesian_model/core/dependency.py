from typing import Dict, Set, Iterable, Tuple
from collections import deque


"""
Tính toán các tập phụ thuộc giá trị I_i^Q và các tập phụ thuộc gradient I_i^GD

Triển khai:
    - Khả năng tiếp cận đầy đủ hoặc cắt cụt theo kappa
Output:
    - I_Q: dict agent -> set agents cần cho Q_i
    - I_GD: dict agent -> set agents phụ thuộc gradient
"""

def compute_value_dependency_sets(
    n_agents: int,
    folded_edge_list: Iterable[Tuple[int, int]],
    I_R: Dict[int, Set[int]] = None,
    kappa: int = None
) -> (Dict[int, Set[int]], Dict[int, Set[int]]):
    """
    folded_edge_list: danh sách cạnh của đồ thị gấp GF (u → v)
    I_R: phần thưởng phụ thuộc (agent → set). Nếu None thì tự suy ra từ GF.
    kappa: giới hạn độ sâu (nếu None thì không giới hạn)
    """

    # 1. Build adjacency list
    adj = {i: set() for i in range(n_agents)}
    for u, v in folded_edge_list:
        adj[int(u)].add(int(v))

    # 2. Nếu không có I_R, tự tạo: mỗi agent ảnh hưởng reward của chính nó,
    #   và mọi agent u ảnh hưởng reward của mọi v mà nó nối tới trong GF.
    if I_R is None:
        I_R = {i: {i} for i in range(n_agents)}
        for u in range(n_agents):
            for v in adj[u]:
                I_R[v].add(u)

    # 3. Khởi tạo I_Q trống
    I_Q: Dict[int, Set[int]] = {i: set() for i in range(n_agents)}

    # Build reverse adjacency (đi ngược chiều để BFS nguồn → target)
    rev = {i: set() for i in range(n_agents)}
    for u in range(n_agents):
        for v in adj[u]:
            rev[v].add(u)

    # 4. Tính I_Q cho từng agent
    for target in range(n_agents):

        start_nodes = I_R.get(target, {target})

        visited = set()
        q = deque()

        # Push sources
        for s in start_nodes:
            visited.add(int(s))
            q.append((int(s), 0))

        # BFS ngược
        while q:
            node, dist = q.popleft()
            I_Q[target].add(node)

            # Nếu có kappa và đã đạt limit thì không mở rộng
            if kappa is not None and dist >= kappa:
                continue

            # Duyệt ngược qua các cha (predecessors)
            for p in rev[node]:
                if p not in visited:
                    visited.add(p)
                    q.append((p, dist + 1))
        
        # Ensure each agent at least includes itself in I_Q (minimum dependency)
        if len(I_Q[target]) == 0:
            I_Q[target].add(target)

    # 5. Tính tập phụ thuộc gradient I_GD: i ảnh hưởng Q_j thì j ∈ I_GD[i]
    I_GD: Dict[int, Set[int]] = {i: set() for i in range(n_agents)}
    for j in range(n_agents):
        for i in I_Q[j]:
            I_GD[i].add(j)

    return I_Q, I_GD
