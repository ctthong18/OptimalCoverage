"""
MABN
Lưu trữ es (state), eo (observation), er (reward) dưới dạng đồ thị
sinh ra folded 2-time-step graph gf
trả về adjacency, parents, edge list giúp downstream
"""

from typing import Iterable, Dict, Set, Tuple, List
from collections import defaultdict

class MABN:
    def __init__(
        self,
        n_agents: int,
        ES: Iterable[Tuple[int, int]],
        EO: Iterable[Tuple[int, int]],
        ER: Iterable[Tuple[int, int]],
    ):
        """
        ES: cạnh liên kết trạng thái (u->v nếu u ảnh hưởng trạng thái v)
        EO: cạnh quan sát (u->v nếu v đượcq quan sát)
        ER: cạnh phần thưởng (u->v nếu u ảnh hưởng đến phần thưởng v)
        """
        self.n = int(n_agents)
        self.ES = list(ES)
        self.EO = list(EO)
        self.ER = list(ER)
        
        self.parents_s = self._build_parents(self.ES)
        self.parents_o = self._build_parents(self.EO)
        self.parents_r = self._build_parents(self.ER)
        
        self.folded = self._build_folded_graph()
        
    def _build_parents(self, edges: Iterable[Tuple[int, int]]) -> Dict[int, Set[int]]:
        parents = {i: set() for i in range(self.n)}
        for u, v in edges:
            parents[v].add(int(u))
        return parents
    
    def _build_folded_graph(self) -> Dict[int, Set[int]]:
        """
        Xây dựng đồ thị ảnh hưởng gấp GF để tính toán khả năng tiếp cận/phụ thuộc giá trị
            - Nếu u trong I_R[v] (u->v thuộc ER) thì u->v trong GF
            - Nếu u->w thuộc ES, w->v thuộc EO thì u->v
            - Gồm các cạnh EO
        """
        GF: Dict[int, Set[int]] = {i: set() for i in range(self.n)}
        
        for u, v in self.ER:
            GF[int(u)].add(int(v))
        
        ES_succ = {i: set() for i in range(self.n)}
        EO_succ = {i: set() for i in range(self.n)}
        
        for u, v in self.ES:
            ES_succ[int(u)].add(int(v))
        for u, v in self.EO:
            EO_succ[int(u)].add(int(v))
           
        for u in range(self.n):
            for w in ES_succ[u]:
                for v in EO_succ.get(w, []):
                    GF[u].add(v)
                for (a, b) in self.ER:
                    if int(a) == int(w):
                        GF[u].add(int(b))
        
        for u, v in self.EO:
            GF[int(u)].add(int(v))
        
        return GF
    
    def edge_list(self) -> List[Tuple[int, int]]:
        edges = []
        for u, s in self.folded.items():
            for v in s:
                edges.append((u, v))
        return edges
    
    def neighbors(self, agent: int) -> Set[int]:
        return set(self.folded.get(int(agent), set()))
    
    def predecessors(self, agent: int) -> Set[int]:
        preds = set()
        for u, s in self.folded.items():
            if int(agent) in s:
                preds.add(int(u))
        return preds
    
            