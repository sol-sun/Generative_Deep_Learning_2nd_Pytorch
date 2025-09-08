"""
ValueFlow: 計算グラフ管理

このモジュールは、Valuator間の依存関係を管理する計算グラフを提供します。
軽量な実装で、networkxとの相互運用性を保ちながら、
Mermaid記法での可視化もサポートします。
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional, Set, Union
from uuid import UUID, uuid4

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None  # type: ignore

from pydantic import BaseModel, Field, field_validator, ConfigDict


class Edge(BaseModel):
    """グラフのエッジ（辺）を表すモデル"""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="エッジの一意ID"
    )
    src: str = Field(
        ...,
        min_length=1,
        description="ソースノードID"
    )
    dst: str = Field(
        ...,
        min_length=1,
        description="デスティネーションノードID"
    )
    label: Optional[str] = Field(
        None,
        max_length=100,
        description="エッジのラベル",
        examples=["roic", "wacc", "beta_calculation"]
    )
    weight: Optional[float] = Field(
        None,
        ge=0.0,
        description="エッジの重み",
        examples=[1.0, 0.5, 2.0]
    )
    edge_type: Optional[str] = Field(
        None,
        max_length=50,
        description="エッジのタイプ",
        examples=["computation", "dependency", "override"]
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="エッジの追加メタデータ"
    )
    
    @field_validator("src", "dst")
    @classmethod
    def validate_node_ids(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("node ID cannot be empty")
        return v.strip()


class Node(BaseModel):
    """グラフのノードを表すモデル"""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    id: str = Field(
        ...,
        min_length=1,
        description="ノードの一意ID"
    )
    label: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="ノードのラベル",
        examples=["operating_profit", "invested_capital", "roic"]
    )
    node_type: Optional[str] = Field(
        None,
        max_length=50,
        description="ノードのタイプ",
        examples=["input", "computation", "output", "override"]
    )
    value_type: Optional[str] = Field(
        None,
        max_length=50,
        description="値の型",
        examples=["float", "series", "dataframe"]
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="ノードの追加メタデータ"
    )
    
    @field_validator("id", "label")
    @classmethod
    def validate_strings(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("ID and label cannot be empty")
        return v.strip()


class ValuationGraph(BaseModel):
    """計算グラフ（軽量実装）"""
    
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"
    )
    
    # 内部データ構造
    _nodes: Dict[str, Node] = Field(
        default_factory=dict,
        description="ノード辞書"
    )
    _edges: Dict[str, Edge] = Field(
        default_factory=dict,
        description="エッジ辞書"
    )
    _outgoing: Dict[str, Set[str]] = Field(
        default_factory=dict,
        description="各ノードからの出エッジ"
    )
    _incoming: Dict[str, Set[str]] = Field(
        default_factory=dict,
        description="各ノードへの入エッジ"
    )
    
    def add_node(
        self, 
        node_id: str, 
        label: Optional[str] = None,
        node_type: Optional[str] = None,
        value_type: Optional[str] = None,
        **metadata: Any
    ) -> None:
        """ノードを追加"""
        if node_id in self._nodes:
            # 既存ノードの更新
            existing = self._nodes[node_id]
            existing.label = label or existing.label
            existing.node_type = node_type or existing.node_type
            existing.value_type = value_type or existing.value_type
            existing.metadata.update(metadata)
        else:
            # 新規ノードの作成
            node = Node(
                id=node_id,
                label=label or node_id,
                node_type=node_type,
                value_type=value_type,
                metadata=metadata
            )
            self._nodes[node_id] = node
            self._outgoing[node_id] = set()
            self._incoming[node_id] = set()
    
    def has_node(self, node_id: str) -> bool:
        """ノードの存在確認"""
        return node_id in self._nodes
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """ノードを取得"""
        return self._nodes.get(node_id)
    
    def add_edge(
        self, 
        src: str, 
        dst: str, 
        label: Optional[str] = None,
        weight: Optional[float] = None,
        edge_type: Optional[str] = None,
        **metadata: Any
    ) -> str:
        """エッジを追加"""
        if src not in self._nodes:
            raise KeyError(f"Source node '{src}' does not exist")
        if dst not in self._nodes:
            raise KeyError(f"Destination node '{dst}' does not exist")
        
        edge_id = f"{src}->{dst}"
        if edge_id in self._edges:
            # 既存エッジの更新
            existing = self._edges[edge_id]
            existing.label = label or existing.label
            existing.weight = weight or existing.weight
            existing.edge_type = edge_type or existing.edge_type
            existing.metadata.update(metadata)
        else:
            # 新規エッジの作成
            edge = Edge(
                src=src,
                dst=dst,
                label=label,
                weight=weight,
                edge_type=edge_type,
                metadata=metadata
            )
            self._edges[edge_id] = edge
            self._outgoing[src].add(edge_id)
            self._incoming[dst].add(edge_id)
        
        return edge_id
    
    def has_edge(self, src: str, dst: str) -> bool:
        """エッジの存在確認"""
        return f"{src}->{dst}" in self._edges
    
    def get_edge(self, src: str, dst: str) -> Optional[Edge]:
        """エッジを取得"""
        return self._edges.get(f"{src}->{dst}")
    
    def predecessors(self, node_id: str) -> Iterable[str]:
        """前駆ノードを取得"""
        if node_id not in self._incoming:
            return []
        for edge_id in self._incoming[node_id]:
            edge = self._edges[edge_id]
            yield edge.src
    
    def successors(self, node_id: str) -> Iterable[str]:
        """後続ノードを取得"""
        if node_id not in self._outgoing:
            return []
        for edge_id in self._outgoing[node_id]:
            edge = self._edges[edge_id]
            yield edge.dst
    
    def ancestors(self, node_id: str) -> Set[str]:
        """祖先ノードを取得（再帰的）"""
        visited: Set[str] = set()
        stack = [node_id]
        
        while stack:
            current = stack.pop()
            for pred in self.predecessors(current):
                if pred not in visited:
                    visited.add(pred)
                    stack.append(pred)
        
        return visited
    
    def descendants(self, node_id: str) -> Set[str]:
        """子孫ノードを取得（再帰的）"""
        visited: Set[str] = set()
        stack = [node_id]
        
        while stack:
            current = stack.pop()
            for succ in self.successors(current):
                if succ not in visited:
                    visited.add(succ)
                    stack.append(succ)
        
        return visited
    
    def subgraph_nodes(self, root_id: str) -> Set[str]:
        """サブグラフのノードを取得（root_idから到達可能なすべてのノード）"""
        nodes = {root_id}
        nodes.update(self.ancestors(root_id))
        nodes.update(self.descendants(root_id))
        return nodes
    
    def topological_sort(self) -> list[str]:
        """トポロジカルソート"""
        in_degree = {node_id: len(list(self.predecessors(node_id))) 
                    for node_id in self._nodes}
        
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            for successor in self.successors(current):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)
        
        if len(result) != len(self._nodes):
            raise ValueError("Graph contains cycles")
        
        return result
    
    def is_acyclic(self) -> bool:
        """グラフが非循環かどうかを確認"""
        try:
            self.topological_sort()
            return True
        except ValueError:
            return False
    
    def to_networkx(self) -> Any:
        """NetworkXグラフに変換"""
        if not HAS_NETWORKX:
            raise RuntimeError("networkx is not installed. Install with: pip install networkx")
        
        g = nx.DiGraph()
        
        # ノードを追加
        for node in self._nodes.values():
            g.add_node(
                node.id,
                label=node.label,
                node_type=node.node_type,
                value_type=node.value_type,
                **node.metadata
            )
        
        # エッジを追加
        for edge in self._edges.values():
            g.add_edge(
                edge.src,
                edge.dst,
                label=edge.label,
                weight=edge.weight,
                edge_type=edge.edge_type,
                **edge.metadata
            )
        
        return g
    
    def to_mermaid(self, root_id: Optional[str] = None) -> str:
        """Mermaid記法に変換"""
        lines = ["flowchart TD"]
        
        # 表示するノードを決定
        if root_id is not None:
            if root_id not in self._nodes:
                raise KeyError(f"Root node '{root_id}' does not exist")
            nodes_to_show = self.subgraph_nodes(root_id)
        else:
            nodes_to_show = set(self._nodes.keys())
        
        # ノード定義
        for node_id in nodes_to_show:
            node = self._nodes[node_id]
            safe_id = node_id.replace("-", "_").replace(".", "_")
            label = node.label.replace('"', "'")  # Mermaidのエスケープ
            lines.append(f'  {safe_id}["{label}"]')
        
        # エッジ定義
        for edge in self._edges.values():
            if edge.src in nodes_to_show and edge.dst in nodes_to_show:
                src_safe = edge.src.replace("-", "_").replace(".", "_")
                dst_safe = edge.dst.replace("-", "_").replace(".", "_")
                
                if edge.label:
                    label_safe = edge.label.replace('"', "'")
                    lines.append(f'  {src_safe} -->|"{label_safe}"| {dst_safe}')
                else:
                    lines.append(f'  {src_safe} --> {dst_safe}')
        
        return "\n".join(lines)
    
    def to_json(self) -> str:
        """JSON形式に変換"""
        data = {
            "nodes": [node.model_dump() for node in self._nodes.values()],
            "edges": [edge.model_dump() for edge in self._edges.values()]
        }
        return json.dumps(data, indent=2, ensure_ascii=False)
    
    def from_json(self, json_str: str) -> None:
        """JSON形式から復元"""
        data = json.loads(json_str)
        
        # ノードを復元
        self._nodes.clear()
        for node_data in data.get("nodes", []):
            node = Node.model_validate(node_data)
            self._nodes[node.id] = node
            self._outgoing[node.id] = set()
            self._incoming[node.id] = set()
        
        # エッジを復元
        self._edges.clear()
        for edge_data in data.get("edges", []):
            edge = Edge.model_validate(edge_data)
            edge_id = f"{edge.src}->{edge.dst}"
            self._edges[edge_id] = edge
            self._outgoing[edge.src].add(edge_id)
            self._incoming[edge.dst].add(edge_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """グラフの統計情報を取得"""
        return {
            "node_count": len(self._nodes),
            "edge_count": len(self._edges),
            "is_acyclic": self.is_acyclic(),
            "max_in_degree": max(
                (len(list(self.predecessors(node_id))) for node_id in self._nodes),
                default=0
            ),
            "max_out_degree": max(
                (len(list(self.successors(node_id))) for node_id in self._nodes),
                default=0
            ),
            "node_types": {
                node_type: sum(1 for node in self._nodes.values() 
                              if node.node_type == node_type)
                for node_type in set(node.node_type for node in self._nodes.values() 
                                   if node.node_type is not None)
            },
            "edge_types": {
                edge_type: sum(1 for edge in self._edges.values() 
                              if edge.edge_type == edge_type)
                for edge_type in set(edge.edge_type for edge in self._edges.values() 
                                   if edge.edge_type is not None)
            }
        }