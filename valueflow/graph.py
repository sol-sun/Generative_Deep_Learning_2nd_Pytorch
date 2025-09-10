"""
ValueFlow: 計算グラフ管理

このモジュールは、Valuatorインスタンス間の依存関係を管理するための
軽量実装であるValuationGraphを提供します。Mermaid記法による可視化と
NetworkXとの相互運用性をサポートします。
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Set

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
)

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None


class Edge(BaseModel):
    """ValuationGraphのエッジを表します。"""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")
    
    src: str = Field(..., min_length=1, description="Source node ID.")
    dst: str = Field(..., min_length=1, description="Destination node ID.")
    label: Optional[str] = Field(None, max_length=100, description="Edge label.")
    weight: Optional[float] = Field(None, ge=0.0, description="Edge weight.")
    edge_type: Optional[str] = Field(None, max_length=50, description="Type of the edge.", examples=["computation", "override"])
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional edge metadata.")


class Node(BaseModel):
    """ValuationGraphのノードを表します。"""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    id: str = Field(..., min_length=1, description="Unique node ID.")
    label: str = Field(..., min_length=1, max_length=100, description="Node label.")
    node_type: Optional[str] = Field(None, max_length=50, description="Type of the node.", examples=["input", "computation"])
    value_type: Optional[str] = Field(None, max_length=50, description="Type of the value held by the node.", examples=["float", "series"])
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional node metadata.")


class ValuationGraph:
    """計算依存関係を表す軽量有向グラフ。"""

    def __init__(self) -> None:
        self._nodes: Dict[str, Node] = {}
        self._edges: Dict[str, Edge] = {}
        self._outgoing: Dict[str, Set[str]] = {}  # node_idをedge_idsのセットにマッピング
        self._incoming: Dict[str, Set[str]] = {}  # node_idをedge_idsのセットにマッピング

    def add_node(self, node: Node) -> None:
        """グラフにノードを追加または更新します。

        Args:
            node: 追加するNodeオブジェクト。
        """
        if node.id not in self._nodes:
            self._outgoing[node.id] = set()
            self._incoming[node.id] = set()
        self._nodes[node.id] = node

    def has_node(self, node_id: str) -> bool:
        """ノードがグラフに存在するかチェックします。"""
        return node_id in self._nodes

    def get_node(self, node_id: str) -> Optional[Node]:
        """IDでノードを取得します。"""
        return self._nodes.get(node_id)

    def add_edge(self, edge: Edge) -> None:
        """グラフにエッジを追加または更新します。

        Args:
            edge: 追加するEdgeオブジェクト。

        Raises:
            KeyError: ソースまたは宛先ノードが存在しない場合。
        """
        if edge.src not in self._nodes:
            raise KeyError(f"Source node '{edge.src}' does not exist")
        if edge.dst not in self._nodes:
            raise KeyError(f"Destination node '{edge.dst}' does not exist")

        edge_id = f"{edge.src}->{edge.dst}"
        self._edges[edge_id] = edge
        self._outgoing[edge.src].add(edge_id)
        self._incoming[edge.dst].add(edge_id)

    def predecessors(self, node_id: str) -> Iterable[str]:
        """ノードの前駆ノードのイテレータを返します。"""
        if node_id in self._incoming:
            for edge_id in self._incoming[node_id]:
                yield self._edges[edge_id].src

    def successors(self, node_id: str) -> Iterable[str]:
        """ノードの後続ノードのイテレータを返します。"""
        if node_id in self._outgoing:
            for edge_id in self._outgoing[node_id]:
                yield self._edges[edge_id].dst

    def ancestors(self, node_id: str) -> Set[str]:
        """ノードのすべての祖先を再帰的に見つけます。"""
        visited: Set[str] = set()
        stack = list(self.predecessors(node_id))
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                stack.extend(self.predecessors(current))
        return visited

    def to_mermaid(self, root_id: Optional[str] = None) -> str:
        """グラフ可視化のためのMermaid記法文字列を生成します。

        Args:
            root_id: 提供された場合、ルートとその祖先を含む
                     サブグラフの図を生成します。

        Returns:
            Mermaidフローチャート記法の文字列。
        """
        lines = ["flowchart TD"]
        nodes_to_show = self._nodes.keys()
        if root_id:
            if root_id not in self._nodes:
                raise KeyError(f"Root node '{root_id}' does not exist")
            nodes_to_show = self.ancestors(root_id) | {root_id}

        for node_id in nodes_to_show:
            node = self._nodes[node_id]
            safe_label = node.label.replace('"', '#quot;')
            lines.append(f'  {node.id}["{safe_label}"]')
        
        for edge in self._edges.values():
            if edge.src in nodes_to_show and edge.dst in nodes_to_show:
                if edge.label:
                    safe_label = edge.label.replace('"', '#quot;')
                    lines.append(f'  {edge.src} -->|"{safe_label}"| {edge.dst}')
                else:
                    lines.append(f'  {edge.src} --> {edge.dst}')

        return "\n".join(lines)
    
    def to_networkx(self) -> Any:
        """グラフをNetworkX DiGraphに変換します。

        Returns:
            計算グラフを表すnetworkx.DiGraphオブジェクト。
        
        Raises:
            ImportError: networkxがインストールされていない場合。
        """
        if not HAS_NETWORKX:
            raise ImportError("networkx is not installed. Please install it with: pip install networkx")
        
        g = nx.DiGraph()
        for node in self._nodes.values():
            g.add_node(node.id, **node.model_dump())
        for edge in self._edges.values():
            g.add_edge(edge.src, edge.dst, **edge.model_dump())
        return g