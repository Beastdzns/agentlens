"""
AgentLens causal analysis (Phase 11).

This package turns a recorded trace into an explicit causal graph and, on top
of that graph, a grounded root-cause diagnosis:

- :mod:`agentlens.causal.graph` — causal graph construction and traversal.
- :mod:`agentlens.causal.analyzer` — root-cause analysis over a graph.
- :mod:`agentlens.causal.explain` — grounded, template-based explanations.

Everything here is deterministic and rule-based: the same trace always yields
the same graph and the same diagnosis.
"""

from agentlens.causal.analyzer import (
    AnalyzerConfig,
    RootCause,
    RootCauseAnalyzer,
)
from agentlens.causal.explain import explain
from agentlens.causal.graph import (
    CausalEdge,
    CausalGraph,
    CausalNode,
    CausalRelation,
    build_graph,
)

__all__ = [
    "AnalyzerConfig",
    "CausalEdge",
    "CausalGraph",
    "CausalNode",
    "CausalRelation",
    "RootCause",
    "RootCauseAnalyzer",
    "build_graph",
    "explain",
]
