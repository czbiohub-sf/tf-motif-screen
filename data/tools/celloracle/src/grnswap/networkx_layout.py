"""Compute one deterministic layout and label placement for both GRN panels."""

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def spring_layout(nodes, edges, random_seed, spring_spacing=1.5):
    """Lay out the sorted union graph, retaining seeds with no fitted edges."""
    graph = nx.Graph()
    graph.add_nodes_from(sorted(nodes.gene))
    graph.add_edges_from(
        (row.source, row.target)
        for row in edges.sort_values(["source", "target"]).itertuples(index=False)
        if row.source != row.target
    )
    positions = nx.spring_layout(
        graph,
        seed=random_seed,
        k=spring_spacing / math.sqrt(len(graph)),
        weight=None,
        iterations=200,
    )
    return {gene: tuple(map(float, xy)) for gene, xy in positions.items()}


def assign_coordinates(nodes, edges, *, seed=17, spring_spacing=1.5):
    """Place labels once so the two arms use identical node and label positions."""
    from adjustText import adjust_text

    positions = spring_layout(nodes, edges, seed, spring_spacing)
    for axis, index in [("x", 0), ("y", 1)]:
        nodes[axis] = nodes.gene.map(
            {gene: xy[index] for gene, xy in positions.items()}
        )
    fig, ax = plt.subplots(figsize=(8, 7))
    xy = nodes[["x", "y"]].to_numpy()
    low, high = xy.min(axis=0), xy.max(axis=0)
    margin = max(float((high - low).max()) * 0.2, 0.25)
    ax.set(
        xlim=(low[0] - margin, high[0] + margin),
        ylim=(low[1] - margin, high[1] + margin),
        aspect="equal",
    )
    markers = ax.scatter(nodes.x, nodes.y, s=np.where(nodes.is_seed, 150, 32))
    texts = [
        ax.text(
            row.x,
            row.y + 0.03,
            row.gene,
            fontsize=9 if row.is_seed else 8,
            ha="center",
            va="center",
        )
        for row in nodes.itertuples(index=False)
    ]
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        adjust_text(texts, x=nodes.x, y=nodes.y, objects=markers, ax=ax, iter_lim=180)
        nodes[["label_x", "label_y"]] = [text.get_position() for text in texts]
    finally:
        np.random.set_state(state)
        plt.close(fig)
