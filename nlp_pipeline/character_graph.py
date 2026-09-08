import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

sys.path.append(str(Path(__file__).parent.parent))
from parser.screenplay_parser import ScreenplayParser


def build_graph(scenes: list[dict], min_shared_scenes: int = 1) -> nx.Graph:
    """scenes: list of {'dialogue': [{'character': ...}, ...]} dicts
    (works with both raw parser output and corpus.jsonl's scene dicts)."""
    G = nx.Graph()
    edge_scene_counts: dict[tuple, int] = defaultdict(int)

    for scene in scenes:
        speakers = sorted({d["character"] for d in scene["dialogue"]})
        for char in speakers:
            G.add_node(char)
        for i in range(len(speakers)):
            for j in range(i + 1, len(speakers)):
                pair = (speakers[i], speakers[j])
                edge_scene_counts[pair] += 1

    for (a, b), count in edge_scene_counts.items():
        if count >= min_shared_scenes:
            G.add_edge(a, b, weight=count)

    return G


def compute_centrality(G: nx.Graph) -> dict:
    if G.number_of_nodes() == 0:
        return {}

    degree = dict(G.degree())
    weighted_degree = dict(G.degree(weight="weight"))
    betweenness = nx.betweenness_centrality(G, weight="weight")
    closeness = nx.closeness_centrality(G)

    results = {}
    for node in G.nodes():
        results[node] = {
            "degree": degree[node],
            "weighted_degree": weighted_degree[node],
            "betweenness_centrality": round(betweenness[node], 4),
            "closeness_centrality": round(closeness[node], 4),
        }
    return results


def plot_graph(G: nx.Graph, centrality: dict, title: str, out_path: Path):
    if G.number_of_nodes() == 0:
        print("  No characters to plot (empty graph).")
        return

    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.spring_layout(G, k=1.2 / (G.number_of_nodes() ** 0.4), seed=42, weight="weight")

    max_between = max((c["betweenness_centrality"] for c in centrality.values()), default=0) or 1
    node_sizes = [300 + 3000 * (centrality[n]["betweenness_centrality"] / max_between) for n in G.nodes()]
    node_colors = [centrality[n]["weighted_degree"] for n in G.nodes()]

    weights = [G[u][v]["weight"] for u, v in G.edges()]
    min_w = min(weights) if weights else 1
    max_w = max(weights) if weights else 1
    if max_w == min_w:
        edge_widths = [1.5 for _ in weights]
    else:
        edge_widths = [0.5 + 5.0 * (w - min_w) / (max_w - min_w) for w in weights]

    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.35, edge_color="#888888", ax=ax)
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                                     cmap="YlOrRd", ax=ax, edgecolors="#333333", linewidths=0.8)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", ax=ax)

    plt.colorbar(nodes, ax=ax, label="Weighted degree (total shared scenes)", shrink=0.7)
    ax.set_title(f"{title} -- character relationship network\n"
                 f"(node size = betweenness centrality / \"bridge\" role, "
                 f"color = total shared scenes)", fontsize=12)
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("script_path", nargs="?", help="Path to a .txt screenplay")
    ap.add_argument("--corpus", help="Path to corpus.jsonl (use with --title instead of script_path)")
    ap.add_argument("--title", help="Title to look up in --corpus")
    ap.add_argument("--min-shared-scenes", type=int, default=1,
                     help="Only draw an edge if characters share at least this many scenes (default 1)")
    args = ap.parse_args()

    if args.corpus:
        screenplays = [json.loads(l) for l in open(args.corpus, encoding="utf-8")]
        matches = [s for s in screenplays if args.title.lower() in s["title"].lower()]
        if not matches:
            print(f"No screenplay matching '{args.title}' found in {args.corpus}")
            sys.exit(1)
        sp = matches[0]
        title = sp["title"]
        scenes = sp["scenes"]
    elif args.script_path:
        parser = ScreenplayParser()
        parsed = parser.parse_file(args.script_path)
        title = parsed.title
        scenes = [{"dialogue": [{"character": d.character} for d in s.dialogue]} for s in parsed.scenes]
    else:
        print("Provide either a script_path or --corpus + --title")
        sys.exit(1)

    G = build_graph(scenes, min_shared_scenes=args.min_shared_scenes)
    centrality = compute_centrality(G)

    print(f"\n{'='*60}")
    print(f"  CHARACTER NETWORK -- {title}")
    print(f"{'='*60}")
    print(f"  Characters (nodes) : {G.number_of_nodes()}")
    print(f"  Relationships (edges): {G.number_of_edges()}")

    ranked = sorted(centrality.items(), key=lambda x: x[1]["weighted_degree"], reverse=True)
    print(f"\n  Top characters by weighted degree (total shared scenes):")
    for name, c in ranked[:10]:
        print(f"    {name:<20} weighted_degree={c['weighted_degree']:>4}  "
              f"betweenness={c['betweenness_centrality']:.3f}  "
              f"closeness={c['closeness_centrality']:.3f}")

    bridge_ranked = sorted(centrality.items(), key=lambda x: x[1]["betweenness_centrality"], reverse=True)
    print(f"\n  Top 'bridge' characters (highest betweenness -- connect otherwise separate groups):")
    for name, c in bridge_ranked[:5]:
        print(f"    {name:<20} betweenness={c['betweenness_centrality']:.3f}")

    safe_title = title.replace(" ", "_").replace("/", "_")
    out_json = Path(f"{safe_title}_character_graph.json")
    out_png = Path(f"{safe_title}_character_graph.png")

    out_json.write_text(json.dumps({
        "title": title,
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
        "centrality": centrality,
        "edges": [{"a": a, "b": b, "shared_scenes": G[a][b]["weight"]} for a, b in G.edges()],
    }, indent=2))
    print(f"\nSaved data to {out_json}")

    plot_graph(G, centrality, title, out_png)


if __name__ == "__main__":
    main()