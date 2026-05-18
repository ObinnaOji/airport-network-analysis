# airport_network_analysis.py
# Graph Analytics and Learning Assessment
# Airport Network Dataset (OpenFlights / Kaggle)
# Sections 1, 2, 3, and 4 combined into one file.

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering


# =============================================================================
# HELPER FUNCTIONS — Plotting
# =============================================================================

def draw_centrality_graph(G, scores, title="Centrality Graph", filename="centrality_graph.png"):
    """Draw graph where node size reflects centrality score."""
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    node_sizes = [5000 * scores.get(n, 0) for n in G.nodes()]
    nx.draw(G, pos, with_labels=True, node_size=node_sizes,
            node_color="orange", font_size=8, edge_color="gray")
    plt.title(title)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def draw_centrality_bar(scores, title="Centrality Scores", filename="centrality_bar.png", top_n=20):
    """Plot centrality scores as a bar chart."""
    plt.figure(figsize=(14, 6))
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    nodes  = [item[0] for item in sorted_scores]
    values = [item[1] for item in sorted_scores]
    plt.bar(nodes, values, color="steelblue")
    plt.title(title)
    plt.xlabel("Airport (IATA Code)")
    plt.ylabel("Centrality Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def draw_communities(G, communities, title="Community Detection", filename="communities.png"):
    """Colour nodes based on community membership."""
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=42)
    colors = ["red", "blue", "green", "orange", "purple",
              "yellow", "pink", "cyan", "brown", "gray"]
    for i, community in enumerate(communities):
        nx.draw_networkx_nodes(G, pos, nodelist=list(community),
                               node_color=colors[i % len(colors)], node_size=600)
    nx.draw_networkx_edges(G, pos, edge_color="lightgray", alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=7)
    plt.title(title)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def draw_graph(G, title="Graph Structure", filename="graph.png"):
    """Draw a simple graph showing nodes and edges."""
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color="lightblue",
            node_size=600, edge_color="gray", font_size=7)
    plt.title(title)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def draw_clusters(G, cluster_labels, node_list, title="Spectral Clustering", filename="clusters.png"):
    """Draw graph with nodes coloured by cluster membership."""
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color=cluster_labels,
            cmap=plt.cm.Set3, node_size=600, edge_color="gray", font_size=7)
    plt.title(title)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


# =============================================================================
# GRAPH CONSTRUCTION — used across all sections
# =============================================================================

def build_airport_graph():
    """Build directed airport graph from CSV files."""
    airports_df = pd.read_csv("airport__airport.csv")
    routes_df   = pd.read_csv("airline_network.csv")

    G = nx.DiGraph()

    for _, row in airports_df.iterrows():
        iata = str(row["IATA"]).strip()
        if iata and iata != "\\N":
            G.add_node(
                iata,
                name    = row["Name"],
                city    = row["City"],
                country = row["Country"],
                lat     = row["Latitude"],
                lon     = row["Longitude"]
            )

    skipped = 0
    for _, row in routes_df.iterrows():
        src = str(row["Source airport"]).strip()
        dst = str(row["Destination airport"]).strip()
        if src in G.nodes and dst in G.nodes:
            G.add_edge(src, dst)
        else:
            skipped += 1

    print(f"  Nodes added: {G.number_of_nodes()}")
    print(f"  Edges added: {G.number_of_edges()}")
    print(f"  Edges skipped (missing airport): {skipped}")
    return G


# =============================================================================
# SECTION 1 — Graph Modelling and Representation
# =============================================================================

def section1():
    print("\n" + "="*60)
    print("SECTION 1 — Graph Modelling and Representation")
    print("="*60)

    G = build_airport_graph()

    print(f"\nNumber of nodes (airports): {G.number_of_nodes()}")
    print(f"Number of edges (routes):   {G.number_of_edges()}")
    print(f"Graph is directed:          {G.is_directed()}")

    print("\nTop 10 airports by total degree:")
    degree_sorted = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:10]
    for airport, deg in degree_sorted:
        name  = G.nodes[airport].get("name", "Unknown")
        in_d  = G.in_degree(airport)
        out_d = G.out_degree(airport)
        print(f"  {airport} ({name[:30]}): total={deg}, in={in_d}, out={out_d}")

    print("\nEdge List (first 10 routes):")
    for src, dst in list(G.edges())[:10]:
        print(f"  {src} -> {dst}")

    print("\nAdjacency List sample for LHR:")
    if "LHR" in G.nodes:
        lhr_neighbours = list(G.neighbors("LHR"))
        print(f"  LHR -> {lhr_neighbours[:15]} ...")
        print(f"  LHR has {len(lhr_neighbours)} outgoing routes")

    print("\nGenerating subgraph visualisation (top 30 hubs)...")
    top_nodes = [n for n, d in sorted(G.degree(), key=lambda x: x[1], reverse=True)[:30]]
    subgraph  = G.subgraph(top_nodes)
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(subgraph, seed=42)
    degrees    = dict(subgraph.degree())
    node_sizes = [degrees[n] * 40 for n in subgraph.nodes()]
    nx.draw(subgraph, pos, with_labels=True, node_color="lightblue",
            node_size=node_sizes, edge_color="gray", arrows=True, font_size=8)
    plt.title("Airport Network - Top 30 Hub Airports (Subgraph)\nNode size = total degree (in + out routes)")
    plt.savefig("subgraph_top30.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: subgraph_top30.png")

    print("\nAdjacency matrix (top 10 airports only):")
    top10_nodes = [n for n, d in degree_sorted]
    subgraph10  = G.subgraph(top10_nodes)
    matrix      = nx.to_numpy_array(subgraph10, nodelist=top10_nodes)
    print("  Node order:", top10_nodes)
    print("  Shape:", matrix.shape)
    print(np.array2string(matrix.astype(int), separator=" "))

    return G


# =============================================================================
# SECTION 2 — Graph Algorithm: Breadth-First Search (BFS)
# =============================================================================

def section2(G):
    print("\n" + "="*60)
    print("SECTION 2 — Graph Algorithm: BFS")
    print("="*60)

    source = "LHR"
    print(f"\nSource airport: {source} - {G.nodes[source].get('name', '')}")

    bfs_layers = list(nx.bfs_layers(G, sources=[source]))

    print(f"\nBFS layers from {source}:")
    for i, layer in enumerate(bfs_layers[:5]):
        print(f"  Layer {i} ({i} flight(s)): {len(layer)} airports")
        if i == 1:
            print(f"    Sample direct destinations: {sorted(layer)[:10]}")

    print(f"\nShortest paths from {source} to selected airports:")
    print("-" * 60)
    destinations = ["JFK", "DXB", "SYD", "NBO", "GKA", "LAX", "SIN", "CPT"]
    for dest in destinations:
        if dest not in G.nodes:
            print(f"  {source} -> {dest}: Airport not in dataset")
            continue
        try:
            path   = nx.shortest_path(G, source=source, target=dest)
            length = nx.shortest_path_length(G, source=source, target=dest)
            dest_name = G.nodes[dest].get("name", dest)
            print(f"  {source} -> {dest} ({dest_name[:25]}): {length} flight(s)")
            print(f"    Route: {' -> '.join(path)}")
        except nx.NetworkXNoPath:
            print(f"  {source} -> {dest}: No reachable path in directed graph")

    print("\nVisualising BFS layers 0-2 (sample of 40 airports)...")
    layer0 = list(bfs_layers[0]) if len(bfs_layers) > 0 else []
    layer1 = list(bfs_layers[1])[:20] if len(bfs_layers) > 1 else []
    layer2 = list(bfs_layers[2])[:20] if len(bfs_layers) > 2 else []
    viz_nodes = layer0 + layer1 + layer2
    bfs_sub   = G.subgraph(viz_nodes)

    color_map = {}
    for n in viz_nodes:
        if n in layer0:
            color_map[n] = "gold"
        elif n in layer1:
            color_map[n] = "red"
        else:
            color_map[n] = "lightblue"

    node_colors = [color_map.get(n, "lightblue") for n in bfs_sub.nodes()]
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(bfs_sub, seed=42)
    nx.draw(bfs_sub, pos, with_labels=True, node_color=node_colors,
            node_size=700, edge_color="gray", arrows=True, font_size=8)
    plt.title("BFS from LHR - Layers 0, 1, 2\nGold=LHR (source) | Red=1 flight | Blue=2 flights")
    plt.savefig("bfs_layers_lhr.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: bfs_layers_lhr.png")

    print("\nBFS from Frankfurt (FRA) for comparison:")
    source2 = "FRA"
    if source2 in G.nodes:
        bfs_fra = list(nx.bfs_layers(G, sources=[source2]))
        for i, layer in enumerate(bfs_fra[:4]):
            print(f"  Layer {i} ({i} flight(s)): {len(layer)} airports")
        lhr_direct = len(bfs_layers[1]) if len(bfs_layers) > 1 else 0
        fra_direct = len(bfs_fra[1])    if len(bfs_fra) > 1 else 0
        print(f"\nComparison:")
        print(f"  LHR direct routes: {lhr_direct}")
        print(f"  FRA direct routes: {fra_direct}")

    print("\nBFS SUMMARY")
    print(f"  Source airport:        {source} (London Heathrow)")
    print(f"  Total reachable nodes: {sum(len(layer) for layer in bfs_layers)}")
    print(f"  Airports in 1 flight:  {len(bfs_layers[1]) if len(bfs_layers)>1 else 0}")
    print(f"  Airports in 2 flights: {len(bfs_layers[2]) if len(bfs_layers)>2 else 0}")
    print(f"  Airports in 3 flights: {len(bfs_layers[3]) if len(bfs_layers)>3 else 0}")


# =============================================================================
# SECTION 3 — Network Analytics: Degree and Betweenness Centrality
# =============================================================================

def section3(G):
    print("\n" + "="*60)
    print("SECTION 3 — Network Analytics")
    print("="*60)

    # --- Degree Centrality ---
    print("\nDegree Centrality Scores (Top 20 airports):")
    degree_centrality = nx.degree_centrality(G)
    sorted_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)

    for node, score in sorted_degree[:20]:
        name = G.nodes[node].get("name", "Unknown")
        print(f"  {node} ({name[:30]}) : {round(score, 4)}")

    top_node  = sorted_degree[0][0]
    top_score = sorted_degree[0][1]
    print(f"\nHighest degree centrality: {top_node} = {round(top_score, 4)}")
    print(f"  {top_node} connects to {round(top_score * 100, 1)}% of all airports in the network")

    print("\nGenerating degree centrality visualisations...")
    top30_deg     = [n for n, s in sorted_degree[:30]]
    sub30_deg     = G.subgraph(top30_deg)
    sub_deg_cent  = {n: degree_centrality[n] for n in top30_deg}

    draw_centrality_graph(
        sub30_deg, sub_deg_cent,
        title="Degree Centrality - Top 30 Airport Hubs\n(Node size = Degree Centrality score)",
        filename="degree_centrality_graph.png"
    )
    draw_centrality_bar(
        degree_centrality,
        title="Degree Centrality Scores - Top 20 Airports",
        filename="degree_centrality_bar.png",
        top_n=20
    )

    # --- Betweenness Centrality ---
    print("\nComputing betweenness centrality (using k=500 sample for speed)...")
    betweenness_centrality = nx.betweenness_centrality(G, k=500, normalized=True)
    sorted_between = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)

    print("\nBetweenness Centrality Scores (Top 20 airports):")
    for node, score in sorted_between[:20]:
        name = G.nodes[node].get("name", "Unknown")
        print(f"  {node} ({name[:30]}) : {round(score, 4)}")

    top_node_b  = sorted_between[0][0]
    top_score_b = sorted_between[0][1]
    print(f"\nHighest betweenness: {top_node_b} = {round(top_score_b, 4)}")
    print(f"  {top_node_b} lies on many shortest paths between airports.")
    print(f"  Removing {top_node_b} could significantly disrupt global connectivity.")

    print("\nGenerating betweenness centrality visualisations...")
    top30_bet    = [n for n, s in sorted_between[:30]]
    sub30_bet    = G.subgraph(top30_bet)
    sub_bet_cent = {n: betweenness_centrality[n] for n in top30_bet}

    draw_centrality_graph(
        sub30_bet, sub_bet_cent,
        title="Betweenness Centrality - Top 30 Airport Hubs\n(Node size = Betweenness score)",
        filename="betweenness_centrality_graph.png"
    )
    draw_centrality_bar(
        betweenness_centrality,
        title="Betweenness Centrality Scores - Top 20 Airports",
        filename="betweenness_centrality_bar.png",
        top_n=20
    )


# =============================================================================
# SECTION 4 — Mini Graph Project: Community Detection
# =============================================================================

def section4(G):
    print("\n" + "="*60)
    print("SECTION 4 — Community Detection")
    print("="*60)

    G_undirected = G.to_undirected()

    # --- Greedy Modularity ---
    print("\nExtracting top 150 airports by degree for community analysis...")
    top150 = [n for n, d in sorted(G.degree(), key=lambda x: x[1], reverse=True)[:150]]
    sub150 = G_undirected.subgraph(top150).copy()
    print(f"  Subgraph: {sub150.number_of_nodes()} airports, {sub150.number_of_edges()} connections")

    draw_graph(
        sub150,
        title="Airport Network - Top 150 Hubs (Before Community Detection)",
        filename="community_raw_graph.png"
    )

    print("\nRunning greedy modularity community detection...")
    communities = list(nx.algorithms.community.greedy_modularity_communities(sub150))

    print(f"\nCommunities Detected: {len(communities)}")
    for i, community in enumerate(communities):
        community_list = sorted(list(community))
        countries = []
        for airport in community_list:
            country = sub150.nodes[airport].get("country", "Unknown")
            if country not in countries:
                countries.append(country)
        print(f"\nCommunity {i+1} ({len(community_list)} airports):")
        print(f"  Airports: {community_list}")
        print(f"  Countries represented: {countries[:8]}")

    draw_communities(
        sub150, communities,
        title=f"Airport Network Community Detection\n{len(communities)} Communities Found (Greedy Modularity)",
        filename="community_detection_greedy.png"
    )

    print("\nCommunity size summary:")
    for i, c in enumerate(communities):
        print(f"  Community {i+1}: {len(c)} airports")
    print(f"\nLargest community:  {max(len(c) for c in communities)} airports")
    print(f"Smallest community: {min(len(c) for c in communities)} airports")

    # --- Spectral Clustering ---
    print("\nExtracting top 80 airports for spectral clustering...")
    top80 = [n for n, d in sorted(G.degree(), key=lambda x: x[1], reverse=True)[:80]]
    sub80 = G_undirected.subgraph(top80).copy()
    print(f"  Subgraph: {sub80.number_of_nodes()} airports, {sub80.number_of_edges()} connections")

    nodes            = list(sub80.nodes())
    adjacency_matrix = nx.to_numpy_array(sub80, nodelist=nodes)
    print(f"  Adjacency matrix shape: {adjacency_matrix.shape}")

    n_clusters = 5
    spectral   = SpectralClustering(n_clusters=n_clusters, affinity="precomputed",
                                    assign_labels="kmeans", random_state=42)
    print(f"\nApplying SpectralClustering with n_clusters={n_clusters}...")
    cluster_labels = spectral.fit_predict(adjacency_matrix)

    print("\nCluster Assignments:")
    clusters = {}
    for node, label in zip(nodes, cluster_labels):
        print(f"  {node} -> Cluster {label}")
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(node)

    print(f"\nCluster Summaries ({n_clusters} clusters):")
    for label in sorted(clusters.keys()):
        airports  = clusters[label]
        countries = list(set(sub80.nodes[a].get("country", "Unknown") for a in airports))
        print(f"\n  Cluster {label} ({len(airports)} airports):")
        print(f"    Airports: {sorted(airports)}")
        print(f"    Countries: {countries[:6]}")

    draw_clusters(
        sub80, list(cluster_labels), nodes,
        title=f"Airport Network - Spectral Clustering ({n_clusters} Clusters)\nTop 80 Hub Airports",
        filename="community_spectral_clustering.png"
    )

    print("\nRe-running with n_clusters=3 for comparison...")
    spectral3    = SpectralClustering(n_clusters=3, affinity="precomputed",
                                      assign_labels="kmeans", random_state=42)
    labels3      = spectral3.fit_predict(adjacency_matrix)

    draw_clusters(
        sub80, list(labels3), nodes,
        title="Airport Network - Spectral Clustering (3 Clusters)\nComparison run",
        filename="community_spectral_3clusters.png"
    )

    print("\nCluster Assignments (n_clusters=3):")
    for node, label in zip(nodes, labels3):
        print(f"  {node} -> Cluster {label}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Airport Network Analysis")
    print("Graph Analytics and Learning Assessment")
    print("="*60)

    G = section1()
    section2(G)
    section3(G)
    section4(G)

    print("\n" + "="*60)
    print("DONE — all sections complete.")
    print("="*60)
