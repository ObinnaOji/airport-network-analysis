# Airport Network Analysis

Graph analysis of the global airport network using Python, NetworkX and scikit-learn.

## Project Overview

This project analyses the OpenFlights global airport dataset to uncover the structural properties of the worldwide aviation network. The analysis covers 6,072 airports and 37,042 routes.

## Techniques Used

- **Graph Construction** — Built a directed, unweighted graph using NetworkX
- **Breadth-First Search (BFS)** — Measured reachability from London Heathrow (LHR)
- **Degree Centrality** — Identified the most connected hub airports
- **Betweenness Centrality** — Identified the most critical transit airports
- **Community Detection** — Greedy modularity and spectral clustering to find regional groupings

## Key Findings

- LHR reaches 171 airports in a single flight and 1,791 with one stop
- Frankfurt (FRA) has the highest degree centrality — directly connected to 7.9% of all airports
- Paris CDG has the highest betweenness centrality — the most critical transit hub globally
- Community detection identified distinct regional clusters: Asia-Pacific, North America, Europe, and intercontinental hubs

## Setup

```bash
pip install networkx matplotlib pandas numpy scikit-learn
```

Download the dataset from Kaggle and place both CSV files in the same directory as the script:
- `airport__airport.csv`
- `airline_network.csv`

https://www.kaggle.com/datasets/alenreuel/airport-network

Then run:
```bash
python airport_network_analysis.py
```
