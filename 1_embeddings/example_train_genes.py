#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: example_train_genes.py
Author: Sisi Shao

[NOTE FOR REPRODUCIBILITY]
This script demonstrates the gene embedding training pipeline used in the study.
Only Gene nodes are included. Final gene embeddings are derived from GraphSAGE.

Drug embeddings (VGAE-based) are trained in a separate script, as described
in the manuscript.

[DATABASE NOTE]
The original study used Neo4j. The current AlzKB backend has migrated to Memgraph.
This script uses the Bolt protocol and is compatible with both.
"""

import sys
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from neo4j import GraphDatabase
import torch.nn.functional as F

# ============================
# 1. CONFIGURATION
# ============================

NEO4J_URI = "neo4j+s://neo4j.alzkb.ai:7687"

# Example target AD gene node IDs (from original study)
TARGET_AD_GENE_IDS = [
    1, 9, 15, 19, 20, 28, 42, 45, 46, 47, 48, 49,
    32764, 32765
]

# ============================
# 2. DATABASE CONNECTION
# ============================

class loadGraph:
    def __init__(self, uri):
        try:
            self.driver = GraphDatabase.driver(uri, auth=None)
            self.driver.verify_connectivity()
            print("Connected to Graph Database.")
        except Exception as e:
            print(f"Database connection failed: {e}")
            self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()

    def run_cypher(self, query):
        if not self.driver:
            return None
        with self.driver.session() as session:
            return session.run(query).data()

# ============================
# 3. LOAD GRAPH DATA
# ============================

alzkb = loadGraph(NEO4J_URI)

nodes_query = "MATCH (n:Gene) RETURN id(n) AS node_id"
edges_query = "MATCH (n1:Gene)-[r]->(n2:Gene) RETURN id(n1) AS source_id, id(n2) AS target_id"

nodes_data = alzkb.run_cypher(nodes_query)
edges_data = alzkb.run_cypher(edges_query)
alzkb.close()

if not nodes_data:
    print("No gene nodes found. Exiting.")
    sys.exit(0)

wanted_genes = set(TARGET_AD_GENE_IDS)

nodes_data = [n for n in nodes_data if n["node_id"] in wanted_genes]
edges_data = [
    e for e in edges_data
    if e["source_id"] in wanted_genes and e["target_id"] in wanted_genes
]

# ============================
# 4. BUILD PyG GRAPH
# ============================

sorted_node_ids = sorted([n["node_id"] for n in nodes_data])
node_id_mapping = {nid: i for i, nid in enumerate(sorted_node_ids)}
node_id_mapping_reverse = {i: nid for nid, i in node_id_mapping.items()}

edge_index = torch.tensor(
    [
        [node_id_mapping[e["source_id"]], node_id_mapping[e["target_id"]]]
        for e in edges_data
    ],
    dtype=torch.long
).t().contiguous()

num_nodes = len(sorted_node_ids)
x = torch.eye(num_nodes)  # identity features

data = Data(x=x, edge_index=edge_index)

alz_node_ids = [
    node_id_mapping[nid] for nid in TARGET_AD_GENE_IDS
    if nid in node_id_mapping
]
alz_node_id_tensor = torch.tensor(alz_node_ids, dtype=torch.long)

# ============================
# 5. MODEL DEFINITION (GraphSAGE)
# ============================

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, out_channels)
        self.conv2 = SAGEConv(out_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

model = GraphSAGE(data.num_node_features, out_channels=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ============================
# 6. TRAINING LOOP
# ============================

model.train()
losses = []

print("Starting Gene GraphSAGE training...")

for epoch in range(100):
    optimizer.zero_grad()
    z = model(data.x, data.edge_index)

    # Reconstruction loss (inner product)
    adj_pred = (z @ z.t()).sigmoid()
    loss_recon = F.binary_cross_entropy(
        adj_pred[data.edge_index[0], data.edge_index[1]],
        torch.ones(data.edge_index.size(1))
    )

    # Clustering loss for AD genes
    if len(alz_node_ids) > 1:
        alz_emb = z[alz_node_id_tensor]
        pairwise_dist = F.pairwise_distance(
            alz_emb.unsqueeze(1),
            alz_emb.unsqueeze(0),
            p=2
        )
        loss_cluster = torch.mean(torch.max(pairwise_dist, dim=1)[0])
    else:
        loss_cluster = 0.0

    loss = loss_recon + loss_cluster
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

# ============================
# 7. SAVE GENE EMBEDDINGS
# ============================

model.eval()
z = model(data.x, data.edge_index)

embeddings_df = pd.DataFrame(
    z.detach().numpy(),
    columns=[f"embedding_{i}" for i in range(z.shape[1])]
)

# strict alignment: row i ↔ node index i
embeddings_df["node_id"] = [
    node_id_mapping_reverse[i] for i in range(len(node_id_mapping_reverse))
]

embeddings_df = embeddings_df[
    ["node_id"] + [c for c in embeddings_df.columns if c != "node_id"]
]

embeddings_df.to_csv("embeddings_gene_sage_ADcluster.csv", index=False)
print("Saved gene embeddings to embeddings_gene_sage_ADcluster.csv")
