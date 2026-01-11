#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: example_train_drugs.py
Author: Sisi Shao

[NOTE FOR REPRODUCIBILITY]
This script demonstrates the drug embedding training pipeline used in the study.
It builds a Drug–Gene bipartite subgraph and trains a VGAE model to obtain node
embeddings. The exported embeddings are restricted to Drug nodes.

As described in the manuscript:
- Gene embeddings are derived from GraphSAGE in a separate script.
- Drug embeddings are derived from VGAE (this script).

[DATABASE NOTE]
The original study used Neo4j. The current AlzKB backend has migrated to Memgraph.
This script uses the standard Bolt protocol and is compatible with both.

[SCALABILITY NOTE]
This is an example script. It intentionally uses identity features (one-hot) and
simple edge reconstruction. For large graphs, consider using real node features
and/or negative sampling / minibatch training.
"""

import sys
import torch
import pandas as pd
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import VGAE, GCNConv
from neo4j import GraphDatabase

# ============================
# 1. CONFIGURATION
# ============================

NEO4J_URI = "neo4j+s://neo4j.alzkb.ai:7687"

# Demo subset for speed. Replace with your full list if you want the full demo.
TARGET_AD_GENE_IDS = [
    1, 9, 15, 19, 20, 28, 42, 45, 46, 47, 48, 49,
    32764, 32765
]

OUT_CHANNELS = 16
EPOCHS = 50
LR = 0.01
BETA_KL = 1.0  # weight on KL term (kept standard)
BETA_CLUSTER = 1.0  # weight on clustering term

OUT_CSV = "embeddings_drug_vgae_ADcluster.csv"

# ============================
# 2. DATABASE CONNECTION
# ============================

class LoadGraph:
    def __init__(self, uri: str):
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

    def run_cypher(self, query: str):
        if not self.driver:
            return None
        with self.driver.session() as session:
            try:
                return session.run(query).data()
            except Exception as e:
                print(f"Query error: {e}")
                return None

# ============================
# 3. LOAD DRUG–GENE GRAPH
# ============================

alzkb = LoadGraph(NEO4J_URI)

# Drug–Gene edges restricted to (protein-coding) genes in the target list (demo-style).
drug_gene_query = f"""
MATCH (a:Drug)--(g:Gene)
WHERE g.typeOfGene = 'protein-coding' AND ID(g) IN {TARGET_AD_GENE_IDS}
RETURN DISTINCT ID(a) AS drug_id, ID(g) AS gene_id, a.commonName AS drug_name
"""

# Drugs directly connected to Alzheimer's Disease concept node (used for clustering target).
alz_drug_query = """
MATCH (a:Drug)-[]-(b {commonName:"Alzheimer's Disease"})
RETURN DISTINCT ID(a) AS drug_id, a.commonName AS drug_name, a.xrefDrugbank AS drugbank
"""

drug_gene_data = alzkb.run_cypher(drug_gene_query)
alz_drug_data = alzkb.run_cypher(alz_drug_query)
alzkb.close()

if not drug_gene_data:
    print("No Drug–Gene edges found for the given target genes. Exiting.")
    sys.exit(0)

# Collect node IDs (original DB IDs)
drug_ids = sorted({row["drug_id"] for row in drug_gene_data if row.get("drug_id") is not None})
gene_ids = sorted({row["gene_id"] for row in drug_gene_data if row.get("gene_id") is not None})

if len(drug_ids) == 0 or len(gene_ids) == 0:
    print("Insufficient nodes (no drugs or no genes) after filtering. Exiting.")
    sys.exit(0)

# Build a unified node index mapping (drugs + genes)
all_node_ids = drug_ids + gene_ids
node_id_to_idx = {nid: i for i, nid in enumerate(all_node_ids)}
idx_to_node_id = {i: nid for nid, i in node_id_to_idx.items()}

# Drug names map (best-effort, may have duplicates; we keep first)
drug_name_map = {}
for row in drug_gene_data:
    did = row.get("drug_id")
    dname = row.get("drug_name")
    if did is not None and did not in drug_name_map and dname is not None:
        drug_name_map[did] = dname

# Build edges (Drug -> Gene) as an undirected set for VGAE reconstruction stability
# (You can keep directed if desired, but undirected is common for VGAE demos.)
edges = []
for row in drug_gene_data:
    did = row["drug_id"]
    gid = row["gene_id"]
    if did in node_id_to_idx and gid in node_id_to_idx:
        u = node_id_to_idx[did]
        v = node_id_to_idx[gid]
        edges.append((u, v))
        edges.append((v, u))  # make symmetric

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

if edge_index.numel() == 0:
    print("No edges found after processing. Exiting.")
    sys.exit(0)

num_nodes = len(all_node_ids)

# Identity features (demo). For large graphs, replace with meaningful features.
x = torch.eye(num_nodes, dtype=torch.float32)

data = Data(x=x, edge_index=edge_index)

# Identify which indices correspond to drugs
is_drug = torch.zeros(num_nodes, dtype=torch.bool)
for did in drug_ids:
    is_drug[node_id_to_idx[did]] = True

# AD-drug indices for clustering target (best-effort intersection)
alz_drug_ids = []
alz_drug_name_map = {}
if alz_drug_data:
    for row in alz_drug_data:
        did = row.get("drug_id")
        if did is not None:
            alz_drug_ids.append(did)
        dname = row.get("drug_name")
        if did is not None and dname is not None:
            alz_drug_name_map[did] = dname

alz_drug_ids = sorted(set(alz_drug_ids))
alz_drug_idx = [node_id_to_idx[did] for did in alz_drug_ids if did in node_id_to_idx]
alz_drug_idx_tensor = torch.tensor(alz_drug_idx, dtype=torch.long)

print(f"Loaded nodes: {num_nodes} (drugs={len(drug_ids)}, genes={len(gene_ids)})")
print(f"Loaded edges (symmetrized): {data.edge_index.size(1)}")
print(f"AD drugs (for clustering): {len(alz_drug_idx)}")

# ============================
# 4. VGAE MODEL
# ============================

class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        h = torch.relu(self.conv1(x, edge_index))
        return self.conv_mu(h, edge_index), self.conv_logstd(h, edge_index)

encoder = Encoder(data.num_node_features, OUT_CHANNELS)
model = VGAE(encoder)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ============================
# 5. TRAINING LOOP
# ============================

model.train()
print("Starting Drug VGAE training...")

for epoch in range(EPOCHS):
    optimizer.zero_grad()

    z = model.encode(data.x, data.edge_index)

    # Standard VGAE objective
    loss_recon = model.recon_loss(z, data.edge_index)
    loss_kl = (1.0 / data.num_nodes) * model.kl_loss()

    # Clustering loss: encourage AD drugs to be close (if we have enough)
    if len(alz_drug_idx) > 1:
        alz_emb = z[alz_drug_idx_tensor]
        pairwise_dist = F.pairwise_distance(
            alz_emb.unsqueeze(1),
            alz_emb.unsqueeze(0),
            p=2
        )
        loss_cluster = torch.mean(torch.max(pairwise_dist, dim=1)[0])
    else:
        loss_cluster = torch.tensor(0.0)

    loss = loss_recon + (BETA_KL * loss_kl) + (BETA_CLUSTER * loss_cluster)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0 or epoch == EPOCHS - 1:
        print(
            f"Epoch {epoch:03d} | "
            f"recon={loss_recon.item():.4f} | kl={loss_kl.item():.4f} | "
            f"cluster={float(loss_cluster):.4f} | total={loss.item():.4f}"
        )

# ============================
# 6. EXPORT DRUG EMBEDDINGS
# ============================

model.eval()
with torch.no_grad():
    z = model.encode(data.x, data.edge_index)

# Export only drug nodes, in strict index order
drug_rows = []
for idx in range(num_nodes):
    if not bool(is_drug[idx]):
        continue
    node_id = idx_to_node_id[idx]  # original DB ID
    name = drug_name_map.get(node_id) or alz_drug_name_map.get(node_id) or ""
    emb = z[idx].cpu().numpy().tolist()
    drug_rows.append([node_id, name] + emb)

cols = ["node_id", "drug"] + [f"embedding_{i}" for i in range(OUT_CHANNELS)]
embeddings_df = pd.DataFrame(drug_rows, columns=cols)

embeddings_df.to_csv(OUT_CSV, index=False)
print(f"Saved drug embeddings to {OUT_CSV}")
