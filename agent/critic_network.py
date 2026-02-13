from torch import nn
from agent.graph_encoder import GATEncoder

class CriticNetwork(nn.Module):
    def __init__(self, node_dim, embedding_dim, hidden_dim, n_layers, n_heads):
        super().__init__()

        self.init_embed = nn.Linear(node_dim, embedding_dim)

        # 1) GAT 기반 encoder: node 임베딩 계산
        self.encoder = GATEncoder(
            node_dim=node_dim,      # x의 차원 (여기선 4: [loc, loc_paired])
            embed_dim=embedding_dim,
            n_layers=n_layers,
            n_heads=n_heads
        )

        # 2) 그래프 풀링 + value head
        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        x: (N_total, node_dim)   PyG Data.x
        edge_index: (2, E)
        return: (1,) 혹은 (batch_size,)  value
        """
        if isinstance(x, dict):
            batch = x['data']  # DataBatch 추출
            print(f"Dict batch -> DataBatch: {type(batch)}")
        else:
            batch = x

        h = batch.x                            # (N_total, node_dim=4)
        h = self.init_embed(h)                 # (N_total, embedding_dim)
        node_emb = self.encoder(h, batch.edge_index)           # (N_total, embedding_dim)

        # 간단히 mean pooling (필요하면 sum/max로 바꿔도 됨)
        graph_emb = node_emb.mean(dim=0, keepdim=True)   # (1, embedding_dim)

        value = self.value_head(graph_emb)               # (1, 1)
        return value.squeeze(-1)                         # (1,)