import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import Linear, TransformerConv, to_hetero


class GAT(nn.Module):
    """Graph encoder with stacked TransformerConv blocks and residual connections."""

    def __init__(self, hidden_channels: int, num_layers: int = 2, heads: int = 3, dropout: float = 0.1,
                 edge_dim: int = 4):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ELU()

        self.lin_in = Linear(-1, hidden_channels)

        self.convs = nn.ModuleList()
        self.residual_projs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            conv = TransformerConv(-1, hidden_channels, edge_dim=edge_dim, heads=heads, dropout=dropout)
            self.convs.append(conv)
            self.residual_projs.append(Linear(-1, hidden_channels * heads))
            self.norms.append(nn.LayerNorm(hidden_channels * heads))

    def forward(self, x, edge_index, edge_attr_dict):
        x = self.activation(self.lin_in(x))
        x = self.dropout(x)

        for conv, res_proj, norm in zip(self.convs, self.residual_projs, self.norms):
            out = conv(x, edge_index, edge_attr_dict)
            out = self.activation(out)
            res = res_proj(x)
            x = norm(out + res)
            x = self.dropout(x)

        return x


class Model(nn.Module):
    """Heterogeneous GAT-based scorer for machine-job execution edges."""

    def __init__(self, hidden_channels: int, metadata, num_layers: int = 2, heads: int = 3, dropout: float = 0.1,
                 edge_dim: int = 4):
        super().__init__()

        self.gnn = GAT(hidden_channels, num_layers=num_layers, heads=heads, dropout=dropout, edge_dim=edge_dim)
        self.gnn = to_hetero(self.gnn, metadata=metadata, aggr="mean")

        self.edge_mlp = nn.Sequential(
            Linear(-1, hidden_channels),
            nn.ELU(),
            nn.Dropout(dropout),
            Linear(-1, hidden_channels),
            nn.ELU(),
            nn.Dropout(dropout),
            Linear(-1, 1),
        )

    def forward(self, data: HeteroData):
        node_embeddings = self.gnn(data.x_dict, data.edge_index_dict, data.edge_attr_dict)

        edge_index = data.edge_index_dict[("machine", "exec", "job")]
        edge_attr = data.edge_attr_dict[("machine", "exec", "job")]

        machine_embeddings = node_embeddings["machine"][edge_index[0]]
        job_embeddings = node_embeddings["job"][edge_index[1]]

        edge_feat = torch.cat([machine_embeddings, edge_attr, job_embeddings], dim=-1)
        return self.edge_mlp(edge_feat)
