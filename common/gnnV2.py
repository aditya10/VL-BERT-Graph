import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=128,
        n_hidden_layers=2,
        is_bias=False,
        use_bn=False
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.use_bn = use_bn

        self.to_hidden = nn.Linear(self.input_size, self.hidden_size, bias=is_bias)
        self.linears = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.hidden_size, bias=is_bias)
                for _ in range(self.n_hidden_layers - 1)
            ]
        )
        self.out = nn.Linear(self.hidden_size, self.output_size, bias=is_bias)

        if self.use_bn:
            self.bn = nn.BatchNorm1d(self.output_size)

    def forward(self, x):
        out = self.to_hidden(x)
        out = nn.ReLU()(out)

        for linear in self.linears:
            out = linear(out)
            out = nn.ReLU()(out)

        out = self.out(out)
        if self.use_bn:
            out = self.bn(out.permute(0, 2, 1)).permute(0, 2, 1)
        
        return out

class GNN_layer(nn.Module):
    def __init__(self,  
               node_feature_dim, 
               edge_feature_dim, 
               message_length, 
               shrink_node_feature_dim,
               node_shrink_network,
               message_gen_network, 
               node_update_network, 
               edge_update_network,
               edge_update_weight=0.2
               ):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.message_length = message_length
        self.shrink_node_feature_dim = shrink_node_feature_dim
        self.edge_update_weight = edge_update_weight

        self.node_shrink_layer = node_shrink_network
        self.message_gen_network = message_gen_network
        self.node_update_network = node_update_network
        self.edge_update_network = edge_update_network

    def forward(self, node_features, edge_features):

        num_nodes = node_features.size(1)
        batch_size = node_features.size(0)

        # Generate a message by each node by shriking its feature
        shrink_node_features = self.node_shrink_layer(node_features) # [batch_size, num_nodes, shrink_node_feature_dim]
        # Repeat this message for each edge, since it is fully connected, all edges
        shrink_node_features_repeat_neighbors = shrink_node_features.repeat(1, num_nodes, 1).view(batch_size, num_nodes, num_nodes, self.shrink_node_feature_dim)
        # Apply the edge features. Edge features are 1-d and can be multipled directly, masked values are 0 (i.e. no contribution from masked nodes)
        messages_all = shrink_node_features_repeat_neighbors * edge_features
        # Get the final message as the mean of all messages
        message = torch.mean(messages_all, dim=2) #[B, L, message_length]
        # Apply an MLP to the message. This also performs batch norm.
        message = self.message_gen_network(message) # [B, L, message_length]
        # Update the node features
        node_features_updated = self.node_update_network(torch.cat((node_features, message), dim=2)) # [B, L, node_feature_dim]
        node_features = node_features + node_features_updated

        # For updating the edge features
        # Get all pairs of [node_i, node_j, edge_ij]
        shrink_node_features = self.node_shrink_layer(node_features)
        shrink_node_features_repeat_self = shrink_node_features.repeat(1, 1, num_nodes).view(batch_size, num_nodes, num_nodes, self.shrink_node_feature_dim)
        shrink_node_features_repeat_neighbors = shrink_node_features.repeat(1, num_nodes, 1).view(batch_size, num_nodes, num_nodes, self.shrink_node_feature_dim)
        input_to_edge_gen = torch.cat((shrink_node_features_repeat_self, shrink_node_features_repeat_neighbors, edge_features), dim=3)
        # Generate new edge features using an MLP
        new_edge_features = self.edge_update_network(input_to_edge_gen) #[B, L, L, De]
        # Mask out the edges that should be 0, and apply softmax
        edge_feature_mask = (edge_features>0).float()*self.edge_update_weight
        new_edge_features = new_edge_features*edge_feature_mask

        # Update the edge features
        edge_features = edge_features + new_edge_features

        return node_features, edge_features

class GNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.node_feature_dim = config.hidden_size # 768
        self.edge_feature_dim = config.edge_size # 1
        self.message_length = config.message_length # 128
        self.num_GNN_layers = config.num_GNN_layers # 4
        self.shrink_node_feature_dim = config.shrink_node_feature_dim # 128
        self.build_edge_mask = (config.gnn_edge_mask_layers != '')

        self.node_shrink_network = nn.Linear(self.node_feature_dim, self.shrink_node_feature_dim)
        self.message_gen_network = MLP(input_size=self.shrink_node_feature_dim, output_size=self.message_length, use_bn=False)
        self.node_update_network = MLP(input_size=self.node_feature_dim+self.message_length, output_size=self.node_feature_dim)
        self.edge_update_network = MLP(input_size=self.shrink_node_feature_dim*2+self.edge_feature_dim, output_size=self.edge_feature_dim)
        
        GNN_layers = []
        for i in range(self.num_GNN_layers):
            GNN_layers.append(GNN_layer(
                self.node_feature_dim, 
                self.edge_feature_dim, 
                self.message_length, 
                self.shrink_node_feature_dim,
                self.node_shrink_network,
                self.message_gen_network, 
                self.node_update_network, 
                self.edge_update_network))
            
        self.GNN_layers = nn.ModuleList(GNN_layers)
        
        if self.build_edge_mask:
            self.edge_mask_layer = nn.Linear(self.edge_feature_dim, config.num_attention_heads)

    # node_features: [B, N, 768], edge_features: [B, N, N, 1] 
    def forward(self, node_features, edge_features):
        new_node_features, new_edge_features = node_features, edge_features
        for GNN_layer in self.GNN_layers:
            new_node_features, new_edge_features = GNN_layer(new_node_features, new_edge_features)

        if self.build_edge_mask:
            new_edge_features = self.edge_mask_layer(new_edge_features)

        return new_node_features, new_edge_features
