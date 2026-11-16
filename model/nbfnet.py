import torch
from torch import nn

from . import tasks, layers
from model.base_nbfnet import BaseNBFNet


# NBFNet to work on the graph of relations with 4 fundamental interactions
# Doesn't have the final projection MLP from hidden dim -> 1, returns all node representations
# of shape [bs, num_rel, hidden]
class RelNBFNet(BaseNBFNet):

    def __init__(self, input_dim, hidden_dims, num_relation=4, **kwargs):
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False)
            )

        if self.concat_hidden:
            feature_dim = sum(hidden_dims) + input_dim
            self.mlp = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, input_dim)
            )

    def bellmanford(self, data, masked_fact, separate_grad=False):
        batch_size = masked_fact.shape[0]
        boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=masked_fact.device)
        for i in range(masked_fact.shape[1]):
            if i % 2 == 1:
                h_index = masked_fact[:, i]
                query = torch.ones(masked_fact.shape[0], self.dims[0], device=masked_fact.device, dtype=torch.float)
                index = h_index.unsqueeze(-1).expand_as(query)
                # initial (boundary) condition - initialize all node states as zeros
                # boundary = torch.zeros(data.num_nodes, *query.shape, device=h_index.device)
                # Indicator function: by the scatter operation we put ones as init features of source (index) nodes
                boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))

        # initialize initial nodes (relations of interest in the batcj) with all ones

        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=masked_fact.device)

        hiddens = []
        edge_weights = []
        layer_input = boundary
        # for i in range(data.edge_index.shape[1]):
        #     print(data.edge_index[:, i])

        for layer in self.layers:
            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            hidden = layer(layer_input, query, boundary, data.edge_index, data.edge_type, size, edge_weight)
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1)  # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
            output = self.mlp(output)
        else:
            output = hiddens[-1]

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, rel_graph, masked_fact):

        # message passing and updated node representations (that are in fact relations)
        output = self.bellmanford(rel_graph, masked_fact=masked_fact)["node_feature"]  # (batch_size, num_nodes, hidden_dim）

        return output


class EntityNBFNet(BaseNBFNet):

    def __init__(self, input_dim, hidden_dims, num_relation=1, **kwargs):

        # dummy num_relation = 1 as we won't use it in the NBFNet layer
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layers.GeneralizedRelattrConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False, project_relations=True)
            )

        feature_dim = (sum(hidden_dims) if self.concat_hidden else hidden_dims[-1]) + input_dim
        self.mlp = nn.Sequential()
        mlp = []
        for i in range(self.num_mlp_layers - 1):
            mlp.append(nn.Linear(feature_dim, feature_dim))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(feature_dim, 1))
        self.mlp = nn.Sequential(*mlp)

    def bellmanford(self, data, masked_fact, separate_grad=False):
        batch_size = masked_fact.shape[0]
        boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=masked_fact.device)
        for i in range(masked_fact.shape[1]):
            if i % 2 == 1:
                h_index = masked_fact[:, i]
                query = torch.ones(masked_fact.shape[0], self.dims[0], device=masked_fact.device, dtype=torch.float)
                index = h_index.unsqueeze(-1).expand_as(query)
                # initial (boundary) condition - initialize all node states as zeros
                # boundary = torch.zeros(data.num_nodes, *query.shape, device=h_index.device)
                # Indicator function: by the scatter operation we put ones as init features of source (index) nodes
                boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))

        # initialize initial nodes (relations of interest in the batcj) with all ones

        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=masked_fact.device)
        # edge_weight = torch.ones(data.num_edges, device=masked_fact.device)

        hiddens = []
        edge_weights = []
        layer_input = boundary
        if torch.isnan(layer_input).any():
            print("NaN detected in layer_input!")

        for layer in self.layers:
            # print("layer start")
            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            hidden = layer(layer_input, query, boundary, data.edge_index, data.edge_type, data.edge_attr, size, edge_weight)
            # if torch.isnan(hidden).any():
            #     print("NaN detected in hidden!")
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1)  # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
            output = self.mlp(output)
        else:
            output = hiddens[-1]

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, data, relation_representations, masked_fact):
        # h_index, t_index, r_index = batch.unbind(-1)

        # initial query representations are those from the relation graph
        self.query = relation_representations

        # initialize relations in each NBFNet layer (with uinque projection internally)
        for layer in self.layers:
            layer.relation = relation_representations

        shape = masked_fact.shape

        # message passing and updated node representations
        output = self.bellmanford(data, masked_fact)  # (num_nodes, batch_size, feature_dim）
        feature = output["node_feature"]
        return feature


class EntGraphNBFNet(BaseNBFNet):

    def __init__(self, input_dim, hidden_dims, num_relation=4, **kwargs):
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False)
            )

        if self.concat_hidden:
            feature_dim = sum(hidden_dims) + input_dim
            self.mlp = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, input_dim)
            )

    def bellmanford(self, data, masked_fact, separate_grad=False):
        batch_size = masked_fact.shape[0]
        boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=masked_fact.device)
        for i in range(masked_fact.shape[1]):
            if i % 2 == 0:
                h_index = masked_fact[:, i]
                query = torch.ones(masked_fact.shape[0], self.dims[0], device=masked_fact.device, dtype=torch.float)
                index = h_index.unsqueeze(-1).expand_as(query)
                # initial (boundary) condition - initialize all node states as zeros
                # boundary = torch.zeros(data.num_nodes, *query.shape, device=h_index.device)
                # Indicator function: by the scatter operation we put ones as init features of source (index) nodes
                boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))

        # initialize initial nodes (relations of interest in the batcj) with all ones

        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=masked_fact.device)

        hiddens = []
        edge_weights = []
        layer_input = boundary

        for layer in self.layers:
            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            hidden = layer(layer_input, query, boundary, data.edge_index, data.edge_type, size, edge_weight)
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1)  # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
            output = self.mlp(output)
        else:
            output = hiddens[-1]

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, rel_graph, masked_fact):

        # message passing and updated node representations (that are in fact relations)
        output = self.bellmanford(rel_graph, masked_fact=masked_fact)["node_feature"]  # (batch_size, num_nodes, hidden_dim）

        return output