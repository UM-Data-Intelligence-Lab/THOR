import copy

import torch
from model.nbfnet import RelNBFNet, EntityNBFNet, EntGraphNBFNet
from model.gran import GRANModel
import collections


class THOR(torch.nn.Module):
    def __init__(self, config, gran_config):
        super(THOR, self).__init__()
        self.meta_relation_num = config["meta_relation_num"]
        self.emb_dim = config["emb_dim"]
        self.layer_dim = config["layer_dim"]
        self.device = config["device"]
        self.max_seq_len = config["max_seq_len"]
        self.meta_relation_num = config["meta_relation_num"]
        self.relation_start = config["relation_start"]

        self.relNBF = RelNBFNet(self.emb_dim, self.layer_dim, self.relation_start)
        self.entNBF = EntGraphNBFNet(self.emb_dim, self.layer_dim, self.relation_start)
        # self.entNBF = EntityNBFNet(self.emb_dim, self.layer_dim)
        self.gran = GRANModel(gran_config)
        self.pad = torch.nn.Parameter(torch.rand([1, self.emb_dim], device=self.device, dtype=torch.float))
        self.mask = torch.nn.Parameter(torch.rand([1, self.emb_dim], device=self.device, dtype=torch.float))
        self.rel_pad = torch.nn.Parameter(torch.rand([1, self.emb_dim], device=self.device, dtype=torch.float))

    def fact_to_masked(self, facts, task_label):
        # print("batch:", len(facts))
        h_index = []
        r_index = []
        t_index = []
        id_facts = [[] for _ in range(self.max_seq_len)]
        id_masks = [[] for _ in range(self.max_seq_len)]
        mask_pos = [[] for _ in range(self.max_seq_len)]
        mask_label = [[] for _ in range(self.max_seq_len)]
        mask_type = [[] for _ in range(self.max_seq_len)]
        rel_query = [[] for _ in range(self.max_seq_len)]
        ent_query = [[] for _ in range(self.max_seq_len)]
        fact_labeler_rel = [[] for _ in range(self.max_seq_len)]
        fact_labeler_ent = [[] for _ in range(self.max_seq_len)]

        for index, fact in enumerate(facts):
            h_index.append(fact[0])
            r_index.append(fact[1])
            t_index.append(fact[2])
            h_index.append(fact[2])
            r_index.append(fact[1] + 1)
            t_index.append(fact[0])
            if len(fact) > 3:
                for i in range(3, len(fact)):
                    if i % 2 == 0:
                        h_index.append(fact[0])
                        r_index.append(fact[1])
                        t_index.append(fact[i])
                        h_index.append(fact[i])
                        r_index.append(fact[1] + 1)
                        t_index.append(fact[0])
                        h_index.append(fact[2])
                        r_index.append(fact[1] + 1)
                        t_index.append(fact[i])
                        h_index.append(fact[i])
                        r_index.append(fact[1])
                        t_index.append(fact[2])
            fact_template = copy.deepcopy(fact) + [0] * (self.max_seq_len - len(fact))
            mask = [1] * len(fact) + [0] * (self.max_seq_len - len(fact))
            # print(fact_template)
            # print(fact)
            for i in range(len(fact)):
                if i % 2 == 1:
                    continue
                temp_fact = copy.deepcopy(fact_template)
                temp_fact[i] = 1
                id_facts[i].append(temp_fact)
                id_masks[i].append(mask)
                mask_pos[i].append(i)
                mask_label[i].append(fact_template[i])
                if i % 2 == 0:
                    mask_type[i].append(1)
                else:
                    mask_type[i].append(-1)
                for j, node in enumerate(fact):
                    if i != j:
                        if j % 2 == 0:
                            ent_query[i].append(node)
                            fact_labeler_ent[i].append(index)
                            # if i == 0:
                            #     print(i, "a")
                        else:
                            rel_query[i].append(node)
                            fact_labeler_rel[i].append(index)
                            # if i == 0:
                            #     print(i, "b")
                    else:
                        if j % 2 == 0:
                            ent_query[i].append(1)
                            fact_labeler_ent[i].append(index)
                            # if i == 0:
                            #     print(i, "a")
                        else:
                            rel_query[i].append(1)
                            fact_labeler_rel[i].append(index)


        return torch.tensor([h_index, t_index, r_index], dtype=torch.long, device=self.device).t(), [id_facts, id_masks, mask_pos, mask_label, mask_type, rel_query, ent_query, fact_labeler_rel, fact_labeler_ent]

    def forward(self, data, facts, task_label, edge_labels):
        remove_index, masked_facts = self.fact_to_masked(facts, task_label)
        graph = data.data[task_label]

        relational_graph = data.relational_data[task_label]
        # for i in range(relational_graph.edge_index.shape[1]):
        #     print(relational_graph.edge_index[:, i])
        entity_graph = data.entity_data[task_label]
        # print(relational_graph.num_nodes)
        # print(relational_graph.num_relations)
        h_index, t_index, r_index = remove_index.unbind(-1)
        if task_label == "train":
            graph = self.entNBF.remove_easy_edges(graph, h_index, t_index, r_index)
        loss = []
        output = []
        info = []
        for i in range(len(masked_facts[0])):
            id_facts, id_masks, mask_pos, mask_label, mask_type, rel_query, ent_query, fact_labeler_rel, fact_labeler_ent = [torch.tensor(x[i], dtype=torch.long, device=self.device) for x in masked_facts]
            if id_facts.shape[0] == 0:
                continue
            batch_size = id_facts.shape[0]
            id_masks = id_masks.float()
            mask_type = mask_type.float()
            # print("start mp")
            relation_representations = self.relNBF(relational_graph, masked_fact=id_facts)
            # print("rel passing done")
            entity_representations = self.entNBF(entity_graph, masked_fact=id_facts)
            # print("ent passing done")
            node_representations = torch.cat([self.pad.unsqueeze(0).expand(batch_size, -1, -1), self.mask.unsqueeze(0).expand(batch_size, -1, -1), self.rel_pad.unsqueeze(0).expand(batch_size, -1, -1), relation_representations[:, 3:, :], entity_representations[:, relation_representations.shape[1]:, :]], dim=1)
            mean_mask_lm_loss, fc_out = self.gran(id_facts, id_masks, edge_labels, mask_pos, mask_label, mask_type, node_representations, relational_graph.num_nodes, graph.num_nodes - graph.num_relations)
            loss.append(mean_mask_lm_loss)
            output.append(fc_out)
            info.append([id_facts, id_masks, mask_pos, mask_label, mask_type, rel_query, ent_query])
        return loss, output, info

