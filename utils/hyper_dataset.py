import copy
import os
import csv
import shutil
import torch
import torch_geometric.utils
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.datasets import RelLinkPredDataset, WordNet18RR
# from ultra.tasks import build_relation_graph
import json
import random
import collections


class HyperJointDataset:
    def __init__(self, config):
        # super().__init__(self)
        # config
        self.device = config["device"]
        self.train_dataset = config["train_dataset"]
        self.test_dataset = config["test_dataset"]
        self.valid_dataset = config["valid_dataset"]
        self.test_count = config["test_count"]

        # basic dataset information
        self.facts = []
        self.train_facts = []
        self.valid_facts = []
        self.test_facts = []
        self.test_graph = []
        self.train_facts_id = []
        self.test_facts_id = []
        self.test_graph_id = []
        self.valid_facts_id = []
        self.all_facts = dict()
        self.all_facts_id = dict()
        self.entities = set()
        self.relations = set()
        self.max_seq_len = 0

        # info in id
        self.node_dict = dict()
        self.rel_num = dict()
        self.node_num = dict()
        self.meta_relation_num = 15
        self.meta_entity_num = 7
        # self.meta_global_relation = 8
        self.relation_start = 25

        # graph
        self.data = dict()
        self.relational_data = dict()
        self.entity_data = dict()
        # self.max_degree = dict()
        self.type_matrix = dict()
        self.attr_matrix = dict()

    def read_data(self, filename, dataset_indicator):
        facts = []
        entities = set()
        relations = set()
        with open(filename) as f:
            facts_in_list = [eval(x) for x in f.readlines()]
        for fact in facts_in_list:
            facts.append(["##" + dataset_indicator + "##" + x for x in fact])
        for fact in facts:
            for i, node in enumerate(fact):
                if i % 2 == 0:
                    entities.add(node)
                else:
                    relations.add(node)
        self.facts += facts
        self.entities.update(entities)
        self.relations.update(relations)
        self.node_num[dataset_indicator] = len(list(entities))
        self.rel_num[dataset_indicator] = len(list(relations)) * 2
        return facts

    def read_node_dict(self, facts):
        node_dict = dict()
        node_dict["[pad]"] = 0
        node_dict["[mask]"] = 1
        node_dict["[rel_pad]"] = 2
        node_dict["h-h"] = 3
        node_dict["h-t"] = 4
        node_dict["t-h"] = 5
        node_dict["t-t"] = 6
        node_dict["r-k"] = 7
        node_dict["k-r"] = 8

        node_dict["s-o"] = 12
        node_dict["o-s"] = 13
        node_dict["s-q"] = 14
        node_dict["q-s"] = 15
        node_dict["o-q"] = 16
        node_dict["q-o"] = 17
        node_dict["o-o"] = 18

        node_dict["k-k"] = 19
        node_dict["share h-v"] = 20
        node_dict["share v-h"] = 21
        node_dict["share t-v"] = 22
        node_dict["share v-t"] = 23
        node_dict["share v-v"] = 24

        id_counter = 25
        relations = set()
        entities = set()
        for fact in facts:
            for i, node in enumerate(fact):
                if i % 2 == 0:
                    entities.add(node)
                else:
                    relations.add(node)
        for rel in relations:
            node_dict[rel] = id_counter
            id_counter += 1
            node_dict[rel + "$$$inv"] = id_counter
            id_counter += 1
        for ent in entities:
            node_dict[ent] = id_counter
            id_counter += 1
        return node_dict

    def basic_info_addition(self, hyper_dataset):
        self.facts += hyper_dataset.facts
        self.entities.update(hyper_dataset.entities)
        self.relations.update(hyper_dataset.relations)

    def process(self):
        for name in self.train_dataset:
            self.train_facts += self.read_data("./dataset/" + name + "/inductive_train.txt", name + "_train")
        self.train_dataset = [x + "_train" for x in self.train_dataset]
        for name in self.valid_dataset:
            self.valid_facts += self.read_data("./dataset/" + name + "/valid.txt", name + "_test")
        self.valid_dataset = [x + "_test" for x in self.valid_dataset]
        for name in self.test_dataset:
            self.test_facts += self.read_data("./dataset/" + name + "/test.txt", name + "_test")
            self.test_graph += self.read_data("./dataset/" + name + "/transductive_train.txt", name + "_test")
        self.test_dataset = [x + "_test" for x in self.test_dataset]
        self.node_dict["train"]= self.read_node_dict(self.train_facts)
        self.node_dict["test"] = self.read_node_dict(self.test_graph + self.test_facts)
        self.max_seq_len = max([len(x) for x in self.train_facts] + [len(x) for x in self.test_facts] + [len(x) for x in self.valid_facts])

        self.all_facts["train"], self.all_facts_id["train"] = self.get_truth(self.train_facts, self.max_seq_len // 2 + 1, self.node_dict["train"])
        self.all_facts["test"], self.all_facts_id["test"] = self.get_truth(self.test_facts + self.test_graph + self.valid_facts, self.max_seq_len // 2 + 1, self.node_dict["test"])
        self.all_facts["valid"], self.all_facts_id["valid"] = self.get_truth(self.valid_facts + self.test_graph + self.test_facts, self.max_seq_len // 2 + 1, self.node_dict["test"])
        # print()
        self.train_facts_id = [[self.node_dict["train"][x] for x in fact] for fact in self.train_facts]
        self.test_graph_id = [[self.node_dict["test"][x] for x in fact] for fact in self.test_graph]
        self.test_facts_id = [[self.node_dict["test"][x] for x in fact] for fact in self.test_facts]
        self.valid_facts_id = [[self.node_dict["test"][x] for x in fact] for fact in self.valid_facts]

        print("max_train_len:", max([len(x) for x in self.train_facts_id]))
        print("max_test_len:", max([len(x) for x in self.test_facts_id]))


        # train dataset
        train_target_edges = []
        train_target_types = []
        train_target_attr = []
        for fact in self.train_facts_id:
            # [h, r, t]
            train_target_edges.append([fact[0], fact[2]])
            train_target_types.append(fact[1])
            train_target_attr.append(2)
            # [t, r-1, h]
            train_target_edges.append([fact[2], fact[0]])
            train_target_types.append(fact[1] + 1)
            train_target_attr.append(2)
            if len(fact) > 3:
                for i in range(3, len(fact)):
                    if i % 2 == 1:
                        # [h, r/k, v]
                        train_target_edges.append([fact[0], fact[i + 1]])
                        train_target_types.append(fact[1])
                        train_target_attr.append(fact[i])
                        # [v, r-1/k-1, h]
                        train_target_edges.append([fact[i + 1], fact[0]])
                        train_target_types.append(fact[1] + 1)
                        train_target_attr.append(fact[i] + 1)
                        # [t, r-1/k, v]
                        train_target_edges.append([fact[2], fact[i + 1]])
                        train_target_types.append(fact[1] + 1)
                        train_target_attr.append(fact[i])
                        # [v, r/k-1, t]
                        train_target_edges.append([fact[i + 1], fact[2]])
                        train_target_types.append(fact[1])
                        train_target_attr.append(fact[i] + 1)
        train_target_edges = torch.tensor(train_target_edges, device=self.device, dtype=torch.long)
        train_target_types = torch.tensor(train_target_types, device=self.device, dtype=torch.long)
        train_target_attr = torch.tensor(train_target_attr, device=self.device, dtype=torch.long)
        self.data["train"] = Data(edge_index=train_target_edges.t(), edge_type=train_target_types, edge_attr=train_target_attr, num_nodes=sum([self.node_num[x] for x in self.train_dataset]) + sum([self.rel_num[x] for x in self.train_dataset]) + self.relation_start, num_relations=sum([self.rel_num[x] for x in self.train_dataset]) + self.relation_start)
        self.relational_data["train"] = self.build_relational_graph(self.train_facts_id, self.data["train"].num_relations)
        self.entity_data["train"] = self.build_entity_graph(self.train_facts_id, self.data["train"].num_nodes)
        print(self.data["train"].num_nodes)
        print(self.data["train"].num_relations)

        # test dataset
        target_edges = []
        target_types = []
        target_attr = []
        for fact in self.test_graph_id:
            # [h, r, t]
            target_edges.append([fact[0], fact[2]])
            target_types.append(fact[1])
            target_attr.append(2)
            # [t, r-1, h]
            target_edges.append([fact[2], fact[0]])
            target_types.append(fact[1] + 1)
            target_attr.append(2)
            if len(fact) > 3:
                for i in range(3, len(fact)):
                    if i % 2 == 1:
                        # [h, r/k, v]
                        target_edges.append([fact[0], fact[i + 1]])
                        target_types.append(fact[1])
                        target_attr.append(fact[i])
                        # [v, r-1/k-1, h]
                        target_edges.append([fact[i + 1], fact[0]])
                        target_types.append(fact[1] + 1)
                        target_attr.append(fact[i] + 1)
                        # [t, r-1/k, v]
                        target_edges.append([fact[2], fact[i + 1]])
                        target_types.append(fact[1] + 1)
                        target_attr.append(fact[i])
                        # [v, r/k-1, t]
                        target_edges.append([fact[i + 1], fact[2]])
                        target_types.append(fact[1])
                        target_attr.append(fact[i] + 1)
        target_edges = torch.tensor(target_edges, device=self.device, dtype=torch.long)
        target_types = torch.tensor(target_types, device=self.device, dtype=torch.long)
        target_attr = torch.tensor(target_attr, device=self.device, dtype=torch.long)
        self.data["test"] = Data(edge_index=target_edges.t(), edge_type=target_types, edge_attr=target_attr, num_nodes=sum([self.node_num[x] for x in self.test_dataset]) + sum([self.rel_num[x] for x in self.test_dataset]) + self.relation_start, num_relations=sum([self.rel_num[x] for x in self.test_dataset]) + self.relation_start)
        self.relational_data["test"] = self.build_relational_graph(self.test_graph_id, self.data["test"].num_relations)
        self.entity_data["test"] = self.build_entity_graph(self.test_graph_id, self.data["test"].num_nodes)
        valid_target_edges = []
        valid_target_types = []
        valid_target_attr = []
        for fact in self.test_graph_id:
            # [h, r, t]
            valid_target_edges.append([fact[0], fact[2]])
            valid_target_types.append(fact[1])
            valid_target_attr.append(2)
            # [t, r-1, h]
            valid_target_edges.append([fact[2], fact[0]])
            valid_target_types.append(fact[1] + 1)
            valid_target_attr.append(2)
            if len(fact) > 3:
                for i in range(3, len(fact)):
                    if i % 2 == 1:
                        # [h, r/k, v]
                        valid_target_edges.append([fact[0], fact[i + 1]])
                        valid_target_types.append(fact[1])
                        valid_target_attr.append(fact[i])
                        # [v, r-1/k-1, h]
                        valid_target_edges.append([fact[i + 1], fact[0]])
                        valid_target_types.append(fact[1] + 1)
                        valid_target_attr.append(fact[i] + 1)
                        # [t, r-1/k, v]
                        valid_target_edges.append([fact[2], fact[i + 1]])
                        valid_target_types.append(fact[1] + 1)
                        valid_target_attr.append(fact[i])
                        # [v, r/k-1, t]
                        valid_target_edges.append([fact[i + 1], fact[2]])
                        valid_target_types.append(fact[1])
                        valid_target_attr.append(fact[i] + 1)
        valid_target_edges = torch.tensor(valid_target_edges, device=self.device, dtype=torch.long)
        valid_target_types = torch.tensor(valid_target_types, device=self.device, dtype=torch.long)
        valid_target_attr = torch.tensor(valid_target_attr, device=self.device, dtype=torch.long)
        # self.data["valid"] = Data(edge_index=valid_target_edges.t(), edge_type=valid_target_types, edge_attr=valid_target_attr, num_nodes=sum([self.node_num[x] for x in self.valid_dataset]) + sum([self.rel_num[x] for x in self.valid_dataset]) + self.relation_start, num_relations=sum([self.rel_num[x] for x in self.valid_dataset]) + self.relation_start)
        self.data["valid"] = Data(edge_index=valid_target_edges.t(), edge_type=valid_target_types, edge_attr=valid_target_attr, num_nodes=sum([self.node_num[x] for x in self.valid_dataset]) + sum([self.rel_num[x] for x in self.valid_dataset]) + self.relation_start, num_relations=sum([self.rel_num[x] for x in self.valid_dataset]) + self.relation_start)
        self.relational_data["valid"] = self.build_relational_graph(self.test_graph_id, self.data["valid"].num_relations)
        self.entity_data["valid"] = self.build_entity_graph(self.test_graph_id, self.data["valid"].num_nodes)

    def build_relational_graph(self, id_facts, num_relations):
        ent2rel = dict()
        edges = []
        types = []
        rels = set()
        exist_rels = set()
        for fact in id_facts:
            if len(fact) > 3:
                rels_inner = [x for i, x in enumerate(fact) if i % 2 == 1]
                rel = rels_inner[0]
                keys = rels_inner[1:]
                inv_rel = rel + 1
                inv_keys = [x + 1 for x in keys]
                for k in keys:
                    key = str([k, 7, rel])
                    if key in exist_rels:
                        pass
                    else:
                        edges.append([k, rel])
                        types.append(7)
                        exist_rels.add(key)
                    key = str([rel, 8, k])
                    if key in exist_rels:
                        pass
                    else:
                        edges.append([rel, k])
                        types.append(8)
                        exist_rels.add(key)
                    # for k2 in keys:
                    #     if k != k2:
                    #         key = str([k2,19, k])
                    #         if key in exist_rels:
                    #             pass
                    #         else:
                    #             edges.append([k2, k])
                    #             types.append(19)
                    #             exist_rels.add(key)
                    #         key = str([k, 19, k2])
                    #         if key in exist_rels:
                    #             pass
                    #         else:
                    #             edges.append([k, k2])
                    #             types.append(19)
                    #             exist_rels.add(key)
            for i, node in enumerate(fact):
                if i == 1:
                    rels.add(node)
                    head = fact[i - 1]
                    tail = fact[i + 1]
                    if head not in ent2rel:
                        ent2rel[head] = []
                    ent2rel[head].append([0, node])
                    if tail not in ent2rel:
                        ent2rel[tail] = []
                    ent2rel[tail].append([1, node])
                elif i % 2 == 1:
                    rels.add(node)
                    if fact[i + 1] not in ent2rel:
                        ent2rel[fact[i + 1]] = []
                    ent2rel[fact[i + 1]].append([2, node])
        for key in ent2rel:
            connected_rel = ent2rel[key]
            for i in range(len(connected_rel)):
                for j in range(len(connected_rel)):
                    pos_0 = connected_rel[i][0]
                    rel_0 = connected_rel[i][1]
                    inv_rel_0 = connected_rel[i][1] + 1
                    pos_1 = connected_rel[j][0]
                    rel_1 = connected_rel[j][1]
                    inv_rel_1 = connected_rel[j][1] + 1
                    if rel_0 != rel_1:
                        if pos_0 == 0 and pos_1 == 0:
                            key = str([rel_0, 3, rel_1])
                            if key in exist_rels:
                                pass
                            else:
                                edges.append([rel_0, rel_1])
                                types.append(3)
                                exist_rels.add(key)
                        if pos_0 == 0 and pos_1 == 1:
                            key = str([rel_0, 4, rel_1])
                            if key in exist_rels:
                                pass
                            else:
                                edges.append([rel_0, rel_1])
                                types.append(4)
                                exist_rels.add(key)

                        if pos_0 == 0 and pos_1 == 2:
                            key = str([rel_0, 20, rel_1])
                            if key in exist_rels:
                                pass
                            else:
                                edges.append([rel_0, rel_1])
                                types.append(20)
                                exist_rels.add(key)

                        if pos_0 == 1 and pos_1 == 0:
                            key = str([rel_0, 5, rel_1])
                            if key in exist_rels:
                                pass
                            else:
                                edges.append([rel_0, rel_1])
                                types.append(5)
                                exist_rels.add(key)
                            # types.append(5)
                        if pos_0 == 1 and pos_1 == 1:
                            key = str([rel_0, 6, rel_1])
                            if key in exist_rels:
                                pass
                            else:
                                edges.append([rel_0, rel_1])
                                types.append(6)
                                exist_rels.add(key)
                            # types.append(6)
                        if pos_0 == 1 and pos_1 == 2:
                            key = str([rel_0, 21, rel_1])
                            if key in exist_rels:
                                pass
                            else:
                                edges.append([rel_0, rel_1])
                                types.append(21)
                                exist_rels.add(key)
                            # types.append(9)
                        if pos_0 == 2 and pos_1 == 0:
                            key = str([rel_0, 22, rel_1])
                            if key in exist_rels:
                                pass
                            else:
                                edges.append([rel_0, rel_1])
                                types.append(22)
                                exist_rels.add(key)
                            # types.append(8)
                        if pos_0 == 2 and pos_1 == 1:
                            key = str([rel_0, 23, rel_1])
                            if key in exist_rels:
                                pass
                            else:
                                edges.append([rel_0, rel_1])
                                types.append(23)
                                exist_rels.add(key)
                            # types.append(10)
                        if pos_0 == 2 and pos_1 == 2:
                            key = str([rel_0, 24, rel_1])
                            if key in exist_rels:
                                pass
                            else:
                                edges.append([rel_0, rel_1])
                                types.append(24)
                                exist_rels.add(key)
                            # types.append(24)

        edges = torch.tensor(edges, dtype=torch.long, device=self.device)
        types = torch.tensor(types, dtype=torch.long, device=self.device)
        # for rel in exist_rels:
        #     print(rel)
        data = Data(edge_index=edges.t(), edge_type=types, num_nodes=num_relations, num_relations=self.relation_start)
        # print("data", data.num_nodes)
        return data

    def neighbor_matrix(self, data):
        num_relations = data.num_relations
        num_nodes = data.num_nodes
        type_matrix = torch.zeros([num_nodes, num_relations], device=self.device, dtype=torch.float)
        attr_matrix = torch.zeros([num_nodes, num_relations], device=self.device, dtype=torch.float)
        for i in range(data.edge_index.shape[0]):
            head_index = data.edge_index[i, 0] - self.relation_start - num_relations
            tail_index = data.edge_index[i, 1] - self.relation_start - num_relations
            type_index = data.edge_type[i] - self.relation_start
            attr_index = data.edge_attr[i] - self.relation_start
            type_matrix[head_index, type_index] += 1
            type_matrix[tail_index, type_index] += 1
            attr_matrix[head_index, attr_index] += 1
            attr_matrix[tail_index, attr_index] += 1
        return type_matrix, type_matrix

    def batching(self, dataset, batch_size):
        if dataset == "train":
            data = copy.deepcopy(self.train_facts_id)
        elif dataset == "valid":
            data = copy.deepcopy(self.valid_facts_id)
        else:
            data = copy.deepcopy(self.test_facts_id)
        random.shuffle(data)
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    def get_truth(self, all_facts, max_n, node_dict):
        max_aux = max_n - 2
        max_seq_length = 2 * max_aux + 3
        gt_dict = collections.defaultdict(lambda: collections.defaultdict(list))
        all_fact_ids = list()
        for fact in all_facts:
            id_fact = list()
            for i, item in enumerate(fact):
                id_fact.append(node_dict[item])
            all_fact_id = copy.copy(id_fact)
            all_fact_ids.append(all_fact_id)
            while len(id_fact) < (2 * max_n - 1):
                id_fact.append(0)
            for pos in range(max_seq_length):
                if id_fact[pos] == 0:
                    continue
                key = " ".join([
                    str(id_fact[x]) for x in range(max_seq_length) if x != pos
                ])
                gt_dict[pos][key].append(id_fact[pos])

        return gt_dict, all_fact_ids

    def build_entity_graph(self, id_facts, num_entities):
        types = []
        edges = []

        exist = set()
        for fact in id_facts:
            key = str([fact[0], 12, fact[2]])
            if False:
                pass
            else:
                edges.append([fact[0], fact[2]])
                types.append(12)
                exist.add(key)
            key = str([fact[2], 13, fact[0]])
            if False:
                pass
            else:
                edges.append([fact[2], fact[0]])
                types.append(13)
                exist.add(key)

            if len(fact) > 3:
                kv = [[fact[x], fact[x + 1]] for x in range(3, len(fact)) if x % 2 == 1]
                for i in range(len(kv)):
                    key = str([fact[0], 14, kv[i][1]])
                    if False:
                        pass
                    else:
                        edges.append([fact[0], kv[i][1]])
                        types.append(14)
                        exist.add(key)
                    key = str([kv[i][1], 15, fact[0]])
                    if False:
                        pass
                    else:
                        edges.append([kv[i][1], fact[0]])
                        types.append(15)
                        exist.add(key)
                    key = str([fact[2], 16, kv[i][1]])
                    if False:
                        pass
                    else:
                        edges.append([fact[2], kv[i][1]])
                        types.append(16)
                        exist.add(key)
                    key = str([kv[i][1], 17, fact[2]])
                    if False:
                        pass
                    else:
                        edges.append([kv[i][1], fact[2]])
                        types.append(17)
                        exist.add(key)
                    for j in range(len(kv)):
                        if i > j:
                            key = str([kv[i][1], 18, kv[j][1]])
                            if False:
                                pass
                            else:
                                edges.append([kv[i][1], kv[j][1]])
                                types.append(18)
                                exist.add(key)
                            key = str([kv[j][1], 18, kv[i][1]])
                            if False:
                                pass
                            else:
                                edges.append([kv[j][1], kv[i][1]])
                                types.append(18)
                                exist.add(key)

        edges = torch.tensor(edges, dtype=torch.long, device=self.device)
        types = torch.tensor(types, dtype=torch.long, device=self.device)
        data = Data(edge_index=edges.t(), edge_type=types, num_nodes=num_entities, num_relations=self.relation_start)
        return data

