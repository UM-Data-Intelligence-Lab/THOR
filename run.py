import argparse
import ast
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from utils.args import ArgumentGroup, print_arguments
from utils.hyper_dataset import HyperJointDataset
import logging
from model.THOR import THOR
# from reader.data_reader import read_input
from utils.data_loader import prepare_EC_info, get_edge_labels
# from model.NYLON import NYLON
import time
import math
import random
import torch
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader
import copy
import numpy as np
from itertools import cycle
from utils.evaluation import batch_evaluation, compute_metrics

torch.set_printoptions(precision=8)

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info(logger.getEffectiveLevel())

parser = argparse.ArgumentParser(description='HyperKE4TI')
NYLON_g = ArgumentGroup(parser, "model", "model and checkpoint configuration.")
NYLON_g.add_arg('device', type=str, default='device', help="")
NYLON_g.add_arg('train_dataset', type=list, default=["JFFI_clean/v2"], help="")
NYLON_g.add_arg('test_dataset', type=list, default=["WD20K33_clean/v1"], help="")
NYLON_g.add_arg('valid_dataset', type=list, default=["WD20K33_clean/v1"], help="")
NYLON_g.add_arg('test_count', type=int, default=4000, help="")
NYLON_g.add_arg('emb_dim', type=int, default=64, help="")
NYLON_g.add_arg('layer_dim', type=list, default=[64, 64, 64, 64, 64, 64], help="")
NYLON_g.add_arg('dim', type=int, default=64, help="")
NYLON_g.add_arg('onto_dim', type=int, default=64, help="")
NYLON_g.add_arg('ins_layer_num', type=int, default=5, help="")
NYLON_g.add_arg('onto_layer_num', type=int, default=5, help="")
NYLON_g.add_arg('neg_typing_margin', type=float, default=0.1, help="")
NYLON_g.add_arg('neg_triple_margin', type=float, default=0.2, help="")
NYLON_g.add_arg('nums_neg', type=int, default=30, help="")
NYLON_g.add_arg('mapping_neg_nums', type=int, default=30, help="")
NYLON_g.add_arg('learning_rate', type=float, default=1e-3, help="")
NYLON_g.add_arg('batch_size', type=int, default=8, help="")
NYLON_g.add_arg('epochs', type=int, default=100, help="")
NYLON_g.add_arg('combine', type=ast.literal_eval, default=True, help="")
NYLON_g.add_arg('ent_top_k', type=list, default=[1, 3, 5, 10], help="")
NYLON_g.add_arg("use_cuda", bool, True, "If set, use GPU for training.")
NYLON_g.add_arg('ins_intermediate_size', type=int, default=128, help="")
NYLON_g.add_arg('onto_intermediate_size', type=int, default=128, help="")
NYLON_g.add_arg('num_hidden_layers', type=int, default=3, help="")
NYLON_g.add_arg('num_attention_heads', type=int, default=2, help="")
NYLON_g.add_arg('hidden_dropout_prob', type=float, default=0.1, help="")
NYLON_g.add_arg('attention_dropout_prob', type=float, default=0.1, help="")
NYLON_g.add_arg('meta_lr', type=float, default=0.1, help="")
NYLON_g.add_arg('num_edges', type=int, default=6, help="")
args = parser.parse_args()

def main(args):
    config = vars(args)
    if args.use_cuda:
        device = torch.device("cuda")
        config["device"] = "cuda"
    else:
        device = torch.device("cpu")
        config["device"] = "cpu"

    joint_dataset = HyperJointDataset(config)
    joint_dataset.process()
    config["meta_relation_num"] = joint_dataset.meta_relation_num
    config["relation_start"] = joint_dataset.relation_start
    config["max_seq_len"] = joint_dataset.max_seq_len
    gran_config = dict()
    gran_config["num_hidden_layers"] = config["num_hidden_layers"]
    gran_config["num_attention_heads"] = config["num_attention_heads"]
    gran_config["hidden_size"] = config["emb_dim"]
    gran_config["intermediate_size"] = config["emb_dim"]
    gran_config["hidden_dropout_prob"] = config["hidden_dropout_prob"]
    gran_config["attention_dropout_prob"] = config["attention_dropout_prob"]
    gran_config["num_edges"] = config["num_edges"]
    gran_config["max_arity"] = joint_dataset.max_seq_len // 2 + 1
    gran_config["device"] = config["device"]
    gran_config["special_nodes"] = joint_dataset.relation_start
    ins_edge_labels = get_edge_labels(gran_config["max_arity"]).to(device)
    model = THOR(config, gran_config).to(device)
    model.train()
    optim = torch.optim.Adam([{"params": model.parameters()}], lr=config['learning_rate'])
    test_node_dict_inv = {v: k for k, v in joint_dataset.node_dict["test"].items()}
    previous_mrr = 0
    for iteration in range(config["epochs"]):
        t0 = time.time()
        model.train()

        for i, batch in enumerate(joint_dataset.batching("train", config["batch_size"])):
            model.train()
            t1 = time.time()
            optim.zero_grad()
            loss, _, _ = model(joint_dataset, batch, "train", ins_edge_labels)
            ins_loss = torch.sum(torch.stack(loss))
            ins_loss.backward()
            optim.step()
            if i % 100 == 0:
                print(str(i) + ' , ins_loss: ' + str(ins_loss.item()) + " with memory " + str(torch.cuda.memory_allocated(device=device) / 1024 / 1024))
        print("epoch", iteration, "cost time:", time.time() - t0)

        model.eval()
        with torch.no_grad():
            predict(model, joint_dataset, config["batch_size"], joint_dataset.all_facts["test"], ins_edge_labels, device, test_node_dict_inv)


def predict(model, joint_dataset, batch_size, ins_all_facts, ins_edge_labels, device, test_node_dict_inv):
    start = time.time()

    step = 0
    ins_ret_ranks = dict()
    ins_ret_ranks['entity'] = torch.empty(0).to(device)
    ins_ret_ranks['2-ht'] = torch.empty(0).to(device)
    ins_ret_ranks['n-ht'] = torch.empty(0).to(device)
    ins_ret_ranks['n-a'] = torch.empty(0).to(device)
    ins_ret_ranks['n-v'] = torch.empty(0).to(device)

    # while steps < max_train_steps:
    for i, data in enumerate(joint_dataset.batching("test", batch_size)):
        _, output, info = model(joint_dataset, data, "test", ins_edge_labels)
        # length = info[0].shape[0]
        for mask_i_output in range(len(output)):
            hyper_output = output[mask_i_output].clone().detach()
            hyper_tag = (torch.sum(info[mask_i_output][1], dim=1) > 3)
            hyper_output = hyper_output[hyper_tag]
            hyper_data = list()
            for item in info[mask_i_output][:5]:
                # print("item:", item)
                hyper_data.append(item[hyper_tag].clone().detach())
            ins_ret_ranks = batch_evaluation(output[mask_i_output], info[mask_i_output][:5], ins_all_facts, ins_ret_ranks, device, test_node_dict_inv)

    ins_eval_performance = compute_metrics(ins_ret_ranks)

    ins_all_entity = "ENTITY\t\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
        ins_eval_performance['entity']['mrr'],
        ins_eval_performance['entity']['hits1'],
        ins_eval_performance['entity']['hits3'],
        ins_eval_performance['entity']['hits5'],
        ins_eval_performance['entity']['hits10'])

    ins_all_ht = "HEAD/TAIL\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
        ins_eval_performance['ht']['mrr'],
        ins_eval_performance['ht']['hits1'],
        ins_eval_performance['ht']['hits3'],
        ins_eval_performance['ht']['hits5'],
        ins_eval_performance['ht']['hits10'])


    logger.info("\n-------- E Evaluation Performance --------\n%s\n%s\n%s" % (
        "\t".join(["TASK\t", "MRR", "Hits@1", "Hits@3", "Hits@5", "Hits@10"]),
        ins_all_ht, ins_all_entity))
    end = time.time()
    logger.info("INS time: " + str(round(end - start, 3)) + 's')

    return ins_eval_performance['entity']['hits1']


if __name__ == '__main__':
    print_arguments(args)

    main(args)


