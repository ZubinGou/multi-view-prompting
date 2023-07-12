import random
import json
import numpy as np
from itertools import permutations
import torch
from torch.utils.data import Dataset
# from transformers import AdamW, T5Tokenizer, T5ForConditionalGeneration

from t5_score import MyT5ForConditionalGenerationScore
from const import *


def get_element_tokens(task):
    dic = {
        "aste":
            ["[A]", "[O]", "[S]"],
        "tasd":
            ["[A]", "[C]", "[S]"],
        "aocs":
        ["[A]", "[O]", "[C]", "[S]"],
        "asqp":
            ["[A]", "[O]", "[C]", "[S]"],
    }
    return dic[task]


def get_orders(task, data, args, sents, labels):
    ## uncomment to calculate orders from scratch
    # if torch.cuda.is_available():
    #     device = torch.device('cuda:0')
    # else:
    #     device = torch.device("cpu")
    # tokenizer = T5Tokenizer.from_pretrained("t5-base").to(device)
    # model = MyT5ForConditionalGenerationScore.from_pretrained(
    #     "t5-base").to(device)
    # optim_orders_all = choose_best_order_global(sents, labels, model,
    #                                         tokenizer, device,
    #                                         args.task)

    if args.single_view_type == 'rank':
        orders = optim_orders_all[task][data]
    elif args.single_view_type == 'rand':
        orders = [random.Random(args.seed).choice(
            optim_orders_all[task][data])]
    elif args.single_view_type == "heuristic":
        orders = heuristic_orders[task]
    return orders


def read_line_examples_from_file(data_path,
                                 task_name,
                                 data_name,
                                 lowercase,
                                 silence=True):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    tasks, datas = [], []
    sents, labels = [], []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        words, labels = [], []
        for line in fp:
            line = line.strip()
            if lowercase:
                line = line.lower()
            if "unified" in task_name:
                _task, _data, line = line.split("\t")
                tasks.append(_task)
                datas.append(_data)
            else:
                tasks.append(task_name)
                datas.append(data_name)
            if line != '':
                words, tuples = line.split('####')
                sents.append(words.split())
                labels.append(eval(tuples))
    if silence:
        print(f"Total examples = {len(sents)}")
    return tasks, datas, sents, labels


def cal_entropy(inputs, preds, model_path, tokenizer, device=torch.device('cuda:0')):
    all_entropy = []
    model = MyT5ForConditionalGenerationScore.from_pretrained(model_path).to(
        device)
    batch_size = 8
    _inputs = [' '.join(s) for s in inputs]
    _preds = [' '.join(s) for s in preds]
    for id in range(0, len(inputs), batch_size):
        in_batch = _inputs[id: min(id + batch_size, len(inputs))]
        pred_batch = _preds[id: min(id + batch_size, len(inputs))]
        assert len(in_batch) == len(pred_batch)
        tokenized_input = tokenizer.batch_encode_plus(in_batch,
                                                      max_length=200,
                                                      padding="max_length",
                                                      truncation=True,
                                                      return_tensors="pt")
        tokenized_target = tokenizer.batch_encode_plus(pred_batch,
                                                       max_length=200,
                                                       padding="max_length",
                                                       truncation=True,
                                                       return_tensors="pt")

        target_ids = tokenized_target["input_ids"].to(device)

        target_ids[target_ids[:, :] == tokenizer.pad_token_id] = -100
        outputs = model(
            input_ids=tokenized_input["input_ids"].to(device),
            attention_mask=tokenized_input["attention_mask"].to(device),
            labels=target_ids,
            decoder_attention_mask=tokenized_target["attention_mask"].to(device))

        loss, entropy = outputs[0]
        all_entropy.extend(entropy)
    return all_entropy


def order_scores_function(quad_list, cur_sent, model, tokenizer, device, task):
    q = get_element_tokens(task)

    all_orders = permutations(q)
    all_orders_list = []

    all_targets = []
    all_inputs = []
    cur_sent = " ".join(cur_sent)
    for each_order in all_orders:
        cur_order = " ".join(each_order)
        all_orders_list.append(cur_order)
        cur_target = []
        for each_q in quad_list:
            cur_target.append(each_q[cur_order][0])

        all_inputs.append(cur_sent)
        all_targets.append(" ".join(cur_target))

    tokenized_input = tokenizer.batch_encode_plus(all_inputs,
                                                  max_length=200,
                                                  padding="max_length",
                                                  truncation=True,
                                                  return_tensors="pt")
    tokenized_target = tokenizer.batch_encode_plus(all_targets,
                                                   max_length=200,
                                                   padding="max_length",
                                                   truncation=True,
                                                   return_tensors="pt")

    target_ids = tokenized_target["input_ids"].to(device)

    target_ids[target_ids[:, :] == tokenizer.pad_token_id] = -100
    outputs = model(
        input_ids=tokenized_input["input_ids"].to(device),
        attention_mask=tokenized_input["attention_mask"].to(device),
        labels=target_ids,
        decoder_attention_mask=tokenized_target["attention_mask"].to(device))

    loss, entropy = outputs[0]
    results = {}
    for i, _ in enumerate(all_orders_list):
        cur_order = all_orders_list[i]
        results[cur_order] = {"loss": loss[i], "entropy": entropy[i]}
    # print(best_quad)
    return results


def choose_best_order_global(sents, labels, model, tokenizer, device, task):
    q = get_element_tokens(task)
    all_orders = permutations(q)
    all_orders_list = []
    scores = []

    for each_order in all_orders:
        cur_order = " ".join(each_order)
        all_orders_list.append(cur_order)
        scores.append(0)

    for i in range(len(sents)):
        label = labels[i]
        sent = sents[i]

        quad_list = []
        for _tuple in label:
            # parse ASTE tuple
            if task == "aste":
                _tuple = parse_aste_tuple(_tuple, sent)

            at, ac, sp, ot = get_task_tuple(_tuple, task)

            element_dict = {"[A]": at, "[O]": ot, "[C]": ac, "[S]": sp}
            element_list = []
            for key in q:
                element_list.append("{} {}".format(key, element_dict[key]))

            x = permutations(element_list)

            permute_object = {}
            for each in x:
                order = []
                content = []
                for e in each:
                    order.append(e[0:4])
                    content.append(e[4:])
                order_name = " ".join(order)
                content = " ".join(content)
                permute_object[order_name] = [content, " ".join(each)]

            quad_list.append(permute_object)

        order_scores = order_scores_function(quad_list, sent, model, tokenizer,
                                             device, task)
        for e in order_scores:
            index = all_orders_list.index(e)
            scores[index] += order_scores[e]['entropy']

    indexes = np.argsort(np.array(scores))  # [::-1]
    returned_orders = []
    for i in indexes:
        returned_orders.append(all_orders_list[i])

    print("Orders:", returned_orders)
    return returned_orders


def parse_aste_tuple(_tuple, sent):
    if isinstance(_tuple[0], str):
        res = _tuple
    elif isinstance(_tuple[0], list):
        # parse at
        start_idx = _tuple[0][0]
        end_idx = _tuple[0][-1] if len(_tuple[0]) > 1 else start_idx
        at = ' '.join(sent[start_idx:end_idx + 1])

        # parse ot
        start_idx = _tuple[1][0]
        end_idx = _tuple[1][-1] if len(_tuple[1]) > 1 else start_idx
        ot = ' '.join(sent[start_idx:end_idx + 1])
        res = [at, ot, _tuple[2]]
    else:
        print(_tuple)
        raise NotImplementedError
    return res


def get_task_tuple(_tuple, task):
    if task == "aste":
        at, ot, sp = _tuple
        ac = None
    elif task == "tasd":
        at, ac, sp = _tuple
        ot = None
    elif task in ["asqp", "acos"]:
        at, ac, sp, ot = _tuple
    else:
        raise NotImplementedError

    if sp:
        sp = sentword2opinion[sp.lower()] if sp in sentword2opinion \
            else senttag2opinion[sp.lower()]  # 'POS' -> 'good'
    if at and at.lower() == 'null':  # for implicit aspect term
        at = 'it'

    return at, ac, sp, ot


def add_prompt(sent, orders, task, data_name, args):
    if args.multi_task:
        # add task and data prefix
        sent = [task, ":", data_name, ":"] + sent

    # add ctrl_token
    if args.ctrl_token == "none":
        pass
    elif args.ctrl_token == "post":
        sent = sent + orders
    elif args.ctrl_token == "pre":
        sent = orders + sent
    else:
        raise NotImplementedError
    return sent


def get_para_targets(sents, labels, data_name, data_type, top_k, task, args):
    """
    Obtain the target sentence under the paraphrase paradigm
    """
    targets = []
    new_sents = []
    if task in ['aste', 'tasd']:
        # at most 5 orders for triple tasks
        top_k = min(5, top_k)

    optim_orders = get_orders(task, data_name, args, sents, labels)[:top_k]

    for i in range(len(sents)):
        label = labels[i]
        cur_sent = sents[i]
        cur_sent_str = " ".join(cur_sent)

        # ASTE: parse at & ot
        if task == 'aste':
            assert len(label[0]) == 3
            parsed_label = []
            for _tuple in label:
                parsed_tuple = parse_aste_tuple(_tuple, sents[i])
                parsed_label.append(parsed_tuple)
            label = parsed_label

        # sort label by order of appearance
        # at, ac, sp, ot
        if args.sort_label and len(label) > 1:
            label_pos = {}
            for _tuple in label:
                at, ac, sp, ot = get_task_tuple(_tuple, task)

                # get last at / ot position
                at_pos = cur_sent_str.find(at) if at else -1
                ot_pos = cur_sent_str.find(ot) if ot else -1
                last_pos = max(at_pos, ot_pos)
                last_pos = 1e4 if last_pos < 0 else last_pos
                label_pos[tuple(_tuple)] = last_pos
            new_label = [
                list(k)
                for k, _ in sorted(label_pos.items(), key=lambda x: x[1])
            ]
            label = new_label

        quad_list = []
        for _tuple in label:
            at, ac, sp, ot = get_task_tuple(_tuple, task)
            element_dict = {"[A]": at, "[O]": ot, "[C]": ac, "[S]": sp}
            token_end = 3

            element_list = []
            for key in optim_orders[0].split(" "):
                element_list.append("{} {}".format(key, element_dict[key]))

            x = permutations(element_list)
            permute_object = {}
            for each in x:
                order = []
                content = []
                for e in each:
                    order.append(e[0:token_end])
                    content.append(e[token_end:])
                order_name = " ".join(order)
                content = " ".join(content)
                permute_object[order_name] = [content, " ".join(each)]

            quad_list.append(permute_object)

        for o in optim_orders:
            tar = []
            for each_q in quad_list:
                tar.append(each_q[o][1])

            targets.append(" [SSEP] ".join(tar))
            # add prompt
            new_sent = add_prompt(cur_sent, o.split(), task, data_name, args)
            new_sents.append(new_sent)

    return new_sents, targets


def get_para_targets_dev(sents, labels, data_name, task, args):
    """
    Obtain the target sentence under the paraphrase paradigm
    """
    new_sents = []
    targets = []
    optim_orders = get_orders(task, data_name, args, sents=None, labels=None)
    top_order = optim_orders[0].split(" ")
    for sent, label in zip(sents, labels):
        all_quad_sentences = []
        for _tuple in label:
            # parse ASTE tuple
            if task == "aste":
                _tuple = parse_aste_tuple(_tuple, sent)

            at, ac, sp, ot = get_task_tuple(_tuple, task)

            element_dict = {"[A]": at, "[O]": ot, "[C]": ac, "[S]": sp}
            element_list = []
            for key in top_order:
                element_list.append("{} {}".format(key, element_dict[key]))

            one_quad_sentence = " ".join(element_list)
            all_quad_sentences.append(one_quad_sentence)

        target = ' [SSEP] '.join(all_quad_sentences)
        targets.append(target)

        # add prompt
        sent = add_prompt(sent, top_order, task, data_name, args)

        new_sents.append(sent)
    return new_sents, targets


def get_transformed_io(data_path, data_name, data_type, top_k, args):
    """
    The main function to transform input & target according to the task
    """
    tasks, datas, sents, labels = read_line_examples_from_file(
        data_path, args.task, args.dataset, args.lowercase)

    # the input is just the raw sentence
    inputs = [s.copy() for s in sents]

    # low resource
    if data_type == 'train' and args.data_ratio != 1.0:
        num_sample = int(len(inputs) * args.data_ratio)
        sample_indices = random.sample(list(range(0, len(inputs))), num_sample)
        sample_inputs = [inputs[i] for i in sample_indices]
        sample_labels = [labels[i] for i in sample_indices]
        inputs, labels = sample_inputs, sample_labels
        print(
            f"Low resource: {args.data_ratio}, total train examples = {num_sample}")
        if num_sample <= 20:
            print("Labels:", sample_labels)

    if data_type == "train" or args.eval_data_split == "dev" or data_type == "test":
        new_inputs, targets = get_para_targets(inputs, labels, data_name,
                                               data_type, top_k, args.task,
                                               args)
    else:
        new_inputs, targets = get_para_targets_dev(inputs, labels, data_name,
                                                   args.task, args)

    print(len(inputs), len(new_inputs), len(targets))
    return new_inputs, targets


def get_transformed_io_unified(data_path, task_name, data_name, data_type,
                               top_k, args):
    """
    The main function to transform input & target according to the task
    """
    tasks, datas, sents, labels = read_line_examples_from_file(
        data_path, task_name, data_name, lowercase=args.lowercase)
    sents = [s.copy() for s in sents]
    new_inputs, targets = [], []
    for task, data, sent, label in zip(tasks, datas, sents, labels):
        if data_type == "train" or (data_type == "test" and args.multi_path):
            new_input, target = get_para_targets([sent], [label], data,
                                                 data_type, top_k, task, args)
        else:
            new_input, target = get_para_targets_dev([sent], [label], data,
                                                     task, args)
        new_inputs.extend(new_input)
        targets.extend(target)

    print("Ori sent size:", len(sents))
    print("Input size:", len(new_inputs), len(targets))
    print("Examples:")
    print(new_inputs[:10])
    print(targets[:10])

    return new_inputs, targets


class ABSADataset(Dataset):

    def __init__(self,
                 tokenizer,
                 task_name,
                 data_name,
                 data_type,
                 top_k,
                 args,
                 max_len=128):
        self.data_path = f'{args.data_path}/{task_name}/{data_name}/{data_type}.txt'
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.task_name = task_name
        self.data_name = data_name
        self.data_type = data_type
        self.args = args

        self.top_k = top_k

        self.inputs = []
        self.targets = []

        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze(
        )  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze(
        )  # might need to squeeze
        return {
            "source_ids": source_ids,
            "source_mask": src_mask,
            "target_ids": target_ids,
            "target_mask": target_mask
        }

    def _build_examples(self):

        if self.args.multi_task:
            inputs, targets = get_transformed_io_unified(
                self.data_path, self.task_name, self.data_name, self.data_type,
                self.top_k, self.args)
        else:
            inputs, targets = get_transformed_io(self.data_path,
                                                 self.data_name,
                                                 self.data_type, self.top_k,
                                                 self.args)

        for i in range(len(inputs)):
            # change input and target to two strings
            input = ' '.join(inputs[i])
            target = targets[i]

            tokenized_input = self.tokenizer.batch_encode_plus(
                [input],
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt")
            
            # for ACOS Restaurant and Laptop dataset
            # the max target length is much longer than 200
            # we need to set a larger max length for inference
            target_max_length = 1024 if self.data_type == "test" else self.max_len

            tokenized_target = self.tokenizer.batch_encode_plus(
                [target],
                max_length=target_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt")

            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)
