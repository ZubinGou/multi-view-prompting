import os
import random
import sys
sys.path.append(".")

from collections import Counter
from data_utils import parse_aste_tuple

join = os.path.join


def process(data_folder, tasks, out_dir):
    """
    1. Aggregate all train, dev, and test sets for the tasks acos/asqp/aste/tasd.
    2. Remove data contamination: delete the test set data that exists in the train/dev sets.
    3. Output data.txt
    Data format: (task, data, words, tuples)
    """
    train_data = []
    test_data = []
    # merge all data
    for task in tasks:
        task_path = join(data_folder, task)
        print("task:", task_path)
        for data_name in os.listdir(task_path):
            data_path = join(task_path, data_name)
            print("data:", data_path)
            # acos data_name
            for split in ["train", "dev", "test"]:
                with open(join(data_path, "{}.txt".format(split)),
                          'r',
                          encoding="utf-8") as fp:
                    for line in fp:
                        line = line.strip().lower()
                        if line != '':
                            words, tuples = line.split('####')
                        # parse aste
                        if task == "aste":
                            aste_tuples = []
                            for _tuple in eval(tuples):
                                parsed_tuple = parse_aste_tuple(
                                    _tuple, words.split())
                                aste_tuples.append(parsed_tuple)
                            tuples = str(aste_tuples)

                        # output
                        if split == "test":
                            test_data.append((task, data_name, words, tuples))
                        else:
                            train_data.append((task, data_name, words, tuples))
    # remove inputs in test set
    test_inputs = set()
    for _, _, words, _ in test_data:
        test_inputs.add(words.replace(" ", ""))
    train_data_safe = []
    for item in train_data:
        if item[2].replace(" ", "") not in test_inputs:
            train_data_safe.append(item)

    print("test inputs size:", len(test_inputs))
    print("train data size (before remove test):", len(train_data))
    print("train data size (after remove test):", len(train_data_safe))

    # dedup
    random.seed(0)
    random.shuffle(train_data_safe)
    train_data_dedup = []
    train_pairs = set()
    for item in train_data_safe:
        pair = (item[2] + item[3]).replace(" ", "")
        if pair not in train_pairs:
            train_pairs.add(pair)
            train_data_dedup.append(item)

    print("train data size (dedup):", len(train_data_dedup))

    # stats
    task_list = []
    data_list = []
    for task, data_name, _, _ in train_data_dedup:
        task_list.append(task)
        data_list.append(data_name)
    print("Tasks counts:", Counter(task_list))
    print("Data counts:", Counter(data_list))

    # output
    for seed in [5, 10, 15, 20, 25]:
        os.makedirs(out_dir + "seed{}".format(seed), exist_ok=True)
        random.seed(seed)
        random.shuffle(train_data_dedup)
        idx = int(len(train_data_dedup) * 0.9)
        train_set = train_data_dedup[:idx]
        dev_set = train_data_dedup[idx:]

        # sort
        train_set = sorted(train_set, key=lambda x: x[2])

        with open(out_dir + "seed{}/train.txt".format(seed),
                  'w',
                  encoding="utf-8") as fp:
            for item in train_set:
                fp.write("{}\t{}\t{}####{}\n".format(*item))

        with open(out_dir + "seed{}/dev.txt".format(seed),
                  'w',
                  encoding="utf-8") as fp:
            for item in dev_set:
                fp.write("{}\t{}\t{}####{}\n".format(*item))


if __name__ == "__main__":
    tasks = ["acos", "asqp", "aste", "tasd"]
    process("../data", tasks, "../data/unified/")
