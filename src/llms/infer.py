import sys
import argparse
import time
import random

sys.path.append(".")
from data_utils import get_transformed_io
from eval_utils import extract_spans_para
from main import init_args, set_seed
from llms.api import llm_chat


opinion2sentword = {'great': 'positive', 'bad': 'negative', 'ok': 'neutral'}

def load_prompt(task, data, prompt_type):
    prompt_path = f"llms/prompts/{task}_{data}_{prompt_type}.txt"
    with open(prompt_path, 'r', encoding='utf-8') as fp:
        prompt = fp.read().strip() + "\n\n"
    return prompt


def inference(args, start_idx=0, end_idx=200):
    data_path = f'{args.data_path}/{args.task}/{args.dataset}/{args.data_type}.txt'
    sources, targets = get_transformed_io(data_path,
                                        args.dataset,
                                        args.data_type, top_k=1, args=args)

    # sample `num_sample` samples from sources and targets
    samples = random.sample(list(zip(sources, targets)), args.num_sample)

    prompt = load_prompt(args.task, args.dataset, args.prompt_type)
    
    for i, (source, target) in enumerate(samples):
        if i < start_idx or i > end_idx:
            continue
        print(i)
        try:
            source = " ".join(source)
            gold_list = extract_spans_para(target, 'gold')
            print(gold_list)

            if args.task in ['asqp', 'acos']:
                gold_list = [(at, ot, ac, opinion2sentword[sp]) for (ac, at, sp, ot) in gold_list]
            elif args.task == "aste":
                gold_list = [(at, ot, opinion2sentword[sp]) for (ac, at, sp, ot) in gold_list]
            elif args.task == "tasd":
                gold_list = [(at, ac, opinion2sentword[sp]) for (ac, at, sp, ot) in gold_list]

            context = f"Text: {source}\n"
            context += "Sentiment Elements: "
            res = llm_chat(prompt + context)
            print(context + res)
            print(f"Gold: {gold_list}\n")
            time.sleep(3)
        except BaseException as e: # jump wrong case
            print(">" * 30, "exception:", e)
            exit()
            continue


if __name__ == "__main__":
    args = init_args()
    set_seed(args.seed)

    # default parameters
    args.data_type = "test"
    args.num_sample = 200
    args.prompt_type = "0shot"

    ## tasks:
    # args.task = "acos"
    # args.dataset = "rest16"

    args.task = "asqp"
    args.dataset = "rest15"

    # args.task = "aste"
    # args.dataset = "laptop14"

    # args.task = "tasd"
    # args.dataset = "rest16"

    inference(args, start_idx=0, end_idx=200)
