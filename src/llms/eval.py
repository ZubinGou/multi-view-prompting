import sys
import json
import ast
sys.path.append(".")

from eval_utils import compute_f1_scores

def eval_log(file_path):
    """
    read the LLMs log file and compute the F1 scores
    """
    all_labels, all_preds = [], []

    with open(file_path, 'r', encoding='utf-8') as fp:
        for line in fp:
            if line.startswith("Sentiment Elements:"):
                line = line.split("Sentiment Elements: ")[1].strip()
                try:
                    pred_list = ast.literal_eval(line)
                except:
                    # print(">>>", line)
                    pred_list = []
            elif line.startswith("Gold:"):
                line = line.split("Gold:")[1].strip()
                gold_list = ast.literal_eval(line)
                all_labels.append(gold_list)
                all_preds.append(pred_list)

    scores = compute_f1_scores(all_preds, all_labels)
    print("Count:", len(all_preds))
    print(scores)


if __name__ == "__main__":
    log_files = [
        "llms/results/chatgpt_test_200_acos_rest16_0shot.log", 
        "llms/results/chatgpt_test_200_acos_rest16_10shot.log",
        "llms/results/chatgpt_test_200_asqp_rest15_0shot.log",
        "llms/results/chatgpt_test_200_asqp_rest15_10shot.log",
        "llms/results/chatgpt_test_200_aste_laptop14_0shot.log",
        "llms/results/chatgpt_test_200_aste_laptop14_10shot.log",
        "llms/results/chatgpt_test_200_tasd_rest16_0shot.log",
        "llms/results/chatgpt_test_200_tasd_rest16_10shot.log"
    ]

    for log_file in log_files:
        print(log_file)
        eval_log(log_file)
        print()
