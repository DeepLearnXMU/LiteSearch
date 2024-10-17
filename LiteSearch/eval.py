import pickle
import re
from search_tree import *

data_fpath = "path/to/prediction"
with open(data_fpath, "rb") as f:
    problems = pickle.load(f)


from transformers import AutoTokenizer
MODEL_PATH = "path/to/policy"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

INVALID_ANS="[invalid]"
def extract_gold_answer(completion):
    ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        try:
            float(match_str)
        except BaseException:
            return INVALID_ANS
        return match_str
    else:
        return INVALID_ANS

def extract_pred_answer(completion):
    ANS_RE = re.compile(r"Answer: .*?(\-?[0-9\.\,]+)")
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        try:
            float(match_str)
        except BaseException:
            return INVALID_ANS
        return match_str
    else:
        return INVALID_ANS

def eq(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-6
    except:
        return False

def read_step_num(tree):
    return len(tree.all_nodes) - 1 # -1 root

def read_token_num(tree):
    return sum([len(tokenizer.tokenize(node.content)) for node in tree.all_nodes if node.content])

correct = 0
total = 0
finished = 0
total_steps = 0
total_token_nums = 0
for problem in problems:
    prediction, _ = problem.return_best_path()
    # prediction = problem.additional_info["greedy"]
    if prediction is not None:
        finished += 1
        hyp = extract_pred_answer(prediction)
        ref = extract_gold_answer(problem.answer)
        if eq(hyp, ref):
            correct += 1
    total += 1
    total_steps += read_step_num(problem)
    total_token_nums += read_token_num(problem)

print("Accuracy:", correct / total)
print("Finished:", finished / total, "Total:", total)
print("Avg Steps:", total_steps / total)
print("Avg Token Nums:", total_token_nums / total)

