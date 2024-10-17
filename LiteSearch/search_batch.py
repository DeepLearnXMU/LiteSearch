import os
from call_joint_service import call
from search_tree import *
import json
import pickle
import numpy as np
import jsonlines


LIMIT=100
MAX_BUDGET=10
EXPECTED=0.9
TEMPERATURE=0.6
MIN_VALUE=0.8
END_LEAF_NUM=5 

# task dependent
def assert_end(text):
    return True if text.startswith("####") else False

def fix_value(state):
    if state.parent is not None: # repeat
        if state.parent.content == state.content:
            state.value = 0
    if state.content is not None and (len(state.content) == 0 or len(state.content) > 1024): # too short or too long
        state.value = 0


SEQ_STOP_TOKENS = ["Question:", "Answer:"]
STEP_STOP_TOKENS = ["Question:", "Answer:", "\n"]

CONTINUE = False
data_fpath = "path/to/data"
proc_data_fpath = "path/to/proc/data"
output_fpath = f"path/to/out/data"

if __name__ == '__main__':
    
    if CONTINUE:    
        problems = pickle.load(open(data_fpath, "rb"))
        start = max([problem.return_timestep() for problem in problems])
    else:
        if os.path.exists(proc_data_fpath):
            dataset = []
            with open(proc_data_fpath, "r") as f:
                for line in f.readlines():
                    dataset.append(json.loads(line))
        else:
            dataset = []
            with open(data_fpath, "r") as f:
                for line in f.readlines():
                    dataset.append(json.loads(line))

            print("question value ...")
            questions = []
            anchors = []
            for instance in dataset:
                if "question_value" not in instance:
                    questions.append(instance["question"])
                    anchors.append(instance)
            if questions:
                _, q_values = call(questions, [None] * len(questions), [None] * len(questions), [None] * len(questions))
                for instance, q_value in zip(anchors, q_values):
                    instance["question_value"] = q_value

            print("greedy ...")
            questions = []
            anchors = []
            for instance in dataset:
                if "greedy" not in instance:
                    questions.append(instance["question"])
                    anchors.append(instance)

            if questions:
                greedy_preds, greedy_values = call(questions, [None] * len(questions), [0] * len(questions), [SEQ_STOP_TOKENS] * len(questions))
                for instance, greedy_pred, greedy_value in zip(anchors, greedy_preds, greedy_values):
                    instance["greedy"] = greedy_pred
                    instance["greedy_value"] = greedy_value

            # save
            with jsonlines.open(proc_data_fpath, "w") as writer:
                writer.write_all(dataset)

        problems = []
        for instance in dataset:
            question = instance["question"]
            answer = instance["answer"]
            additional_info = {"greedy": instance["greedy"], "greedy_value": instance["greedy_value"],
                               "max_budget": MAX_BUDGET, "expected": EXPECTED}
            problem = Tree(question, answer, additional_info)
            problem.init_root_node(instance["question_value"])
            problems.append(problem)
        start = 0

    for i in range(start, LIMIT):
        questions = []
        anchors = []
        paths = []
        temperatures = []
        finished = 0
        for problem in problems:
            state = problem.select_best_node()
            is_finished = problem.is_finished(MIN_VALUE)
            leaf_num = problem.get_leaf_num()
            if not is_finished and state is not None and leaf_num < END_LEAF_NUM:
                for j in range(state.budget):
                    anchors.append(state)
                    questions.append(problem.question)
                    paths.append(state.print_path())
                    if j == 0:
                        temperatures.append(0)
                    else:
                        temperatures.append(TEMPERATURE)
            else:
                finished += 1
        
        print(f"iteration {i}")
        print(f"finished {finished} / {len(problems)}")

        if len(questions) == 0:
            break

        next_steps, next_values = call(questions, paths, temperatures, [STEP_STOP_TOKENS] * len(questions))
        for state, next_step, next_value in zip(anchors, next_steps, next_values):
            child = state.tree.expand_node(state, next_step, next_value, i + 1, assert_end(next_step))
            fix_value(child)

        pickle.dump(problems, open(output_fpath, "wb"))
