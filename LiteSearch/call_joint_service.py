from tqdm import tqdm
import requests
import json
import multiprocessing
import re
import numpy as np
import random
import time

class PolicyArgument:
    url='127.0.0.1'
    port=9000
    model_name="path/to/policy"
    max_tokens=512

class ValueArgument:
    url='127.0.0.1'
    port=9009

ADD_CALC = True

from string import Template

POLICY_INSTRUCTION = Template("""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a math problem solver. Given a problem, you should think step by step and then provide a numerical answer after `Answer:`.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nJen and Tyler are gymnasts practicing flips. Jen is practicing the triple-flip while Tyler is practicing the double-flip. Jen did sixteen triple-flips during practice. Tyler flipped in the air half the number of times Jen did. How many double-flips did Tyler do?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nLet's think step by step.\nJen did 16 triple-flips, so she did 16 * 3 = <<16*3=48>>48 flips.\nTyler did half the number of flips, so he did 48 / 2 = <<48/2=24>>24 flips.\nA double flip has two flips, so Tyler did 24 / 2 = <<24/2=12>>12 double-flips.\nAnswer: 12<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nFour people in a law firm are planning a party. Mary will buy a platter of pasta for $$20 and a loaf of bread for $$2. Elle and Andrea will split the cost for buying 4 cans of soda which cost $$1.50 each, and chicken wings for $$10. Joe will buy a cake that costs $$5. How much more will Mary spend than the rest of the firm put together?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nLet's think step by step.\nMary will spend $$20 + $$2 = $$<<20+2=22>>22.\nElle and Andrea will spend $$1.5 x 4 = $$<<1.5*4=6>>6 for the soda.\nElle and Andrea will spend $$6 + $$10 = $$<<6+10=16>>16 for the soda and chicken wings.\nElle, Andrea, and Joe together will spend $$16 + $$5 = $$<<16+5=21>>21.\nSo, Mary will spend $$22 - $$21 = $$<<22-21=1>>1 more than all of them combined.\nAnswer: 1<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nA charcoal grill burns fifteen coals to ash every twenty minutes of grilling. The grill ran for long enough to burn three bags of coals. Each bag of coal contains 60 coals. How long did the grill run?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nLet's think step by step.\nThe grill burned 3 * 60 = <<3*60=180>>180 coals.\nIt takes 20 minutes to burn 15 coals, so the grill ran for 180 / 15 * 20 = <<180/15*20=240>>240 minutes.\nAnswer: 240<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nA bear is preparing to hibernate for the winter and needs to gain 1000 pounds. At the end of summer, the bear feasts on berries and small woodland animals. During autumn, it devours acorns and salmon. It gained a fifth of the weight it needed from berries during summer, and during autumn, it gained twice that amount from acorns. Salmon made up half of the remaining weight it had needed to gain. How many pounds did it gain eating small animals?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nLet's think step by step.\nThe bear gained 1 / 5 * 1000 = <<1/5*1000=200>>200 pounds from berries.\nIt gained 2 * 200 = <<2*200=400>>400 pounds from acorns.\nIt still needed 1000 - 200 - 400 = <<1000-200-400=400>>400 pounds.\nThus, it gained 400 / 2 = <<400/2=200>>200 pounds from salmon.\nTherefore, the bear gained 400 - 200 = <<400-200=200>>200 pounds from small animals.\nAnswer: 200<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nBrendan can cut 8 yards of grass per day, he bought a lawnmower and it helped him to cut more yards by Fifty percent per day. How many yards will Brendan be able to cut after a week?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nLet's think step by step.\nThe additional yard Brendan can cut after buying the lawnmower is 8 x 0.50 = <<8*0.50=4>>4 yards.\nSo, the total yards he can cut with the lawnmower is 8 + 4 = <<8+4=12>>12.\nTherefore, the total number of yards he can cut in a week is 12 x 7 = <<12*7=84>>84 yards.\nAnswer: 84<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nSteve finds 100 gold bars while visiting Oregon. He wants to distribute his gold bars evenly to his 4 friends. If 20 gold bars were lost on the way back to San Diego, how many gold bars will each of his 4 friends get when he returns?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nLet's think step by step.\nHe only has 100 - 20 = <<100-20=80>>80 gold bars after losing 20 of them.\nHe then gives each of his friends 80 ÷ 4 = <<80/4=20>>20 gold bars.\nAnswer: 20<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nIn a week, Mortdecai collects 8 dozen  eggs every Tuesday and Thursday, and he delivers 3 dozen  eggs to the market and 5 dozen eggs to the mall. He then uses 4 dozen eggs to make a pie every Saturday. Mortdecai donates the remaining eggs to the charity by Sunday. How many  eggs does he donate to the charity?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nLet's think step by step.\nMortdecai collects a total of 8x2 = <<8*2=16>>16 dozens of eggs.\nHe sells a total of 3 + 5 = <<3+5=8>>8 dozens of eggs.\nSo, 16 - 8 = <<16-8=8>>8 dozens of eggs are left.\nAfter using 4 dozens of eggs to make a pie, 8 - 4 = <<8-4=4>>4 dozens of eggs are left.\nSince there are 12 in 1 dozen, then Mortdecai donates 4 x 12 = <<4*12=48>>48 pieces of eggs to the charity.\nAnswer: 48<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nCorey downloaded two movie series from his Netflix account with 12 and 14 seasons per series, respectively. However, in the week, his computer got a mechanical failure, and he lost two episodes from each season for both series. If each season in the movie series that Corey downloaded had 16 episodes, how many episodes remained after the computer's mechanical failure?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nLet's think step by step.\nIn the first movie series with 12 seasons, after the mechanical failure, the number of episodes that Corey lost is 2*12 = <<2*12=24>>24\nOriginally, the movie series with 12 seasons had 12*16 = <<12*16=192>>192 episodes.\nAfter the mechanical failure, Corey had 192-24 = <<192-24=168>>168 episodes remaining in the first movie series.\nSimilarly, the 14 season movie series also had 14*2 = <<14*2=28>>28 lost after the computer's mechanical failure.\nOriginally, the movie series with 14 seasons has 14*16 = <<14*16=224>>224 episodes.\nThe mechanical failure of the computer reduced the number of episodes in the 14 season movie series to 224-28 = <<224-28=196>>196\nAfter the loss, Corey had 196+168 = <<196+168=364>>364 episodes remaining from the two movie series he had downloaded.\nAnswer: 364<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n${question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nLet's think step by step.\n${path}""")
# POLICY_INSTRUCTION = Template("""Question: Jen and Tyler are gymnasts practicing flips. Jen is practicing the triple-flip while Tyler is practicing the double-flip. Jen did sixteen triple-flips during practice. Tyler flipped in the air half the number of times Jen did. How many double-flips did Tyler do?\nAnswer: Jen did 16 triple-flips, so she did 16 * 3 = <<16*3=48>>48 flips.\nTyler did half the number of flips, so he did 48 / 2 = <<48/2=24>>24 flips.\nA double flip has two flips, so Tyler did 24 / 2 = <<24/2=12>>12 double-flips.\n#### 12\n\nQuestion: Four people in a law firm are planning a party. Mary will buy a platter of pasta for $$20 and a loaf of bread for $$2. Elle and Andrea will split the cost for buying 4 cans of soda which cost $$1.50 each, and chicken wings for $$10. Joe will buy a cake that costs $$5. How much more will Mary spend than the rest of the firm put together?\nAnswer: Mary will spend $$20 + $$2 = $$<<20+2=22>>22.\nElle and Andrea will spend $$1.5 x 4 = $$<<1.5*4=6>>6 for the soda.\nElle and Andrea will spend $$6 + $$10 = $$<<6+10=16>>16 for the soda and chicken wings.\nElle, Andrea, and Joe together will spend $$16 + $$5 = $$<<16+5=21>>21.\nSo, Mary will spend $$22 - $$21 = $$<<22-21=1>>1 more than all of them combined.\n#### 1\n\nQuestion: A charcoal grill burns fifteen coals to ash every twenty minutes of grilling. The grill ran for long enough to burn three bags of coals. Each bag of coal contains 60 coals. How long did the grill run?\nAnswer: The grill burned 3 * 60 = <<3*60=180>>180 coals.\nIt takes 20 minutes to burn 15 coals, so the grill ran for 180 / 15 * 20 = <<180/15*20=240>>240 minutes.\n#### 240\n\nQuestion: A bear is preparing to hibernate for the winter and needs to gain 1000 pounds. At the end of summer, the bear feasts on berries and small woodland animals. During autumn, it devours acorns and salmon. It gained a fifth of the weight it needed from berries during summer, and during autumn, it gained twice that amount from acorns. Salmon made up half of the remaining weight it had needed to gain. How many pounds did it gain eating small animals?\nAnswer: The bear gained 1 / 5 * 1000 = <<1/5*1000=200>>200 pounds from berries.\nIt gained 2 * 200 = <<2*200=400>>400 pounds from acorns.\nIt still needed 1000 - 200 - 400 = <<1000-200-400=400>>400 pounds.\nThus, it gained 400 / 2 = <<400/2=200>>200 pounds from salmon.\nTherefore, the bear gained 400 - 200 = <<400-200=200>>200 pounds from small animals.\n#### 200\n\nQuestion: Brendan can cut 8 yards of grass per day, he bought a lawnmower and it helped him to cut more yards by Fifty percent per day. How many yards will Brendan be able to cut after a week?\nAnswer: The additional yard Brendan can cut after buying the lawnmower is 8 x 0.50 = <<8*0.50=4>>4 yards.\nSo, the total yards he can cut with the lawnmower is 8 + 4 = <<8+4=12>>12.\nTherefore, the total number of yards he can cut in a week is 12 x 7 = <<12*7=84>>84 yards.\n#### 84\n\nQuestion: Steve finds 100 gold bars while visiting Oregon. He wants to distribute his gold bars evenly to his 4 friends. If 20 gold bars were lost on the way back to San Diego, how many gold bars will each of his 4 friends get when he returns?\nAnswer: He only has 100 - 20 = <<100-20=80>>80 gold bars after losing 20 of them.\nHe then gives each of his friends 80 ÷ 4 = <<80/4=20>>20 gold bars.\n#### 20\n\nQuestion: In a week, Mortdecai collects 8 dozen  eggs every Tuesday and Thursday, and he delivers 3 dozen  eggs to the market and 5 dozen eggs to the mall. He then uses 4 dozen eggs to make a pie every Saturday. Mortdecai donates the remaining eggs to the charity by Sunday. How many  eggs does he donate to the charity?\nAnswer: Mortdecai collects a total of 8x2 = <<8*2=16>>16 dozens of eggs.\nHe sells a total of 3 + 5 = <<3+5=8>>8 dozens of eggs.\nSo, 16 - 8 = <<16-8=8>>8 dozens of eggs are left.\nAfter using 4 dozens of eggs to make a pie, 8 - 4 = <<8-4=4>>4 dozens of eggs are left.\nSince there are 12 in 1 dozen, then Mortdecai donates 4 x 12 = <<4*12=48>>48 pieces of eggs to the charity.\n#### 48\n\nQuestion: Corey downloaded two movie series from his Netflix account with 12 and 14 seasons per series, respectively. However, in the week, his computer got a mechanical failure, and he lost two episodes from each season for both series. If each season in the movie series that Corey downloaded had 16 episodes, how many episodes remained after the computer's mechanical failure?\nAnswer: In the first movie series with 12 seasons, after the mechanical failure, the number of episodes that Corey lost is 2*12 = <<2*12=24>>24\nOriginally, the movie series with 12 seasons had 12*16 = <<12*16=192>>192 episodes.\nAfter the mechanical failure, Corey had 192-24 = <<192-24=168>>168 episodes remaining in the first movie series.\nSimilarly, the 14 season movie series also had 14*2 = <<14*2=28>>28 lost after the computer's mechanical failure.\nOriginally, the movie series with 14 seasons has 14*16 = <<14*16=224>>224 episodes.\nThe mechanical failure of the computer reduced the number of episodes in the 14 season movie series to 224-28 = <<224-28=196>>196\nAfter the loss, Corey had 196+168 = <<196+168=364>>364 episodes remaining from the two movie series he had downloaded.\n#### 364\n\nQuestion: ${question}\nAnswer: ${path}""")
# POLICY_INSTRUCTION = Template("""Question: ${question} Answer: ${path}""")

# VALUE_INSTRUCTION = Template("Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nYou are given a math problem, followed by a step-by-step reasoning process. Your task is to read the problem carefully, understand and follow the solving steps, then predict the chance of generating correct final answer with your internal knowledge. Output 'True' if you think the final answer will be correct, and 'False' otherwise.\n\n### Input:\nQuestion: ${question}\nAnswer: ${path}")
VALUE_INSTRUCTION = Template("Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nYou are given a math problem, followed by a step-by-step reasoning process. Your task is to read the problem carefully, understand and follow the solving steps, then predict the chance of generating correct final answer with your internal knowledge. Output 'True' if you think the final answer will be correct, and 'False' otherwise.\n\n### Input:\n${question}\n${path}")


def add_calculator(step):
    p3 = re.compile('<<[^<>]*>>')
    p4 = re.compile('<<([^=<>]*)=([^=<>]*)>>')
    raw_expr_list = re.findall(p3, step)

    for opseq_text in raw_expr_list:
        filtered_opseq_text = opseq_text.replace("$$", "").replace(",", "").replace("[", "(").replace("]", ")").replace("^", "**")
        m = p4.match(filtered_opseq_text)
        if m:
            v0, v1 = m.group(1, 2)
            if "%" in v0 and "+" not in v0 and "/" not in v0:
                v0 = v0.replace("%", "*0.01")
            try:
                # without any unknowns
                step = step[:step.find(opseq_text)+len(opseq_text)] + "<calc>" + str(eval(v0))+"</calc>" + step[step.find(opseq_text)+len(opseq_text):]
            except:
                try:
                    expression = convert_to_left_equation(filtered_opseq_text.replace("<<","").replace(">>",""))
                    variables = sympify(expression).free_symbols
                    result = solve(expression)
                    # currently only consider one unknown
                    if len(result) == 1 and len(variables) == 1:
                        variable = str(variables.pop())
                        # to filter unit word ("pill", "hour")
                        if len(variable) == 1:
                            short_result = str(result[-1])
                            if is_float(result[-1]):
                                short_result = str(eval(str(result[-1])))
                            step = (step[:step.find(opseq_text) + len(opseq_text)] + "<calc>" + variable + "=" +
                                    short_result + "</calc>" + step[step.find(opseq_text) + len(opseq_text):])
                except:
                    # Replace instances of '2x' with '2*x' using regular expressions
                    # opseq_text_new = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', opseq_text)
                    continue

    return step


def add_calculator_for_trajecotry(cur_answer):
        """
        Add calculator for each step in the trajectory
        """
        split_sign = "\n"
        cur_answer_step = cur_answer.split(split_sign)
        n_current_steps = len(cur_answer_step)

        answer_w_calculator = []
        for i in range(n_current_steps):
            if len(cur_answer_step[i]) == 0:
                continue
            step_w_calculator = add_calculator(cur_answer_step[i])
            answer_w_calculator.append(step_w_calculator)
        return split_sign.join(answer_w_calculator)

def wrap_query_for_policy(query, path):
    wrapped_query = POLICY_INSTRUCTION.substitute(question=query, path=path+"\n" if path else "")
    if wrapped_query[-1] != "\n":
        wrapped_query = wrapped_query.strip()
    return wrapped_query

def wrap_query_for_value(query, path):
    return VALUE_INSTRUCTION.substitute(question=query, path=path+"\n" if path else "")

class Worker(object):

    def __init__(self, policy_args, value_args):
        self.policy_args = policy_args
        self.policy_url = f"http://{policy_args.url}:{policy_args.port}/v1/completions"
        self.value_args = value_args
        self.value_url = f"http://{value_args.url}:{value_args.port}/predict"
    
    def encode_wrapper(self, args):
        return self.encode(*args)
    
    def encode(self, question, path, temp=1.0, stop=[]):
        headers ={"User-Agent":"Test Client"}

        if temp is not None: # None, only call value
            query = wrap_query_for_policy(question, path)
            pload ={"prompt": query, "model": self.policy_args.model_name, "temperature": temp,
                    "max_tokens": self.policy_args.max_tokens, "stop": stop}

            response =requests.post(self.policy_url, headers=headers, json=pload)
            
            try:
                next_step = json.loads(response.content)["choices"][0]["text"].strip()
            except:
                print("Policy Server Error")
                return "", 0 # usually because over-length
        else:
            next_step = ""

        new_path = path + "\n" + next_step if path else next_step
        if ADD_CALC:
            answer_w_calc = add_calculator_for_trajecotry(new_path).strip()
            pload ={"texts": [wrap_query_for_value(question, answer_w_calc)]}
        else:
            pload ={"texts": [wrap_query_for_value(question, new_path)]}
        
        response = requests.post(self.value_url, headers=headers, json=pload)
        
        try:
            value = min(max((response.json()["values"][0] + 1.) / 2., 0.), 1.)
        except:
            print("Value Server Error")
            return "", 0 # usually because oom

        return next_step, value


def call(questions, paths, temperatures, stops):
    policy_args = PolicyArgument()
    value_args = ValueArgument()
    worker = Worker(policy_args, value_args)
    pool = multiprocessing.Pool(80)
    model_outputs = list(tqdm(pool.imap(worker.encode_wrapper, [(a, b, c, d) for a, b, c, d in zip(questions, paths, temperatures, stops)], 4), total=len(questions)))
    next_steps, values = zip(*model_outputs)
    return next_steps, values
