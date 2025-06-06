import argparse
import re
import os
import json
import logging
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from EasyChatTemplating.util_tools import convert_userprompt_transformers, skip_special_tokens_transformers

result_pattern = r'\{.*\}'
valid_pattern = r'\[\[(.*?)\]\]'
model_path_dict = {"llama3-chat": "../../pretrained_models/llama3-chat"}
dataset_path_dict = {"conll2003": "./datasets/conll2003",
                     "ace04": "./datasets/ace04",
                     "ace05": "./datasets/ace05",
                     "genia": "./datasets/genia"}

conll2003_prompt = """Task: Summarize the generic rules for each named entity category for the named entity recognition task based on the provided text and their corresponding annotations. The output must be structured in JSON format, where the keys represent the entity categories, and the values are lists of rules that have been summarized from the input text and their annotations.

Guidelines: 
(1) Avoid including specific entity names in the output and instead describe general patterns or features. 
(2) Only summarize rules for the entity categories that appear in the provided annotations. Do not include rules for any other categories.
(3) For each annotation provided, generate exactly one summarized rule corresponding to that label.
(4) The order of the summarized rules should strictly correspond to the order of the annotations, and the number of summarized rules must match the number of annotations.

Examples: 
Input Text: EU rejects German call to boycott British lamb . 
Annotations: [["EU", "organization"], ["German", "miscellaneous"], ["British", "miscellaneous"]]. 
Output: {{"organization": ["union"], "miscellaneous": ["ethnic groups", "ethnic groups"]}}
Input Text: Iraq 's Saddam meets Russia 's Zhirinovsky .
Annotations: [["Iraq", "location"], ["Saddam", "person"], ["Russia", "location"], ["Zhirinovsky", "person"]]
Output: {{"location": ["country", "country"], "person": ["name", "name"]}}
Input Text: S&P = DENOMS ( K ) 1-10-100 SALE LIMITS US / UK / CA
Annotations: [["S&P", "organization"], ["US", "location"], ["UK", "location"], ["CA", "location"]]
Output: {{"organization": ["financial institution"], "location": ["country", "country", "country"]}}

Summarize for:
Input Text: {input_text}
Annotations: {input_annotations}
Output:
"""

conll2003_valid_prompt= """Task: Please identify Person, Organization, Location and Miscellaneous Entity from the given text and rules.
The rules are in JSON format where the key is the entity category and the value is the schema contained in that category.

Examples:
Input Text: EU rejects German call to boycott British lamb.
Rules: {{"organization": ["union"], "miscellaneous": ["nationality"]}}
Output: [["EU", "organization"], ["German", "miscellaneous"], ["British", "miscellaneous"]]
Input Text: S&P = DENOMS ( K ) 1-10-100 SALE LIMITS US / UK / CA
Rules: {{"organization": ["financial institution"], "location": ["country", "country", "country"]}}
Output: [["Iraq", "location"], ["Saddam", "person"], ["Russia", "location"], ["Zhirinovsky", "person"]]
Input Text: -- E. Auchard , Wall Street bureau , 212-859-1736
Rules: {{"person": ["journalist"], "organization": ["newspaper bureau"]}}
Output: [["E. Auchard", "person"], ["Wall Street bureau", "organization"]]

Instructions:

Input Text: {input_text}
Rules: {summarized_rules}
Output:
"""

# Wheather labels and result are equal and correspoding
def type_num_equal(labels, result):
    labels_len = len(labels)
    result_len = 0
    for k, v in result.items():
        result_len += len(v)
    if labels_len != result_len:
        return False
    
    tmp_dict = {}
    for label in labels:
        label_type = label[-1]
        if label_type not in tmp_dict:
            tmp_dict[label_type] = 0
        tmp_dict[label_type] += 1
        
    tmp_dict2 ={}
    for k,v in result.items():
        if k not in tmp_dict2:
            tmp_dict2[k] = 0
        tmp_dict2[k] += len(v)
    
    for k,v in tmp_dict.items():
        if k not in tmp_dict2:
            return False
        if v != tmp_dict2[k]:
            return False
    
    return True
    
    
# get the label and result correspondings list
def correspondings(labels, result):
    label_type_dict = {}
    final_result = []
    for label in labels:
        label_type = label[-1]
        if label_type not in label_type_dict:
            label_type_dict[label_type] = 0
        label_type_dict[label_type] += 1
        
        idx = label_type_dict[label_type] - 1
        rule = result[label_type][idx]

        final_result.append([label, rule])
    return final_result
        



def summary(rule_file_name, label_file, fw, top_k=20):
    result_dict = {}
    with open(label_file, 'r', encoding='utf8') as f:
        labels_dict = f.readlines()[0]
        labels_dict = json.loads(labels_dict)
        for k in labels_dict:
            if "geo" in k:
                k = "geo-political entity"
            result_dict[k] = {}

        
    with open(rule_file_name, 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            line_json = json.loads(line)
            if "right_rules" not in line_json:
                continue
            right_rules = line_json["right_rules"]
            if len(right_rules) == 0:
                continue
            for right_rule in right_rules:
                for k, v in right_rule.items():
                    if "geo" in k:
                        k = "geo-political entity" 
                    entity_type = k
                    rule = v
                    if rule not in result_dict[entity_type]:
                        result_dict[entity_type][rule] = 0
                    result_dict[entity_type][rule] += 1   
    rules_dict = {} 
    for k in result_dict:
        rules_dict[k] = []
        tmp_list = sorted(result_dict[k].items(), key=lambda x:x[-1], reverse=True)
        for j, tmp in enumerate(tmp_list):
            if j > top_k:
                break
            rules_dict[k].append(tmp[0])
     
    fw.write(json.dumps(rules_dict))
    fw.close()

            

def predict_batch(outputs, tokenizer, fw, texts, labels):
    for j, output in enumerate(outputs):
        clean_text = skip_special_tokens_transformers(tokenizer, output.outputs[0].text)
        result = re.search(result_pattern, clean_text, re.DOTALL)
        
        result_dict = {}
        result_dict["text"] = texts[j]
        result_dict["labels"] = labels[j]
        
        # If llm generate the right result
        if result is not None:
            try:
                result = eval(result.group())
                result_dict["status"] = "success"
                result_dict["predicted_rules"] = result
            except:
                result_dict["status"] = "eval_wrong"
                result_dict["predicted_rules"] = []
        # if llm generate the wrong result or generate nothing
        else:
            result_dict["status"] = "none_wrong"
            result_dict["predicted_rules"] = []
        try:
            fw.write(json.dumps(result_dict))
            fw.write("\n")
            fw.flush()
        except:
            result_dict["status"] = "write_wrong"
            result_dict["predicted_rules"] = []
            fw.write(json.dumps(result_dict))
            fw.write("\n")
            fw.flush()
            
def valid_batch(outputs, tokenizer, fw, texts, labels, rules_list):
    for j, output in enumerate(outputs):
        text = texts[j]
        real_labels = labels[j]
        rules = rules_list[j]
        corres = correspondings(real_labels, rules)
        right_rules = []
        wrong_rules = []
        
        result_dict = {}

        clean_text = skip_special_tokens_transformers(tokenizer, output.outputs[0].text)
        result = re.search(valid_pattern, clean_text, re.DOTALL)
        result_dict["text"] = text
        result_dict["label"] = real_labels
        result_dict["orignal_rules"] = rules
        
        if result is not None:
            try:
                # [[entity_name, entity_type]]
                result = eval(result.group())
                for i in range(len(result)):
                    if "geo" in result[i][-1]:
                        result[i][-1] = "geo-political entity"
                    
                # [(entity_name, entity_type), pattern]
                for k, cor in enumerate(corres):
                    label = cor[0]
                    type = label[-1]
                    if "geo" in type:
                        type = "geo-political entity"
                        label[-1] = "geo-political entity"
                    rules = {type:cor[-1]}
                    if label in result:
                        right_rules.append(rules)
                    else:
                        wrong_rules.append(rules)
                
                result_dict["right_rules"] = right_rules
                result_dict["wrong_rules"] = wrong_rules
                result_dict["status"] = "success"
                result_dict["predict_labels"] = result
            except:
                result_dict["status"] = "eval_wrong"
                result_dict["predict_labels"] = []
        else:
            result_dict["status"] = "none_wrong"
            result_dict["predict_labels"] = []
            
        try:
            fw.write(json.dumps(result_dict))
            fw.write("\n")
            fw.flush()
        except:
            result_dict["status"] = "write_wrong"
            result_dict["predict_labels"] = []
            fw.write(json.dumps(result_dict))
            fw.write("\n")
            fw.flush()
            
def valied_rules(fr, fw, batch_size, valid_prompt, tokenizer, llm, sampling_params):
    messages = []
    texts = []
    labels = []
    rules_list = []
    for i, line in enumerate(fr):
        line_json = json.loads(line)
        result_dict = {}
        text = line_json["text"]
        entity_labels = line_json["labels"]
        rules = line_json["predicted_rules"]
        
        if len(entity_labels) == 0:
            continue
        if not isinstance(rules, dict):
            continue
        if not type_num_equal(entity_labels, rules):
            continue
        
        prompt_predict = valid_prompt.format(except_rules="", input_text = text, summarized_rules = rules)
        message = convert_userprompt_transformers(tokenizer, prompt_predict, add_generation_prompt=True)
        
        if len(messages) < batch_size - 1:
            texts.append(text)
            labels.append(entity_labels)
            messages.append(message)
            rules_list.append(rules)
        else:
            texts.append(text)
            labels.append(entity_labels)
            messages.append(message)
            rules_list.append(rules)
           
            outputs = llm.generate(messages, sampling_params)
            valid_batch(outputs, tokenizer, fw, texts, labels, rules_list)
            messages = []
            texts = []
            labels = []
            rules_list = []
    
    if len(messages) > 0:
        outputs = llm.generate(messages, sampling_params)
        valid_batch(outputs, tokenizer, fw, texts, labels, rules_list)
    
    fw.close()
            
            
        


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name',
                        default='conll2003',
                        choices=["conll2003", "ace04", "ace05", "genia"])
    parser.add_argument('--model_name',
                        default='llama3-chat')
    parser.add_argument('--temperature',
                        default=0.8,
                        type=float),
    parser.add_argument('--top_p',
                        default=0.95,
                        type=float),
    batch_size = 32
    args = parser.parse_args()
    
    model_path = model_path_dict[args.model_name]
    dataset_path = dataset_path_dict[args.dataset_name]
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model=model_path)
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=256)
    
    rule_file_name = os.path.join(dataset_path, f"{args.model_name}_rules.txt")
    valid_rule_file_name = os.path.join(dataset_path, f"{args.model_name}_validrules.txt")
    label_file = os.path.join(dataset_path, "labels.jsonl")
    summary_file_name = os.path.join(dataset_path, f"{args.model_name}_summaryrules.txt")
    fw = open(rule_file_name, "a", encoding='utf8')
    
    messages = []
    texts = []
    labels = []
    
    task_prompt = eval(f"{args.dataset_name}_prompt")
    valid_prompt = eval(f"{args.dataset_name}_valid_prompt")
    
    with open(os.path.join(dataset_path, "train.jsonl"), "r", encoding='utf8') as f:
        for i, line in tqdm(enumerate(f)):
            line_json = json.loads(line)
            
            text = line_json["text"]
            entity_labels = line_json["entity_labels"]
            
            if len (entity_labels) == 0:
                continue
            
            prompt_predict = task_prompt.format(input_text = text, input_annotations = entity_labels)
            message = convert_userprompt_transformers(tokenizer, prompt_predict, add_generation_prompt=True)
            
            if len(messages) < batch_size - 1:
                texts.append(text)
                labels.append(entity_labels)
                messages.append(message)
            else:
                texts.append(text)
                labels.append(entity_labels)
                messages.append(message)
                
                outputs = llm.generate(messages, sampling_params)
                
                predict_batch(outputs, tokenizer, fw, texts, labels)
                
                messages = []
                texts = []
                labels = []
        
        if len(messages) > 0:
            outputs = llm.generate(messages, sampling_params)
            predict_batch(outputs, tokenizer, fw, texts, labels)
        
        fw.close()
    
    fr = open(rule_file_name, 'r', encoding='utf8')
    fw = open(valid_rule_file_name, 'a', encoding='utf8')
    
    valied_rules(fr, fw, batch_size, valid_prompt, tokenizer, llm, sampling_params)
    
    fw = open(summary_file_name, 'a', encoding='utf8')
    summary(valid_rule_file_name, label_file, fw)
    
    
if __name__ == "__main__":
    main()