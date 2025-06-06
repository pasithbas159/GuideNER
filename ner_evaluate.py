import argparse
import os
import json
from transformers import AutoTokenizer

class Evaluate():
    def __init__(self) -> None:
        self.correct_preds = 0
        self.total_correct = 0
        self.total_preds = 0
        
        
    def update(self, label_preds, labels, labels_golden):
        
        for i in range(len(labels_golden)):
            labels_golden[i] = labels_golden[i].lower()
        tmp_label_preds = label_preds[:]
        label_preds = []
        for label in tmp_label_preds:
            if len(label) != 2:
                return
            for l in label:
                if not isinstance(l, str):
                    return
            if label[-1].lower() not in labels_golden:
                return
            
            label_preds.append(label)
            
        for label in labels:
            label[0] = label[0].lower()
            label[-1] = label[-1].lower()
                
        result_preds = []
        
        for label_pred in label_preds:
            label_pred[0] = label_pred[0].lower()
            label_pred[-1] = label_pred[-1].lower()
            if "geo" in label_pred[-1]:
                label_pred[-1] = "geopolitical entity" 
            result_preds.append(label_pred)
        
        for label_pred in result_preds:
            if label_pred in labels:
                self.correct_preds += 1
        
        self.total_correct += len(labels)
        self.total_preds += len(result_preds)
        
    def evaluate(self):
        p = self.correct_preds / self.total_preds
        r = self.correct_preds / self.total_correct
        f1 = 2 * p * r / (p + r)
        
        return p, r, f1



model_path_dict = {"llama3-chat": "../../pretrained_models/llama3-chat"}
dataset_path_dict = {"conll2003": "./datasets/conll2003",
                     "ace04": "./datasets/ace04",
                     "ace05": "./datasets/ace05",
                     "genia": "./datasets/genia"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name',
                        default='conll2003',
                        choices=["conll2003", "ace04", "ace05", "genia"])
    parser.add_argument('--model_name',
                        default='llama3-chat')
    args = parser.parse_args()
    
    evaluator = Evaluate()
    
    eval_file = os.path.join("./datasets", args.dataset_name, f"{args.model_name}_withrule_result_detail.txt")
    labels_golden = []
    label_file = os.path.join("./datasets", args.dataset_name, "labels.jsonl")
    with open(label_file, 'r', encoding='utf8') as f:
        tmp_labels = json.loads(f.readlines()[0])
        for k,v in tmp_labels.items():
            labels_golden.append(k)
    
    with open(eval_file, "r", encoding='utf8') as f:
        count = 0
        for i, line in enumerate(f):
            line_eval = json.loads(line)
            status = line_eval["status"]
            if status != "success":
                count += 1
                continue
            labels_pred = line_eval["predicted_labels"]
            labels = line_eval["labels"]
                
            evaluator.update(labels_pred, labels, labels_golden)
        p, r, f1 = evaluator.evaluate()
        print(f"p: {p}, r: {r}, f1: {f1}")
        print(count)
if __name__ == "__main__":
    main()
