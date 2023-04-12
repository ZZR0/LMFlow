import json
import os
import random

os.chdir(os.path.dirname(os.path.abspath(__file__)))

output_dir = "../data/koala/"

def generate_data(split):
    input_dir = f"/data1/BigCodeModel/koala_data_pipeline/data/language/chat/instruct_po_v3_{split}.jsonl"
    data = []
    with open(input_dir, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    random.shuffle(data)
    
    if split == "eval":
        split = "test"
    os.makedirs(output_dir+split, exist_ok=True)
    new_data = {"type": "chat_list", "instances": []}
    for d in data:
        if d["fields"] == "": continue
        new_data["instances"].append({"chat":d})
    # with open(output_dir + f"{split}/{split}_{len(new_data['instances'])}.json", "w") as f:
    with open(output_dir + f"{split}/{split}.json", "w") as f:
        json.dump(new_data, f, indent=4)

if __name__ == "__main__":
    generate_data("train")
    generate_data("eval")