import json
import copy
with open("gsm8k_train_easy_2500.json") as f:
    easy_data = json.load(f)
with open("gsm8k_train_hard_2500.json") as f:
    hard_data = json.load(f)
with open("dataset_info.json") as f:
    info = json.load(f)


for i in range(0,11):
    new_data = easy_data[:i*250]+hard_data[:2500-i*250]
    new_info = copy.deepcopy(info['gsm8k_train'])
    new_info['file_name'] = f"gsm8k_train_easy_{i*250}_hard_{2500-i*250}.json"
    info[f"gsm8k_train_easy_{i*250}_hard_{2500-i*250}"] = new_info
    with open(f"gsm8k_train_easy_{i*250}_hard_{2500-i*250}.json",'w') as f:
        json.dump(new_data,f,indent=1)
with open("dataset_info_.json",'w') as f:
    json.dump(info,f,indent=1)

