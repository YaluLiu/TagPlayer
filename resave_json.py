import json
import os

def get_all_json(root_dir):
    jsons = []
    for root,dirs,names in os.walk(root_dir):
        for filename in names:
            if filename.endswith(".json"):
                jsons.append(os.path.join(root,filename))
    return jsons


if __name__ == "__main__":
    jsons = get_all_json("image")
    for json_path in jsons:
        with open(json_path,"r") as f:
            json_data = json.load(f)
        with open(json_path,"w") as f:
            json.dump(json_data,f,indent=4)