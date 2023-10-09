import json
file1 = "task2error.json"
file2 = ".workspace/cm/task_runtime_cache_small.json"

with open(file1, 'r') as fp:
    json1 = json.load(fp)

with open(file2, 'r') as fp:
    json2 = json.load(fp)

print(len(json1), len(json2))
for _k in json1:
    if _k not in json2:
        json2[_k] = json1[_k]
    else:
        json2[_k]["error"] = min(json2[_k]["error"], json1[_k]["error"])

print(len(json2))

with open(file2, 'w') as fp:
    json.dump(json2, fp, indent=4)