import json
p='tools/extracted_config.json'
with open(p,'r',encoding='utf-8') as f:
    data=json.load(f)

names=set()

def walk(obj):
    if isinstance(obj,dict):
        if 'class_name' in obj:
            names.add(obj['class_name'])
        for v in obj.values():
            walk(v)
    elif isinstance(obj,list):
        for v in obj:
            walk(v)

walk(data)
for n in sorted(names):
    print(n)
