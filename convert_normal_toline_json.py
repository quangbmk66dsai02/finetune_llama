import json

with open('data/combined-5k.json', 'r', encoding='utf-8-sig') as infile:
    data = json.load(infile)

with open('data/combined-5k-line.jsonl', 'w', encoding='utf-8') as outfile:
    for entry in data:
        json_line = json.dumps(entry, ensure_ascii=False)
        outfile.write(json_line + '\n')
