import json

input_file = "arxiv-metadata-oai-snapshot.json"
output_file = "arxiv-csAI.jsonl"

with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        record = json.loads(line)
        cats = record.get("categories", "").split()
        if "cs.AI" in cats:
            fout.write(json.dumps(record) + "\n")

print("✅ File creato:", output_file)
