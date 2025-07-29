import csv
import json
import chardet

csv_file = 'data/training/csv/nlpMentalHealthConversation.csv'
json_file = 'data/training/json/nlpMentalHealthConversation.json'

# Detect encoding
with open(csv_file, "rb") as f:
    rawdata = f.read(10000)
    result = chardet.detect(rawdata)
    encoding = result['encoding']
    print(f"Detected encoding: {encoding}")

# Read the CSV with detected encoding
with open(csv_file, mode='r', encoding=encoding, errors='replace') as f:
    reader = csv.DictReader(f)
    data = list(reader)

# Write to JSON
with open(json_file, mode='w', encoding='utf-8') as f:
    json.dump(data, f, indent=4)

print(f"Converted {csv_file} to {json_file}")
