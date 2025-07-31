import json
import os

print("Combining Alpaca dataset with custom dataset...")

# Load the alpaca dataset
with open("data/training/alpaca.json", "r", encoding="utf-8") as f:
    alpaca_data = json.load(f)
    print(f"Loaded {len(alpaca_data['data'])} examples from Alpaca dataset")

# Load the custom dataset
with open("data/training/custom_conversations.json", "r", encoding="utf-8") as f:
    custom_data = json.load(f)
    print(f"Loaded {len(custom_data['data'])} examples from custom dataset")

# Combine them
combined_data = {"data": alpaca_data["data"] + custom_data["data"]}
print(f"Combined dataset has {len(combined_data['data'])} examples")

# Save the combined dataset
output_path = "data/training/combined_dataset.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(combined_data, f, indent=2)
    
print(f"Combined dataset saved to {output_path}")
