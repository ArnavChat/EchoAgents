import json
import os

def convert_alpaca_format(input_path, output_path):
    """Convert Alpaca dataset to our format."""
    print(f"Converting {input_path} to {output_path}...")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    converted_data = {"data": []}
    
    for item in data:
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output = item.get("output", "")
        
        # Format as a conversation
        if input_text:
            conversation = f"User: {instruction} {input_text}\nAssistant: {output}"
        else:
            conversation = f"User: {instruction}\nAssistant: {output}"
        
        converted_data["data"].append({"text": conversation})
    
    print(f"Converted {len(converted_data['data'])} examples")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2)
    
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    input_path = "data/training/alpaca_raw.json"
    output_path = "data/training/alpaca.json"
    
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found")
        exit(1)
        
    convert_alpaca_format(input_path, output_path)
