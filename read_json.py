import json

# Replace with your actual JSON file path
json_file_path = r"D:\Code Files\EchoAgents\data\reddit_opinion_democrats.json"

try:
    with open(json_file_path, 'r', encoding='utf-8') as file:
        for _ in range(100):
            print(file.readline())

except FileNotFoundError:
    print("❌ File not found at:", json_file_path)
except json.JSONDecodeError as e:
    print("❌ JSON decoding error:", str(e))
except Exception as e:
    print("❌ An unexpected error occurred:", str(e))
