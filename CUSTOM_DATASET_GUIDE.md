# Building Your Own Custom Language Model Dataset

This guide will help you create a high-quality dataset for training your voice assistant's language model.

## Understanding Dataset Formats

For training your own model, your dataset should be formatted as a JSON file with conversation examples:

```json
{
  "data": [
    {
      "text": "User: What's the weather today?\nAssistant: It's sunny with a high of 75°F."
    },
    {
      "text": "User: Tell me a joke.\nAssistant: Why don't scientists trust atoms? Because they make up everything!"
    }
  ]
}
```

Each example should include a user query and the desired assistant response.

## Creating Your Own Dataset

### Step 1: Identify Your Use Cases

Before creating examples, identify what you want your assistant to do:

- Answer factual questions
- Provide conversational responses
- Perform specific tasks (weather, jokes, recipes, etc.)
- Handle specific domain knowledge

### Step 2: Write Diverse Examples

Create examples covering:

- **Different question types**: What, where, how, why, etc.
- **Different response lengths**: Short answers and detailed explanations
- **Different tones**: Formal, casual, friendly, professional
- **Edge cases**: Clarifying questions, handling ambiguity

### Step 3: Organizing Your Dataset

1. Create a new file `custom_conversations.json` in the `data/training` directory
2. Follow the format with User/Assistant exchanges
3. Aim for at least 50-100 examples for basic training

## Using Existing Datasets as a Starting Point

You can use the download_dataset.py script to get existing datasets:

```powershell
python download_dataset.py --dataset alpaca --output_dir data/training
```

Then you can combine them with your own examples or modify them:

```python
# Example code to combine datasets
import json

# Load the downloaded dataset
with open("data/training/alpaca.json", "r", encoding="utf-8") as f:
    alpaca_data = json.load(f)

# Load your custom dataset
with open("data/training/custom_conversations.json", "r", encoding="utf-8") as f:
    custom_data = json.load(f)

# Combine them
combined_data = {"data": alpaca_data["data"] + custom_data["data"]}

# Save the combined dataset
with open("data/training/combined_dataset.json", "w", encoding="utf-8") as f:
    json.dump(combined_data, f, indent=2)
```

## Dataset Examples by Category

Here are example conversations by category that you can use as templates:

### General Knowledge

```
User: What is the capital of Japan?
Assistant: The capital of Japan is Tokyo. It's the largest city in Japan and one of the most populous metropolitan areas in the world.

User: How many planets are in our solar system?
Assistant: There are eight planets in our solar system: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. Pluto was reclassified as a dwarf planet in 2006.
```

### Task-Based

```
User: Set an alarm for 7 AM tomorrow
Assistant: I've set an alarm for 7:00 AM tomorrow. Is there anything else you'd like me to help you with?

User: Remind me to call mom at 6 PM
Assistant: I'll remind you to call mom at 6:00 PM today. Your reminder has been set.
```

### Personal Assistant

```
User: What's on my calendar for today?
Assistant: You have a meeting at 10 AM with the marketing team, lunch with Sarah at 12:30 PM, and a dentist appointment at 3 PM.

User: What's the weather forecast for tomorrow?
Assistant: Tomorrow will be partly cloudy with a high of 72°F and a low of 58°F. There's a 30% chance of rain in the afternoon.
```

## Tips for Creating High-Quality Examples

1. **Be specific**: Include details in responses that directly address the query
2. **Be consistent**: Maintain a consistent voice and style across examples
3. **Be diverse**: Include a wide range of topics and question types
4. **Be natural**: Write conversations as they would naturally occur
5. **Include follow-ups**: Add examples of multi-turn conversations

## Next Steps

After creating your dataset:

1. Run the training script:

```powershell
python create_custom_model.py --dataset data/training/your_dataset.json --epochs 5
```

2. Update your config to use the custom model:

```python
USE_CUSTOM_MODEL = True
CUSTOM_MODEL_PATH = "models/custom_llm"
CUSTOM_TOKENIZER_PATH = "models/custom_llm"
```

3. Run your voice assistant with your new model:

```powershell
python main.py
```
