import json
from django.http import JsonResponse

def reddit_opinion_view(request):
    json_file_path = r"D:\Code Files\EchoAgents\data\reddit_opinion_democrats.json"

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return JsonResponse(data, safe=False)
    except FileNotFoundError:
        return JsonResponse({"error": "File not found."}, status=404)
