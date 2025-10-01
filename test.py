import requests

payload = {
    "request_id": 1,
    "query": "Provide accident history for forklift operations",
    "type": 2
}
resp = requests.post("http://localhost:8000/genai/doosan_chatbot/", json=payload)
print(resp.json())
