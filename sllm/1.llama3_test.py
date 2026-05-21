import requests

prompt = """Please briefly tell me about three uses of LLM in three lines or less."""

res = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "llama3.2", "prompt": prompt, "stream": False},
    timeout=300,
)
print(res.json()["response"])