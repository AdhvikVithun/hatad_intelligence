import httpx

# Test 1: think=False (Ollama API parameter)
body = {
    "model": "qwen3-vl:8b",
    "messages": [{"role": "user", "content": "Return only valid JSON with key greeting set to hello"}],
    "stream": False,
    "options": {"num_predict": 256, "temperature": 0.1},
    "think": False,
}
r = httpx.post("http://localhost:11434/api/chat", json=body, timeout=60)
d = r.json()
m = d["message"]
print("=== Test 1: think=False ===")
print("content:", repr(m.get("content", "")[:300]))
print("thinking:", repr(m.get("thinking", "")[:200]))
print("eval_count:", d.get("eval_count"))
print()

# Test 2: No think param at all
body2 = {
    "model": "qwen3-vl:8b",
    "messages": [{"role": "user", "content": "Return only valid JSON with key greeting set to hello"}],
    "stream": False,
    "options": {"num_predict": 256, "temperature": 0.1},
}
r2 = httpx.post("http://localhost:11434/api/chat", json=body2, timeout=60)
d2 = r2.json()
m2 = d2["message"]
print("=== Test 2: no think param ===")
print("content:", repr(m2.get("content", "")[:300]))
print("thinking:", repr(m2.get("thinking", "")[:200]))
print("eval_count:", d2.get("eval_count"))
