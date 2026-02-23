import requests
import json

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gpt-oss:20b-cloud"

SYSTEM_PROMPT = """
You are a placement information assistant.

Rules:
- Answer using ONLY the provided context.
- Look both at the question and context
- State facts and comparisons; do NOT answer using yes/no decisions.
- You may compare a given CGPA value against stated minimum CGPA requirements.
- Do NOT predict placement outcomes.
- Do NOT assess or guarantee eligibility.
- If information is missing, say so clearly.
- Be neutral, factual, and concise.


"""

def generate_answer(analysis_output, question: str, stream: bool = False):
    """
    Generate answer from context.
    
    Args:
        analysis_output: Context to use for answering
        question: User's question
        stream: If True, yields answer chunks as they arrive. If False, returns complete answer.
    
    Returns:
        If stream=False: Complete answer string
        If stream=True: Generator yielding answer chunks
    """
    prompt = f"""
Context:
{analysis_output}

Question:
{question}

Answer the question using ONLY the context above.
"""

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "stream": stream,
        "temperature": 0.2
    }

    if stream:
        return _generate_streaming(payload)
    else:
        return _generate_complete(payload)


def _generate_complete(payload):
    """Generate complete answer without streaming."""
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


def _generate_streaming(payload):
    """Generate answer with streaming."""
    response = requests.post(OLLAMA_URL, json=payload, stream=True)
    response.raise_for_status()
    
    for line in response.iter_lines():
        if line:
            try:
                chunk = json.loads(line)
                if "message" in chunk and "content" in chunk["message"]:
                    content = chunk["message"]["content"]
                    if content:
                        yield content
            except json.JSONDecodeError:
                continue

