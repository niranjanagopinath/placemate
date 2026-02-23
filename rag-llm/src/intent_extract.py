import requests
import json
import re

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2:3b"

SYSTEM_PROMPT = """
You are an intent classifier for a placement information system.

Your job is ONLY to classify the user's question and extract the company name if mentioned.

Possible intents:
- company_info
  (questions about CGPA cutoff, roles, packages, interview rounds of a company)

- policy_info
  (questions about placement rules, eligibility rules, arrears, attendance, offers)

- placement_statistics
  (questions about numbers, percentages, highest package, placement summary)

- cgpa_coverage
  (questions asking whether a specific CGPA value is "enough", "safe", or "sufficient")

- general_placement
  (general or unclear placement-related questions)

Rules:
- If the question asks for "cutoff", "minimum CGPA", or "requirement", use company_info.
- If the question asks whether a specific CGPA value is "enough", "safe", or "sufficient",
  use cgpa_coverage.
- Do NOT predict outcomes.
- Do NOT judge eligibility.
- Do NOT add extra fields.
- If no company is mentioned, set company to null.

Additional rules:
- If the question asks about placement policy, rules, or regulations related to CGPA,
  classify it as policy_info, even if CGPA is mentioned.
- Use cgpa_coverage ONLY when the question asks whether a specific CGPA value
  (for example: "8 CGPA") is enough, safe, or sufficient.


Return ONLY valid JSON in this exact format:

{
  "intent": "<one of the intents above>",
  "company": "<company name or null>"
}

"""

def extract_intent(query: str) -> dict:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ],
        "stream": False,
        "temperature": 0
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()

    content = response.json()["message"]["content"]

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise ValueError(f"Invalid intent output:\n{content}")

    return json.loads(match.group(0))
