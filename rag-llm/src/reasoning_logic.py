import json
import sys
import os

from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ─── Prompts ────────────────────────────────────────────────────────────────

INFERENCE_TEMPLATE = """
You are an assistant that extracts student profile information.

From the text below, infer:
- year (e.g. 1st year, 2nd year, 3rd year, final year, unknown)
- cgpa_band (high / medium / low / unknown)
- skill_level (beginner / medium / strong / unknown)

If something is not mentioned, set it to "unknown".

Return ONLY valid JSON, with no extra text, like this:
{{"year": "...", "cgpa_band": "...", "skill_level": "..."}}

Text:
{question}
"""

REASONING_SYSTEM_PROMPT = """
You are a placement guidance assistant focused on reasoning and strategy.

You do NOT provide:
- Exact CGPA cutoffs
- Company-specific eligibility rules
- Guaranteed placement outcomes

You MUST:
- Reason through trade-offs
- Adapt advice to year of study
- Consider time, risk, and uncertainty
- Avoid absolute or deterministic language
- Clearly say when information is insufficient

If a question requires factual placement data, say so instead of guessing.
"""

REASONING_TEMPLATE = """
{system_prompt}

Student profile:
- Year of study: {year}
- CGPA level: {cgpa_band}
- Skill level: {skill_level}

Student question:
{question}

Respond strictly in this format:

1. Understanding the situation
2. Key constraints
3. Reasoned assessment
4. What to do next
5. Optional clarification (only if needed)
"""

# ─── LLM Setup ──────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434"

_llm = Ollama(model="gpt-oss:20b-cloud", base_url=OLLAMA_BASE_URL, temperature=0.3)

_inference_chain = LLMChain(
    llm=_llm,
    prompt=PromptTemplate(
        input_variables=["question"],
        template=INFERENCE_TEMPLATE,
    ),
)

_reasoning_chain = LLMChain(
    llm=_llm,
    prompt=PromptTemplate(
        input_variables=["question", "year", "cgpa_band", "skill_level", "system_prompt"],
        template=REASONING_TEMPLATE,
    ),
)

# ─── Public API ─────────────────────────────────────────────────────────────

def extract_profile(query: str) -> dict:
    """Run the inference chain and return the parsed profile dict."""
    raw = _inference_chain.run(question=query)
    # Strip markdown code fences if the model wraps the JSON
    raw = raw.strip().strip("```json").strip("```").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: return an unknown profile so the app doesn't crash
        return {"year": "unknown", "cgpa_band": "unknown", "skill_level": "unknown"}


def generate_reasoning(query: str, profile: dict, stream: bool = False):
    """
    Generate a reasoning response.

    If stream=True, yields text chunks so FastAPI can SSE-stream them.
    If stream=False, returns the full string.
    """
    year       = profile.get("year", "unknown")
    cgpa_band  = profile.get("cgpa_band", "unknown")
    skill_level = profile.get("skill_level", "unknown")

    if stream:
        # Ollama supports streaming via the underlying client; we use invoke
        # and chunk the output word-by-word so the frontend gets a live feel.
        full_response = _reasoning_chain.run(
            system_prompt=REASONING_SYSTEM_PROMPT,
            question=query,
            year=year,
            cgpa_band=cgpa_band,
            skill_level=skill_level,
        )
        # Yield word-by-word for streaming effect
        words = full_response.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
    else:
        return _reasoning_chain.run(
            system_prompt=REASONING_SYSTEM_PROMPT,
            question=query,
            year=year,
            cgpa_band=cgpa_band,
            skill_level=skill_level,
        )
