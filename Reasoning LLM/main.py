from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json

from prompts import (
    INFERENCE_TEMPLATE,
    REASONING_TEMPLATE,
    REASONING_SYSTEM_PROMPT
)

OLLAMA_BASE_URL = "http://localhost:11434"

llm = Ollama(
    model="gpt-oss:20b-cloud",
    base_url=OLLAMA_BASE_URL,
    temperature=0.3
)

# Inference prompt
inference_prompt = PromptTemplate(
    input_variables=["question"],
    template=INFERENCE_TEMPLATE
)

inference_chain = LLMChain(
    llm=llm,
    prompt=inference_prompt
)

reasoning_prompt = PromptTemplate(
    input_variables=[
        "question",
        "year",
        "cgpa_band",
        "skill_level",
        "system_prompt"
    ],
    template=REASONING_TEMPLATE
)

reasoning_chain = LLMChain(
    llm=llm,
    prompt=reasoning_prompt
)
def run_reasoning_llm(free_text):
    # Step 1: Infer profile
    inference_output = inference_chain.run(question=free_text)

    profile = json.loads(inference_output)

    # Step 2: Reason using inferred profile
    return reasoning_chain.run(
        system_prompt=REASONING_SYSTEM_PROMPT,
        question=free_text,
        year=profile["year"],
        cgpa_band=profile["cgpa_band"],
        skill_level=profile["skill_level"]
    )
    
if __name__ == "__main__":
    
    text=input("enter the text ")
    response = run_reasoning_llm(
        text
    )
    print("ANSWER\n")
    print(response)
