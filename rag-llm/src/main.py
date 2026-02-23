from filter_extract import extract_filters  # Fast keyword-based extraction (replaces LLM call)
from retrieval import retrieve
from answer_generate import generate_answer
import sys
import time

DEBUG = True
STREAM_RESPONSES = True  # Set to False for non-streaming responses

def handle_query(query: str) -> str:
    total_start = time.time()
    # Fast keyword-based filter extraction (no LLM call!)
    filter_start = time.time()
    parsed = extract_filters(query)
    filter_time = time.time() - filter_start

    if DEBUG:
        print("\n[DEBUG] Filter Extract Output (fast):")
        print(parsed)
        print(f"[DEBUG] Filter extraction time: {filter_time*1000:.2f}ms")


    intent = parsed["intent"]
    company = parsed["company"]

    filters = {}

    if intent == "company_info":
        filters["knowledge_type"] = "company_facts"
        if company:
            filters["company"] = company

    elif intent == "policy_info":
        filters["knowledge_type"] = "policy"

    elif intent == "placement_statistics":
        filters["knowledge_type"] = "statistics"

    # cgpa_coverage and general_placement → no filters

    retrieval_start = time.time()
    chunks = retrieve(
        query=query,
        filters=filters if filters else None
    )
    retrieval_time = time.time() - retrieval_start
    
    if DEBUG:
        print(f"\n[DEBUG] Retrieved {len(chunks)} chunks")
        print(f"[DEBUG] Retrieval time: {retrieval_time*1000:.2f}ms")


    if not chunks:
        return "I could not find relevant information in the available data."

    context = "\n\n".join(c["text"] for c in chunks)

    answer_start = time.time()
    
    if STREAM_RESPONSES:
        # Streaming mode
        if DEBUG:
            print("\n[DEBUG] Answer Generator Output (streaming):")
        
        answer_parts = []
        for chunk in generate_answer(context, query, stream=True):
            print(chunk, end='', flush=True)
            answer_parts.append(chunk)
        
        answer = ''.join(answer_parts)
        print()  # New line after streaming
    else:
        # Non-streaming mode
        answer = generate_answer(context, query, stream=False)
        
        if DEBUG:
            print("\n[DEBUG] Answer Generator Output:")
            print(answer)
    
    answer_time = time.time() - answer_start
    total_time = time.time() - total_start
    
    if DEBUG:
        print(f"\n[DEBUG] Answer generation time: {answer_time:.2f}s")
        print(f"[DEBUG] TOTAL TIME: {total_time:.2f}s")
        print(f"[DEBUG] Breakdown: Filter={filter_time*1000:.0f}ms | Retrieval={retrieval_time*1000:.0f}ms | Answer={answer_time:.1f}s")
    
    return answer




if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (or type 'exit'): ")
        if q.lower() == "exit":
            break
        print("\nANSWER:\n")
        print(handle_query(q))
