from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import json
import time
import asyncio
from filter_extract import extract_filters
from retrieval import retrieve
from answer_generate import generate_answer

import os

app = FastAPI()

# Get absolute path to the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

async def stream_rag_response(query: str):
    total_start = time.time()
    
    # 1. Filter Extraction
    filter_start = time.time()
    parsed = extract_filters(query)
    filter_time = (time.time() - filter_start) * 1000
    
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

    # 2. Retrieval
    retrieval_start = time.time()
    chunks = retrieve(query=query, filters=filters if filters else None)
    retrieval_time = (time.time() - retrieval_start) * 1000
    
    if not chunks:
        yield f"data: {json.dumps({'error': 'No relevant information found'})}\n\n"
        return

    context = "\n\n".join(c["text"] for c in chunks)
    
    # Send initial metadata/metrics
    yield f"data: {json.dumps({'type': 'start', 'metrics': {'filter': f'{filter_time:.0f}ms', 'retrieval': f'{retrieval_time:.0f}ms'}})}\n\n"

    # 3. Answer Generation (Streaming)
    answer_start = time.time()
    for chunk in generate_answer(context, query, stream=True):
        yield f"data: {json.dumps({'type': 'content', 'chunk': chunk})}\n\n"
        await asyncio.sleep(0.01) # Small sleep to help yield to event loop

    answer_time = time.time() - answer_start
    total_time = time.time() - total_start
    
    # Final metrics
    yield f"data: {json.dumps({'type': 'end', 'metrics': {'answer': f'{answer_time:.1f}s', 'total': f'{total_time:.1f}s'}})}\n\n"

@app.get("/query")
async def query_rag(q: str):
    return StreamingResponse(stream_rag_response(q), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
