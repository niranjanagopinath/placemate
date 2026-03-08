from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json
import time
import asyncio
import os
import csv
from filter_extract import extract_filters
from retrieval import retrieve_async
from answer_generate import generate_answer
from reasoning_logic import extract_profile, generate_reasoning

app = FastAPI()

class ContributionData(BaseModel):
    # Student Profile
    year: int
    branch: str
    cgpa: float
    college_tier: int
    skills: str
    num_projects: int
    num_internships: int
    
    # Placement Details
    company: str
    role: str
    package_lpa: float
    placement_type: str
    interview_rounds: int
    interview_focus: str
    
    # Preparation Strategy
    dsa_level: str
    platforms_used: str
    questions_solved: int
    prep_duration_months: int
    
    # Advice
    advice: str

# Get absolute path to the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/contribute")
async def read_contribute():
    return FileResponse(os.path.join(STATIC_DIR, "contribute.html"))

async def stream_reasoning_response(query: str):
    total_start = time.time()
    
    # 1. Profile Inference (Fast step to understand the user)
    inference_start = time.time()
    profile_json = extract_profile(query)
    inference_time = (time.time() - inference_start) * 1000
    
    # Send metadata
    yield f"data: {json.dumps({'type': 'start', 'metrics': {'filter': f'{inference_time:.0f}ms', 'retrieval': 'N/A'}})}\n\n"

    # 2. Reasoning Generation (Streaming)
    reason_start = time.time()
    for chunk in generate_reasoning(query, profile_json, stream=True):
        yield f"data: {json.dumps({'type': 'content', 'chunk': chunk})}\n\n"
        await asyncio.sleep(0.01)

    reason_time = time.time() - reason_start
    total_time = time.time() - total_start
    
    # Final metrics
    yield f"data: {json.dumps({'type': 'end', 'metrics': {'answer': f'{reason_time:.1f}s', 'total': f'{total_time:.1f}s'}})}\n\n"

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
    chunks = await retrieve_async(query=query, filters=filters if filters else None)
    retrieval_time = (time.time() - retrieval_start) * 1000
    
    if not chunks:
        yield f"data: {json.dumps({'error': 'No relevant information found'})}\n\n"
        return

    context = "\n\n".join(c["text"] for c in chunks)
    
    # Send initial metadata/metrics
    yield f"data: {json.dumps({'type': 'start', 'metrics': {'filter': f'{filter_time:.0f}ms', 'retrieval': f'{retrieval_time:.0f}ms'}})}\n\n"

    # 3. Answer Generation (Streaming)
    answer_start = time.time()
    async for chunk in await generate_answer(context, query, stream=True):
        yield f"data: {json.dumps({'type': 'content', 'chunk': chunk})}\n\n"
        await asyncio.sleep(0.01) # Small sleep to help yield to event loop

    answer_time = time.time() - answer_start
    total_time = time.time() - total_start
    
    # Final metrics
    yield f"data: {json.dumps({'type': 'end', 'metrics': {'answer': f'{answer_time:.1f}s', 'total': f'{total_time:.1f}s'}})}\n\n"

@app.get("/query")
async def query_llm(q: str, mode: str = "rag"):
    if mode == "reasoning":
        return StreamingResponse(stream_reasoning_response(q), media_type="text/event-stream")
    return StreamingResponse(stream_rag_response(q), media_type="text/event-stream")

@app.post("/api/contribute")
async def add_contribution(data: ContributionData):
    try:
        # Determine paths
        # Assuming app.py is in rag-llm/src
        src_dir = os.path.dirname(os.path.abspath(__file__))
        rag_llm_dir = os.path.dirname(src_dir)
        placemate_dir = os.path.dirname(rag_llm_dir)
        dataset_dir = os.path.join(placemate_dir, "dataset", "structured")
        
        os.makedirs(dataset_dir, exist_ok=True)
        
        file_path = os.path.join(dataset_dir, "student_experiences.csv")
        file_exists = os.path.isfile(file_path)
        
        # Determine the next contribution id
        contribution_id = "EXP001"
        if file_exists:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
                if len(rows) > 1:
                    last_row = rows[-1]
                    if last_row and last_row[0].startswith("EXP"):
                        last_id = last_row[0]
                        if last_id[3:].isdigit():
                            next_num = int(last_id[3:]) + 1
                            contribution_id = f"EXP{next_num:03d}"
        
        # Append data to the CSV file
        with open(file_path, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            headers = [
                "contribution_id", "year", "branch", "cgpa", "college_tier", "skills",
                "num_projects", "num_internships", "company", "role", "package_lpa",
                "placement_type", "interview_rounds", "interview_focus", "dsa_level",
                "platforms_used", "questions_solved", "prep_duration_months", "advice"
            ]
            
            if not file_exists:
                writer.writerow(headers)
            
            writer.writerow([
                contribution_id,
                data.year, data.branch, data.cgpa, data.college_tier, data.skills,
                data.num_projects, data.num_internships, data.company, data.role,
                data.package_lpa, data.placement_type, data.interview_rounds,
                data.interview_focus, data.dsa_level, data.platforms_used,
                data.questions_solved, data.prep_duration_months, data.advice
            ])
            
        return {"message": f"Successfully added placement experience for {data.company}.", "contribution_id": contribution_id}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9000)
