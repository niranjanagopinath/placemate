import csv
import json
import os
import re

DATASET_DIR = "dataset/structured"
OUTPUT_DIR = "rag-llm/chunks"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_batch_year(filename: str) -> int:
    match = re.search(r"(20\d{2})", filename)
    if not match:
        raise ValueError(f"Batch year not found in filename: {filename}")
    return int(match.group(1))

def generate_chunks(csv_path: str):
    filename = os.path.basename(csv_path)
    batch_year = extract_batch_year(filename)
    output_file = os.path.join(
        OUTPUT_DIR, f"company_facts_{batch_year}.json"
    )

    chunks = []

    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            chunk_text = f"""
Company: {row['company_name']}
Role: {row['role']}
Batch Year: {row['batch_year']}
Company Type: {row['company_type']}

Eligibility Criteria:
- Minimum CGPA: {row['min_cgpa']}
- Allowed Backlogs: {row['allowed_backlogs']}
- Eligible Branches: {row['allowed_branches']}

Compensation:
- CTC: {row['ctc_lpa']} LPA
- Base Salary: {row['base_lpa']} LPA

Selection Rounds:
- {row['selection_rounds'].replace(',', ', ')}

Job Location: {row['location']}
""".strip()

            metadata = {
                "knowledge_type": "company_facts",
                "company_id": row["company_id"],
                "company": row["company_name"],
                "role": row["role"],
                "batch_year": int(row["batch_year"]),
                "topic": "eligibility_and_package",
                "authority": "conditional",
                "source_file": csv_path
            }

            chunks.append({
                "text": chunk_text,
                "metadata": metadata
            })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"[OK] {len(chunks)} chunks generated → {output_file}")

def main():
    for file in os.listdir(DATASET_DIR):
        if file.startswith("companies_") and file.endswith(".csv"):
            csv_path = os.path.join(DATASET_DIR, file)
            generate_chunks(csv_path)

if __name__ == "__main__":
    main()
