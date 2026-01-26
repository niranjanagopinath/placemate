from retrieval import retrieve

# results = retrieve(
#     query="What is the minimum CGPA required?",
#     filters={"knowledge_type": "policy"}
# )

# for r in results:
#     print(r["metadata"]["topic"])
#     print(r["text"])
#     print("-" * 40)


results = retrieve(
    query="What is the CGPA requirement for TCS Digital Engineer?",
    filters={
        "knowledge_type": "company_facts",
        "batch_year": 2024,
        "company": "TCS"
    }
)
for r in results:
    print(r["metadata"]["topic"])
    print(r["text"])
    print("-" * 40)