# Knowledge Contract  
_Placement RAG System_

---

## 1. Knowledge Source Inventory

This document defines the role, scope, and authority of each dataset used by the Placement RAG system.  
Each dataset has a **strict responsibility boundary**. The system must not mix responsibilities across datasets.

---

## 2. Dataset Classification

---

### 1. `placement_policy_2026.txt`

**Purpose**  
Defines global placement rules, eligibility conditions, approval processes, and terminology that apply across all companies and batches unless explicitly overridden by CIR.

**Knowledge Type**  
`policy`

**Scope**  
`global`

**Authority Level**  
`highest`

**What this dataset CAN answer**
- Minimum CGPA required to be eligible for placements
- Whether arrears are allowed in general
- Attendance requirements for placement eligibility
- Definitions of recruiter categories (IT Services, Core, Dream, Marquee, Unclassified)
- CIR and department approval conditions
- General eligibility constraints applicable to all students

**What this dataset MUST NOT answer**
- Company-specific CGPA cutoffs
- Role-specific eligibility criteria
- Company-wise CTC or salary details
- Selection rounds of individual companies
- Batch-specific placement outcomes

---

### 2. `companies_2024.csv` / `companies_2025.csv`

**Purpose**  
Provides factual, company- and role-specific placement information for a given batch year.

**Knowledge Type**  
`company_facts`

**Scope**  
`company + batch`

**Authority Level**  
`conditional`

**What this dataset CAN answer**
- Company-wise minimum CGPA requirements
- Allowed backlogs per role
- Eligible branches for a company and role
- Offered CTC and base salary
- Selection rounds
- Job location
- Batch-specific company participation

**What this dataset MUST NOT answer**
- Global placement rules (attendance %, CIR approval)
- Definitions of recruiter categories
- Aggregate placement statistics
- Overall placement success rates

---

### 3. `placement_statistics_2024.txt`

**Purpose**  
Provides aggregate, descriptive statistics summarizing placement outcomes for a batch.

**Knowledge Type**  
`statistics`

**Scope**  
`batch_aggregate`

**Authority Level**  
`informational`

**What this dataset CAN answer**
- Total number of students eligible
- Number of students placed
- Highest CTC offered in the batch
- Average CTC of the batch
- Percentage distribution of recruiter types (service, product, startup)

**What this dataset MUST NOT answer**
- Company-specific eligibility
- CGPA cutoffs
- Role or branch eligibility
- Selection processes
- Individual student outcomes

---

## 3. Conflict Resolution Rules

If multiple datasets appear relevant to a query:

1. **Policy data overrides company data**
2. **Company data overrides statistics**
3. **Statistics must never be used for eligibility or rule enforcement**

---

## 4. System Enforcement Rule

If a user query requires information that belongs to a dataset **outside the permitted scope**, the system must:
- Reject the query for that dataset
- Respond with **“Information not available in the current data”**

---

## 5. Design Principle

The Placement RAG system is a **fact-bounded retrieval system**, not an advisory or predictive agent.  
If a fact is not explicitly present in the allowed dataset, the system must state that it does not know.

---

**End of Document**
