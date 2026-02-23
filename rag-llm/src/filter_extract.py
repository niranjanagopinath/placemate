"""
Fast keyword-based filter extraction to replace LLM-based intent extraction.
This eliminates the 2-5 second LLM call while maintaining high accuracy.
"""

# Company names extracted from the data
COMPANIES = [
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "ibm", "hcl", "tech mahindra", "amazon", "google", "microsoft",
    "adobe", "intuit", "flipkart", "meesho", "zomato", "swiggy",
    "zoho", "freshworks", "oracle", "paytm", "razorpay", "phonepe",
    "byju", "byjus", "unacademy", "deloitte", "kpmg"
]

# Keywords for different query types
POLICY_KEYWORDS = [
    "policy", "rule", "eligibility", "attendance", "arrear", "arrears",
    "backlog", "backlogs", "regulation", "requirement", "allowed",
    "mandatory", "compulsory", "skip", "absence", "register"
]

STATISTICS_KEYWORDS = [
    "highest", "lowest", "average", "percentage", "statistics",
    "summary", "total", "count", "how many", "placed", "placement rate"
]

CGPA_COVERAGE_KEYWORDS = [
    "enough", "safe", "sufficient", "can i get", "will i get",
    "chances", "eligible for", "qualify"
]

COMPANY_INFO_KEYWORDS = [
    "cutoff", "minimum cgpa", "requirement", "package", "ctc",
    "salary", "role", "roles", "position", "offer", "interview",
    "selection", "round", "rounds", "location"
]


def extract_filters(query: str) -> dict:
    """
    Fast keyword-based filter extraction.
    
    Returns same format as intent_extract for compatibility:
    {
        "intent": str,
        "company": str or None
    }
    
    Then converted to filters dict for retrieval.
    """
    query_lower = query.lower()
    
    # Extract company name
    company = None
    for comp in COMPANIES:
        if comp in query_lower:
            # Normalize company name
            company = comp.upper()
            if company == "TECH MAHINDRA":
                company = "Tech Mahindra"
            elif company in ["BYJU", "BYJUS"]:
                company = "Byju's"
            break
    
    # Determine intent based on keywords
    intent = "general_placement"  # default
    
    # Check for CGPA coverage queries (specific CGPA value + coverage keywords)
    has_cgpa_value = any(word in query_lower for word in ["cgpa", "gpa", "7", "8", "9", "6"])
    has_coverage_keyword = any(keyword in query_lower for keyword in CGPA_COVERAGE_KEYWORDS)
    
    if has_cgpa_value and has_coverage_keyword:
        intent = "cgpa_coverage"
    
    # Check for policy queries
    elif any(keyword in query_lower for keyword in POLICY_KEYWORDS):
        intent = "policy_info"
    
    # Check for statistics queries
    elif any(keyword in query_lower for keyword in STATISTICS_KEYWORDS):
        intent = "placement_statistics"
    
    # Check for company-specific queries
    elif company or any(keyword in query_lower for keyword in COMPANY_INFO_KEYWORDS):
        intent = "company_info"
    
    return {
        "intent": intent,
        "company": company
    }


def get_retrieval_filters(query: str) -> dict:
    """
    Extract filters for retrieval based on query.
    Returns filters dict ready for retrieve() function.
    """
    parsed = extract_filters(query)
    
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
    
    return filters if filters else None
