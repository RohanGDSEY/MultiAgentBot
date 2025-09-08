import os, json, math,re
from typing import List, Dict, Any, Optional
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from datetime import datetime
from src.risk_assessment import calculate_comprehensive_risk_score, parse_medical_information, make_underwriting_decision
# from streamlit_app import


EMB_MODEL = r"C:\TextSummarization\TS\all-MiniLM-L6-v2"
DB_DIR = "storage/chroma"

def load_json(path: str) -> Dict[str, Any]:
    """Load JSON with error handling"""
    try:
        with open(path, "r", encoding="utf-8") as f: 
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {path} not found, returning empty dict")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from {path}: {e}")
        return {}

# --- Lookup / CRM / Outbox (dummy stores) ---
def lookup_person(query: str) -> Dict[str, Any]:
    """Enhanced person lookup with better matching"""
    data = load_json("data/persons.json").get("people", [])
    if not data:
        return {"error": "No person data found"}
    
    # Search by name, org, email, or phone
    search_fields = []
    for p in data:
        search_text = f"{p.get('name', '')} {p.get('org', '')} {p.get('email', '')}"
        search_fields.append(search_text)
    
    best = process.extractOne(query, search_fields, scorer=fuzz.WRatio)
    if best and best[1] > 60:  # Confidence threshold
        idx = search_fields.index(best[0])
        person = data[idx].copy()
        person["match_confidence"] = best[1]
        return person
    
    return {"error": f"No good match found for '{query}'"}

def create_crm_contact(person: Dict[str, Any]) -> Dict[str, Any]:
    """Create CRM contact with timestamp and validation"""
    if not person or "name" not in person:
        return {"error": "Invalid person data - name is required"}
    
    os.makedirs("storage", exist_ok=True)
    path = "storage/crm.json"
    
    # Handle corrupted or empty JSON file
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    crm = json.loads(content)
                else:
                    crm = {"contacts": []}
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"Warning: Corrupted {path}, creating new file")
            crm = {"contacts": []}
    else:
        crm = {"contacts": []}
    
    # Ensure contacts key exists
    if "contacts" not in crm:
        crm["contacts"] = []
    
    # Add metadata
    contact = person.copy()
    contact["created_at"] = datetime.now().isoformat()
    contact["source"] = "agent_lookup"
    
    crm["contacts"].append(contact)
    
    try:
        with open(path, "w", encoding="utf-8") as f: 
            json.dump(crm, f, indent=2)
        return {"status": "success", "contact_id": len(crm["contacts"]), "contact": contact}
    except Exception as e:
        return {"error": f"Failed to save contact: {e}"}

def send_email(to: str, subject: str, body: str, attachments: List[str] = None):
    """Enhanced email sending with validation"""
    if not to or not subject or not body:
        return {"error": "Missing required email fields"}
    
    os.makedirs("storage", exist_ok=True)
    path = "storage/outbox.json"
    box = load_json(path) if os.path.exists(path) else {"emails": []}
    
    email = {
        "to": to,
        "subject": subject,
        "body": body,
        "attachments": attachments or [],
        "sent_at": datetime.now().isoformat(),
        "status": "sent"
    }
    
    box["emails"].append(email)
    
    try:
        with open(path, "w", encoding="utf-8") as f: 
            json.dump(box, f, indent=2)
        return {"status": "success", "email_id": len(box["emails"])}
    except Exception as e:
        return {"error": f"Failed to send email: {e}"}

# --- Vector search over brochures ---
_embed = None

def brochure_search(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """Enhanced brochure search with better error handling"""
    global _embed
    
    try:
        if _embed is None:
            print("Loading embedding model...")
            _embed = SentenceTransformer(EMB_MODEL)
        
        if not os.path.exists(DB_DIR):
            return {"error": "ChromaDB not initialized. Run embeddings_index.py first"}
        
        client = chromadb.PersistentClient(path=DB_DIR, settings=Settings(allow_reset=False))
        col = client.get_or_create_collection("brochures")
        
        # Check if collection has data
        count = col.count()
        if count == 0:
            return {"error": "No brochures indexed. Run embeddings_index.py first"}
        
        qv = _embed.encode([query], normalize_embeddings=True).tolist()[0]
        res = col.query(query_embeddings=[qv], n_results=min(k, count))
        
        out = []
        if res["metadatas"] and res["documents"]:
            for mid, mdoc, dist in zip(res["metadatas"][0], res["documents"][0], res["distances"][0]):
                out.append({
                    "product_id": mid["product_id"], 
                    "name": mid["name"], 
                    "company": mid["company"], 
                    "snippet": mdoc[:600],
                    "relevance_score": round(1 / (1 + dist), 3)  # Convert distance to similarity
                })
        
        return out
        
    except Exception as e:
        return {"error": f"Brochure search failed: {e}"}

# --- Product & quoting ---
def get_products_for_interest(interest: str) -> List[Dict[str, Any]]:
    """Get products by category with enhanced filtering"""
    try:
        products_data = load_json("data/products.json")
        products = products_data.get("products", [])
        
        if not products:
            return {"error": "No products found in database"}
        
        matching = [p for p in products if p.get("category", "").lower() == interest.lower()]
        
        if not matching:
            # Try partial matching
            all_categories = list(set(p.get("category", "") for p in products))
            suggestion = process.extractOne(interest, all_categories, scorer=fuzz.WRatio)
            return {
                "error": f"No products found for '{interest}'",
                "available_categories": all_categories,
                "suggestion": suggestion[0] if suggestion and suggestion[1] > 60 else None
            }
        
        # Sort by base rate for consistent ordering
        matching.sort(key=lambda x: x.get("base_rate", 0))
        return matching
        
    except Exception as e:
        return {"error": f"Product search failed: {e}"}

def _pick_band(bands: List[int], si: int) -> int:
    """Pick closest sum insured band"""
    return min(bands, key=lambda b: abs(b - si))

def estimate_premium(person: Dict[str, Any], product: Dict[str, Any], 
                    sum_insured: int = 500000, riders: Optional[List[str]] = None) -> Dict[str, Any]:
    """Enhanced premium calculation with better validation"""
    try:
        # Validation
        if not person or not product:
            return {"error": "Missing person or product data"}
        
        if "age" not in person:
            return {"error": "Person age is required for premium calculation"}
        
        age = int(person["age"])
        industry = person.get("risk_factors", {}).get("industry", "retail")
        base_rate = float(product.get("base_rate", 0))
        
        if base_rate <= 0:
            return {"error": "Invalid base rate in product"}
        
        # Age factor calculation - FIXED
        age_factor = 1.0
        age_factors = product.get("age_factor", [])
        
        # Handle different age_factor formats
        if age_factors:
            if isinstance(age_factors[0], list):
                # Format: [[min_age, max_age, factor], ...]
                for age_range in age_factors:
                    if len(age_range) >= 3:
                        lo, hi, factor = age_range[0], age_range[1], age_range[2]
                        if lo <= age <= hi:
                            age_factor = factor
                            break
            else:
                # Format: [factor1, factor2, factor3, factor4] for age bands
                # Assume bands: 18-30, 31-40, 41-50, 51+
                if age <= 30:
                    age_factor = age_factors[0] if len(age_factors) > 0 else 1.0
                elif age <= 40:
                    age_factor = age_factors[1] if len(age_factors) > 1 else 1.0
                elif age <= 50:
                    age_factor = age_factors[2] if len(age_factors) > 2 else 1.0
                else:
                    age_factor = age_factors[3] if len(age_factors) > 3 else 1.0
        
        # Industry factor
        ind_factor = product.get("industry_factor", {}).get(industry, 1.0)
        
        # Sum insured factor
        bands = product.get("sum_insured_bands", [sum_insured])
        selected_band = _pick_band(bands, sum_insured)
        si_factor = 1.0 + (selected_band - bands[0]) / max(bands[-1], 1) * 0.3
        
        # Riders factor
        rider_factor = 1.0
        riders = riders or []
        for rider in product.get("riders", []):
            if isinstance(rider, dict) and rider.get("code") in riders:
                rider_factor *= float(rider.get("factor", 1.0))
            elif isinstance(rider, str) and rider in riders:
                rider_factor *= 1.05  # Default 5% increase for each rider
        
        # Calculate final premium
        premium = base_rate * age_factor * ind_factor * si_factor * rider_factor
        
        return {
            "product_id": product["id"],
            "product_name": product["name"],
            "company": product["company"],
            "sum_insured": selected_band,
            "riders": riders,
            "estimated_annual_premium": round(premium, 2),
            "factors": {
                "age_factor": age_factor,
                "industry_factor": ind_factor,
                "si_factor": round(si_factor, 3),
                "rider_factor": round(rider_factor, 3)
            }
        }
        
    except Exception as e:
        return {"error": f"Premium calculation failed: {str(e)}"}

def required_docs_for(category: str) -> Dict[str, Any]:
    """Get required documents with validation"""
    try:
        std = load_json("data/standard_docs.json")
        if category not in std:
            available = list(std.keys())
            return {
                "error": f"Category '{category}' not found",
                "available_categories": available
            }
        return std[category]
    except Exception as e:
        return {"error": f"Failed to get required docs: {e}"}

# Additional utility functions
def get_person_risk_profile(person: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate risk profile for underwriting"""
    risk_factors = person.get("risk_factors", {})
    age = person.get("age", 35)
    
    risk_score = 0
    factors = []
    
    # Age risk
    if age > 50:
        risk_score += 2
        factors.append("Age > 50")
    elif age > 35:
        risk_score += 1
        factors.append("Age > 35")
    
    # Smoking
    if risk_factors.get("smoker", False):
        risk_score += 3
        factors.append("Smoker")
    
    # Industry risk
    high_risk_industries = ["manufacturing", "construction", "mining"]
    if risk_factors.get("industry") in high_risk_industries:
        risk_score += 1
        factors.append(f"High-risk industry: {risk_factors.get('industry')}")
    
    # Claims history
    claims = risk_factors.get("claims_last_3y", 0)
    if claims > 2:
        risk_score += 2
        factors.append(f"Multiple claims: {claims}")
    elif claims > 0:
        risk_score += 1
        factors.append(f"Previous claims: {claims}")
    
    risk_level = "Low" if risk_score <= 2 else "Medium" if risk_score <= 4 else "High"
    
    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_factors": factors
    }

def validate_person_category_match(person: Dict[str, Any], requested_category: str) -> Dict[str, Any]:
    """
    Validate if person's interests match the requested insurance category
    Returns validation result with recommendations
    """
    if not person or "error" in person:
        return {"valid": False, "error": "Person not found"}
    
    person_interests = person.get("interests", [])
    if isinstance(person_interests, str):
        person_interests = [person_interests]
    
    # Convert to lowercase for comparison
    person_interests_lower = [interest.lower() for interest in person_interests]
    requested_lower = requested_category.lower()
    
    # Check direct match
    if requested_lower in person_interests_lower:
        return {"valid": True, "message": f"Perfect match! {person['name']} is interested in {requested_category}"}
    
    # Check partial matches
    category_synonyms = {
        "health": ["medical", "healthcare", "wellness"],
        "motor": ["auto", "car", "vehicle", "automobile"],
        "life": ["term", "whole life", "endowment"],
        "commercial": ["business", "corporate", "property"]
    }
    
    synonyms = category_synonyms.get(requested_lower, [])
    for interest in person_interests_lower:
        if any(syn in interest for syn in synonyms) or interest in synonyms:
            return {"valid": True, "message": f"Good match! {person['name']} has related interest in {interest}"}
    
    # No match found
    return {
        "valid": False,
        "person_interests": person_interests,
        "requested_category": requested_category,
        "suggestion": f"{person['name']} is interested in {', '.join(person_interests)} but you selected {requested_category}",
        "alternative_categories": [cat for cat in person_interests if cat.lower() in ["health", "motor", "life", "commercial"]]
    }

def analyze_application_comprehensively(application_text: str) -> Dict[str, Any]:
    """Comprehensive application analysis with structured data extraction"""
    
    # Extract structured medical information
    medical_info = parse_medical_information(application_text)
    
    # Extract basic applicant info using regex with defaults
    applicant_info = {}
    
    # Age extraction with multiple patterns and default
    age_match = re.search(r'Age[:\s]*(\d+)', application_text, re.IGNORECASE)
    if age_match:
        applicant_info['age'] = int(age_match.group(1))
    else:
        # Try alternative patterns for commercial applications
        director_age_match = re.search(r'Director[^\n]*(\d+)\s*years?', application_text, re.IGNORECASE)
        if director_age_match:
            applicant_info['age'] = int(director_age_match.group(1))
        else:
            applicant_info['age'] = 35  # Default age to prevent None comparison errors
    
    # Income extraction with default
    income_match = re.search(r'(?:Income|Turnover)[:\s]*Rs\.?\s*([\d,]+)', application_text, re.IGNORECASE)
    if income_match:
        income_str = income_match.group(1).replace(',', '')
        applicant_info['annual_income'] = int(income_str)
    else:
        applicant_info['annual_income'] = 500000  # Default income
    
    # Occupation with better extraction
    occ_match = re.search(r'(?:Occupation|Business Type)[:\s]*([^\n]+)', application_text, re.IGNORECASE)
    if occ_match:
        applicant_info['occupation'] = occ_match.group(1).strip()
    else:
        applicant_info['occupation'] = 'Not specified'
    
    # Ensure medical_info has required structure
    if not isinstance(medical_info, dict):
        medical_info = {
            'conditions': [],
            'medications': [], 
            'family_history': [],
            'bmi': None,
            'smoking_status': 'unknown',
            'height': None,
            'weight': None
        }
    
    # Calculate comprehensive risk score with error handling
    try:
        risk_assessment = calculate_comprehensive_risk_score(applicant_info, medical_info)
    except Exception as e:
        print(f"Risk calculation error: {e}")
        # Fallback risk assessment
        risk_assessment = {
            'risk_score': 3,
            'risk_level': 'Medium',
            'risk_factors': [f'Default assessment due to parsing error: {str(e)}']
        }
    
    return {
        'applicant_info': applicant_info,
        'medical_info': medical_info,
        'risk_assessment': risk_assessment,
        'raw_content': application_text
    }

def enhanced_compliance_check(application_text: str, category: str) -> Dict[str, Any]:

    """Enhanced compliance checking with medical test requirements"""
    
    # Get standard requirements
    req_docs_result = required_docs_for(category)
    if "error" in req_docs_result:
        return req_docs_result
    
    required_docs = req_docs_result.get("required", [])
    optional_docs = req_docs_result.get("optional", [])
    
    # Check document presence
    req_status, opt_status = improved_document_checker(application_text, required_docs, optional_docs)
    
    # Special handling for life insurance medical tests
    medical_tests_required = []
    medical_tests_status = {}
    
    if category.lower() == "life":
        # Check if sum assured > 50L requires medical tests
        sum_match = re.search(r'Sum Assured[:\s]*Rs\.?\s*([\d,]+)', application_text, re.IGNORECASE)
        if sum_match:
            sum_amount = int(sum_match.group(1).replace(',', ''))
            if sum_amount > 5000000:  # 50L
                medical_tests_required = ["CBC", "Lipid Profile", "ECG", "Stress Test"]
        
        # Check each medical test
        for test in medical_tests_required:
            test_present = False
            if test.lower() in application_text.lower():
                if "scheduled" in application_text.lower() or "pending" in application_text.lower():
                    medical_tests_status[test] = "scheduled"
                else:
                    medical_tests_status[test] = "completed"
                    test_present = True
            else:
                medical_tests_status[test] = "missing"
    
    missing_required = [doc for doc, status in req_status.items() if not status]
    missing_medical = [test for test, status in medical_tests_status.items() if status == "missing"]
    pending_medical = [test for test, status in medical_tests_status.items() if status == "scheduled"]
    
    return {
        "required_docs": req_status,
        "optional_docs": opt_status,
        "medical_tests": medical_tests_status,
        "missing_required": missing_required,
        "missing_medical": missing_medical,
        "pending_medical": pending_medical,
        "compliance_score": len([d for d in req_status.values() if d]) / len(req_status) if req_status else 1.0
    }

def improved_document_checker(submission_text: str, required_docs: list, optional_docs: list = None):
    """Enhanced document compliance checking with better matching logic"""
    import re
    # Document name mappings for better matching
    doc_mappings = {
        "proposal form": ["proposal", "application form", "form"],
        "kyc documents": ["kyc", "aadhaar", "pan", "identity", "address proof"],
        "medical report": ["medical", "health report", "doctor", "physician"],
        "previous policy": ["previous", "existing policy", "current policy"],
        "age proof": ["age", "birth certificate", "date of birth"],
        "passport size photographs": ["photograph", "photos", "passport size"],
        "income proof": ["income", "salary", "form 16", "itr", "financial"],
        "rc book": ["rc", "registration certificate", "vehicle registration"],
        "driving license": ["license", "dl", "driving"],
        "vehicle inspection": ["inspection", "survey", "vehicle check"],
        "ncb certificate": ["ncb", "no claim bonus", "claim free"],
        "fitness certificate": ["fitness", "vehicle fitness"],
        "certificate of incorporation": ["incorporation", "company registration", "roc"],
        "gst certificate": ["gst", "tax registration"],
        "financial statements": ["financial", "balance sheet", "p&l", "audit"],
        "business license": ["business license", "trade license"],
        "fire safety certificate": ["fire safety", "fire noc", "fire certificate"]
    }
    
    submission_lower = submission_text.lower()
    
    def check_document(doc_name):
        doc_lower = doc_name.lower()
        # Direct match
        if doc_lower in submission_lower:
            return True
        
        # Check mappings
        for standard_name, variations in doc_mappings.items():
            if any(keyword in doc_lower for keyword in [standard_name]):
                return any(variation in submission_lower for variation in variations)
        
        # Fallback: check individual words
        doc_words = doc_lower.split()
        return any(word in submission_lower for word in doc_words if len(word) > 3)
    
    required_status = {}
    for doc in required_docs:
        required_status[doc] = check_document(doc)
    
    optional_status = {}
    if optional_docs:
        for doc in optional_docs:
            optional_status[doc] = check_document(doc)
    
    return required_status, optional_status
def load_customer_policies() -> Dict[str, Any]:
    """Load customer policies from JSON"""
    try:
        with open("data/customer_policies.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"policies": []}

def search_customer_policies(customer_name: str = None, customer_email: str = None) -> List[Dict[str, Any]]:
    """
    Search for existing policies by customer name or email
    Returns list of active policies for the customer
    """
    policies_data = load_customer_policies()
    all_policies = policies_data.get("policies", [])
    
    customer_policies = []
    
    for policy in all_policies:
        # Match by name or email
        if customer_name and policy.get("customer_name", "").lower() == customer_name.lower():
            customer_policies.append(policy)
        elif customer_email and policy.get("customer_email", "").lower() == customer_email.lower():
            customer_policies.append(policy)
    
    if not customer_policies:
        return {"message": "No policies found for this customer", "policies": []}
    
    return {
        "message": f"Found {len(customer_policies)} active policies",
        "customer_name": customer_policies[0].get("customer_name"),
        "customer_email": customer_policies[0].get("customer_email"),
        "policies": customer_policies
    }

def get_policy_details(policy_number: str) -> Dict[str, Any]:
    """Get detailed information about a specific policy"""
    policies_data = load_customer_policies()
    all_policies = policies_data.get("policies", [])
    
    for policy in all_policies:
        if policy.get("policy_number") == policy_number:
            return {"status": "found", "policy": policy}
    
    return {"status": "not_found", "message": f"Policy {policy_number} not found"}

def create_policy_embeddings():
    """Create embeddings for policy documents for RAG"""
    global _embed
    
    if _embed is None:
        _embed = SentenceTransformer(EMB_MODEL)
    
    client = chromadb.PersistentClient(path=DB_DIR, settings=Settings(allow_reset=False))
    
    # Create a separate collection for customer policies
    col = client.get_or_create_collection("customer_policies")
    
    policies_data = load_customer_policies()
    policies = policies_data.get("policies", [])
    
    docs, ids, metas = [], [], []
    
    for policy in policies:
        # Create searchable text from policy details
        policy_text = f"""
        Policy Number: {policy.get('policy_number')}
        Customer: {policy.get('customer_name')}
        Type: {policy.get('policy_type')}
        Product: {policy.get('product_name')}
        Company: {policy.get('insurance_company')}
        Status: {policy.get('status')}
        Coverage Amount: {policy.get('sum_insured')}
        Premium: {policy.get('annual_premium')}
        Start Date: {policy.get('start_date')}
        End Date: {policy.get('end_date')}
        Benefits: {', '.join(policy.get('benefits', []))}
        Terms: {policy.get('terms_conditions', '')}
        Claim Process: {policy.get('claim_process', '')}
        Exclusions: {', '.join(policy.get('exclusions', []))}
        """
        
        docs.append(policy_text)
        ids.append(policy.get('policy_number'))
        metas.append({
            "policy_number": policy.get('policy_number'),
            "customer_name": policy.get('customer_name'),
            "customer_email": policy.get('customer_email'),
            "policy_type": policy.get('policy_type')
        })
    
    if ids:
        vecs = _embed.encode(docs, normalize_embeddings=True).tolist()
        col.upsert(ids=ids, documents=docs, embeddings=vecs, metadatas=metas)
        return {"status": "success", "policies_indexed": len(ids)}
    
    return {"status": "no_policies", "message": "No policies to index"}

def policy_rag_search(query: str, customer_email: str = None, k: int = 3) -> List[Dict[str, Any]]:
    """
    Enhanced RAG search for policy-related queries with better understanding of exclusions and negations
    """
    global _embed
    
    try:
        # Load policies data
        policies_data = load_customer_policies()
        all_policies = policies_data.get("policies", [])
        
        # Filter by customer if email provided
        if customer_email:
            customer_policies = [p for p in all_policies if p.get("customer_email", "").lower() == customer_email.lower()]
        else:
            customer_policies = all_policies
        
        if not customer_policies:
            return [{"message": "No policies found for this customer", "relevant_info": "Please check your email or contact support"}]
        
        query_lower = query.lower()
        
        # Enhanced query analysis with negation and exclusion detection
        is_exclusion_query = any(word in query_lower for word in [
            "not included", "not covered", "excluded", "exclusion", "what's not", 
            "doesn't cover", "not eligible", "limitation", "restriction"
        ])
        
        is_expiry_query = any(word in query_lower for word in [
            "expir", "renew", "end", "valid till", "when does", "validity"
        ])
        
        is_coverage_query = any(word in query_lower for word in [
            "coverage", "sum insured", "amount", "covered", "protection"
        ]) and not is_exclusion_query
        
        is_premium_query = any(word in query_lower for word in [
            "premium", "cost", "price", "payment", "how much"
        ])
        
        is_benefit_query = any(word in query_lower for word in [
            "benefit", "feature", "advantage", "what does", "include"
        ]) and not is_exclusion_query
        
        is_claim_query = any(word in query_lower for word in [
            "claim", "how to claim", "claim process", "file claim"
        ])
        
        # Handle exclusions/not covered queries
        if is_exclusion_query:
            results = []
            for policy in customer_policies:
                exclusions = policy.get("exclusions", [])
                # If no explicit exclusions in data, provide common exclusions by policy type
                if not exclusions:
                    policy_type = policy.get("policy_type", "").lower()
                    if policy_type == "health":
                        exclusions = [
                            "Pre-existing diseases (first 3 years)",
                            "Cosmetic surgery",
                            "Dental treatment (unless accident)",
                            "Self-inflicted injuries",
                            "War and nuclear risks",
                            "Experimental treatments"
                        ]
                    elif policy_type == "motor":
                        exclusions = [
                            "Driving without valid license",
                            "Drunk driving",
                            "Racing or speed trials",
                            "Normal wear and tear",
                            "Consequential losses",
                            "Depreciation"
                        ]
                    elif policy_type == "life":
                        exclusions = [
                            "Suicide within first year",
                            "Death due to war",
                            "Self-inflicted injuries",
                            "Death while under influence",
                            "Participation in hazardous activities"
                        ]
                    else:
                        exclusions = ["Standard policy exclusions apply"]
                
                exclusions_text = "; ".join(exclusions[:5])  # Limit to first 5
                results.append({
                    "policy_number": policy.get("policy_number"),
                    "customer_name": policy.get("customer_name"),
                    "policy_type": policy.get("policy_type"),
                    "product_name": policy.get("product_name"),
                    "exclusions": exclusions,
                    "relevant_info": f"Policy {policy.get('policy_number')} ({policy.get('policy_type').title()}) exclusions: {exclusions_text}",
                    "relevance_score": 1.0
                })
            return results[:k] if results else [{"message": "No exclusion information found"}]
        
        # Handle expiry/renewal queries
        elif is_expiry_query:
            results = []
            for policy in customer_policies:
                end_date = policy.get("end_date")
                next_renewal = policy.get("next_renewal", end_date)
                results.append({
                    "policy_number": policy.get("policy_number"),
                    "customer_name": policy.get("customer_name"),
                    "policy_type": policy.get("policy_type"),
                    "product_name": policy.get("product_name"),
                    "end_date": end_date,
                    "next_renewal": next_renewal,
                    "auto_renewal": policy.get("auto_renewal", False),
                    "relevant_info": f"Policy {policy.get('policy_number')} ({policy.get('policy_type').title()}) expires on {end_date}. Auto-renewal: {'Yes' if policy.get('auto_renewal') else 'No'}",
                    "relevance_score": 1.0
                })
            return results[:k] if results else [{"message": "No expiry information found"}]
        
        # Handle coverage amount queries
        elif is_coverage_query:
            results = []
            total_coverage = 0
            for policy in customer_policies:
                sum_insured = policy.get("sum_insured", 0)
                total_coverage += sum_insured
                results.append({
                    "policy_number": policy.get("policy_number"),
                    "customer_name": policy.get("customer_name"),
                    "policy_type": policy.get("policy_type"),
                    "product_name": policy.get("product_name"),
                    "sum_insured": sum_insured,
                    "relevant_info": f"Policy {policy.get('policy_number')} ({policy.get('policy_type').title()}): Coverage ₹{sum_insured:,}",
                    "relevance_score": 1.0
                })
            
            # Add total coverage summary
            if len(results) > 1:
                results.insert(0, {
                    "summary": "total_coverage",
                    "customer_name": customer_policies[0].get("customer_name"),
                    "total_policies": len(customer_policies),
                    "total_coverage": total_coverage,
                    "relevant_info": f"Total coverage across all {len(customer_policies)} policies: ₹{total_coverage:,}",
                    "relevance_score": 1.0
                })
            
            return results[:k+1] if results else [{"message": "No coverage information found"}]
        
        # Handle premium queries
        elif is_premium_query:
            results = []
            total_premium = 0
            for policy in customer_policies:
                annual_premium = policy.get("annual_premium", 0)
                total_premium += annual_premium
                results.append({
                    "policy_number": policy.get("policy_number"),
                    "customer_name": policy.get("customer_name"),
                    "policy_type": policy.get("policy_type"),
                    "product_name": policy.get("product_name"),
                    "annual_premium": annual_premium,
                    "next_premium_due": policy.get("next_premium_due", policy.get("end_date")),
                    "relevant_info": f"Policy {policy.get('policy_number')} ({policy.get('policy_type').title()}): Premium ₹{annual_premium:,}/year",
                    "relevance_score": 1.0
                })
            
            # Add total premium summary
            if len(results) > 1:
                results.insert(0, {
                    "summary": "total_premium",
                    "customer_name": customer_policies[0].get("customer_name"),
                    "total_policies": len(customer_policies),
                    "total_premium": total_premium,
                    "relevant_info": f"Total annual premium across all {len(customer_policies)} policies: ₹{total_premium:,}",
                    "relevance_score": 1.0
                })
            
            return results[:k+1] if results else [{"message": "No premium information found"}]
        
        # Handle benefits queries  
        elif is_benefit_query:
            results = []
            for policy in customer_policies:
                benefits = policy.get("benefits", [])
                benefits_text = ", ".join(benefits[:3]) if benefits else "No benefits listed"
                all_benefits_text = "; ".join(benefits) if benefits else "No benefits available"
                results.append({
                    "policy_number": policy.get("policy_number"),
                    "customer_name": policy.get("customer_name"),
                    "policy_type": policy.get("policy_type"),
                    "product_name": policy.get("product_name"),
                    "benefits": benefits,
                    "riders": policy.get("riders", []),
                    "relevant_info": f"Policy {policy.get('policy_number')} ({policy.get('policy_type').title()}) benefits: {all_benefits_text}",
                    "relevance_score": 1.0
                })
            return results[:k] if results else [{"message": "No benefits information found"}]
        
        # Handle claim process queries
        elif is_claim_query:
            results = []
            for policy in customer_policies:
                claim_process = policy.get("claim_process", "Standard claim process applies")
                claim_history = policy.get("claim_history", [])
                results.append({
                    "policy_number": policy.get("policy_number"),
                    "customer_name": policy.get("customer_name"),
                    "policy_type": policy.get("policy_type"),
                    "product_name": policy.get("product_name"),
                    "claim_process": claim_process,
                    "claim_history": claim_history,
                    "relevant_info": f"Policy {policy.get('policy_number')} claim process: {claim_process}",
                    "relevance_score": 1.0
                })
            return results[:k] if results else [{"message": "No claim information found"}]
        
        # For complex or unclear queries, fall back to vector search
        else:
            if _embed is None:
                _embed = SentenceTransformer(EMB_MODEL)
            
            # Check if ChromaDB collection exists and has data
            client = chromadb.PersistentClient(path=DB_DIR, settings=Settings(allow_reset=False))
            
            try:
                col = client.get_collection("customer_policies")
            except:
                # Collection doesn't exist, create it
                create_policy_embeddings()
                col = client.get_collection("customer_policies")
            
            count = col.count()
            if count == 0:
                # No embeddings, create them
                create_policy_embeddings()
                count = col.count()
                if count == 0:
                    return [{"message": "No policies found in database", "relevant_info": "Please ensure policies are loaded"}]
            
            # Encode query
            qv = _embed.encode([query], normalize_embeddings=True).tolist()[0]
            
            # Build where clause for filtering
            where_clause = None
            if customer_email:
                where_clause = {"customer_email": customer_email}
            
            # Query with optional filter
            res = col.query(
                query_embeddings=[qv], 
                n_results=min(k, count),
                where=where_clause if where_clause else None
            )
            print(f"🎯 Vector search returned {len(res['documents'][0]) if res['documents'] else 0} results")
    
            # DEBUG: Print what vector search actually found
            if res["metadatas"] and res["documents"]:
                for i, (meta, doc, dist) in enumerate(zip(res["metadatas"][0], res["documents"][0], res["distances"][0])):
                    print(f"\n--- RESULT {i+1} ---")
                    print(f"Policy: {meta.get('policy_number')}")
                    print(f"Distance: {dist}")
                    print(f"Relevance Score: {round(1 / (1 + dist), 3)}")
                    print(f"Document snippet (first 200 chars): {doc[:200]}...")
                    print(f"Full metadata: {meta}")
            
            results = []
            if res["metadatas"] and res["documents"]:
                for meta, doc, dist in zip(res["metadatas"][0], res["documents"][0], res["distances"][0]):
                    # Extract actual policy data
                    policy_num = meta.get("policy_number")
                    
                    # Find the full policy data
                    full_policy = None
                    for p in customer_policies:
                        if p.get("policy_number") == policy_num:
                            full_policy = p
                            break
                    
                    if full_policy:
                        results.append({
                            "policy_number": policy_num,
                            "customer_name": meta.get("customer_name"),
                            "policy_type": meta.get("policy_type"),
                            "product_name": full_policy.get("product_name"),
                            "end_date": full_policy.get("end_date"),
                            "sum_insured": full_policy.get("sum_insured"),
                            "annual_premium": full_policy.get("annual_premium"),
                            "relevant_info": doc[:500],
                            "relevance_score": round(1 / (1 + dist), 3)
                        })
            
            return results if results else [{"message": "No relevant information found", "relevant_info": "Try asking a different question"}]
        
    except Exception as e:
        return [{"error": f"Policy RAG search failed: {e}", "relevant_info": "Please try again or contact support"}]

def generate_policy_response(query: str, policy_context: List[Dict[str, Any]], customer_name: str = None) -> str:
    """
    Generate a response to customer query based on their policy context
    This would use your LLM but I'll provide a template-based response for now
    """
    if not policy_context:
        return "I couldn't find any relevant information about your policies. Please ensure you have active policies with us."
    
    # Extract key information from context
    policy_info = []
    for ctx in policy_context:
        policy_info.append(f"Policy {ctx['policy_number']} ({ctx['policy_type']}): {ctx['relevant_info']}")
    
    # In production, you'd use your LLM here
    # For now, return structured information
    response = f"""Based on your policies, here's what I found:

{chr(10).join(policy_info)}

Is there anything specific about your policies you'd like to know more about?"""
    
    if customer_name:
        response = f"Hello {customer_name},\n\n" + response
    
    return response