from crewai import Task, Crew
from src.consolidated_agents import customer_sales_agent, underwriting_agent, policy_agent
from src.underwriting_vectordb import UnderwritingVectorDB
from typing import Optional

def run_customer_flow(name_or_org: str, desired_category: str, sum_insured: int = 500000, riders=None):
    """Complete customer journey using single agent"""
    
    task = Task(
        description=f"""
        Handle complete customer workflow for '{name_or_org}' requesting {desired_category} insurance:
        
        STEP 1: Find the prospect using Lookup Person tool
        STEP 2: Research suitable {desired_category} products using Get Products and Brochure Search tools
        STEP 3: Calculate accurate premiums using Estimate Premium tool for each product
        
        CRITICAL REQUIREMENTS:
        - Use actual tools to get real data and calculations
        - Present top 3 products with real calculated premiums
        - Format as: Product | Company | Coverage: ₹{sum_insured:,} | Premium: ₹[calculated] | Benefits
        - Include key selling points from brochures
        
        Sum Insured: {sum_insured}
        Riders: {riders or []}
        """,
        expected_output="Complete quote comparison with real calculated premiums and product highlights",
        agent=customer_sales_agent
    )
    
    crew = Crew(agents=[customer_sales_agent], tasks=[task], verbose=True)
    return crew.kickoff()

def run_customer_email_flow(name_or_org: str, desired_category: str, sum_insured: int = 500000, riders=None):
    """Customer journey with email using single agent"""
    
    task = Task(
        description=f"""
        Complete customer workflow with email for '{name_or_org}' requesting {desired_category} insurance:
        
        STEP 1: Find prospect using Lookup Person tool
        STEP 2: Research products using Get Products and Brochure Search tools  
        STEP 3: Calculate premiums using Estimate Premium tool
        STEP 4: Send personalized email using Send Email tool
        
        Email must include:
        - Personalized greeting with prospect's name
        - Product comparison table with calculated premiums
        - Key benefits from brochures
        - Clear call to action
        
        CRITICAL: Actually use Send Email tool with prospect's email from step 1
        """,
        expected_output="Email sent confirmation with complete email content",
        agent=customer_sales_agent
    )
    
    crew = Crew(agents=[customer_sales_agent], tasks=[task], verbose=True)
    return crew.kickoff()

def run_sales_flow(name_or_org: str, product_category: str):
    """Sales workflow using single agent"""
    
    task = Task(
        description=f"""
        Complete sales workflow for prospect '{name_or_org}' interested in {product_category}:
        
        STEP 1: Find prospect using Lookup Person tool
        STEP 2: Create CRM contact using CRM Contact tool
        STEP 3: Research {product_category} products using Brochure Search tool
        STEP 4: Send outreach email using Send Email tool
        
        Email should be professional and include:
        - Personalized greeting
        - Relevant product benefits
        - Clear next steps
        
        Execute all tools in sequence.
        """,
        expected_output="CRM contact created and outreach email sent with confirmations",
        agent=customer_sales_agent
    )
    
    crew = Crew(agents=[customer_sales_agent], tasks=[task], verbose=True)
    return crew.kickoff()

def run_underwriter_flow(product_category: str, submitted_docs_summary_text: str):
    """Complete underwriting analysis using single agent"""
    
    task = Task(
        description=f"""
        Perform comprehensive underwriting analysis for {product_category} insurance application:
        
        DOCUMENT TEXT TO ANALYZE:
        {submitted_docs_summary_text}
        
        YOUR COMPLETE ANALYSIS MUST INCLUDE:
        
        1. DOCUMENT EXTRACTION:
        - Applicant demographics (name, age, occupation, income)
        - Medical information (conditions, medications, BMI, family history)
        - Financial profile and adequacy
        - Document inventory from the text
        
        2. RISK ASSESSMENT:
        - Calculate risk score (0-15 scale) based on:
          * Age factors
          * Medical conditions and medications  
          * Smoking/lifestyle factors
          * BMI and health indicators
          * Family medical history
        - Classify risk level (Low/Medium/High/Very High)
        - Identify specific risk factors
        
        3. COMPLIANCE CHECK:
        Use Docs Required tool to get standard requirements for {product_category}, then analyze:
        - Check presence of each required document
        - Identify missing critical items
        - Note any pending medical tests needed
        - Special requirements for high sum assured
        
        4. UNDERWRITING DECISION:
        Based on risk score and compliance status, provide:
        
        DECISION: [ACCEPT/ACCEPT WITH CONDITIONS/ACCEPT WITH RATE-UP/DECLINE]
        
        RISK SCORE: X/15 (Risk Level: Low/Medium/High)
        
        RATIONALE:
        - [Specific medical/risk factors]
        - [Document compliance status]
        - [Key decision drivers]
        
        CONDITIONS (if applicable):
        - [Specific conditions required]
        
        PREMIUM ADJUSTMENT: [Standard/+X%/Decline]
        
        NEXT STEPS:
        - [Immediate actions required]
        
        Use quantitative analysis and standard underwriting guidelines.
        """,
        expected_output="Structured underwriting decision with risk score, rationale, and clear next steps",
        agent=underwriting_agent
    )
    
    crew = Crew(agents=[underwriting_agent], tasks=[task], verbose=True)
    return crew.kickoff()

def run_policy_chat_flow(query: str, customer_email: str = None, customer_name: str = None):
    """Enhanced policy chat using single agent with better query understanding"""
    
    task = Task(
        description=f"""
        Answer customer policy question: "{query}"
        
        Customer: {customer_name or 'Not specified'}
        Email: {customer_email or 'Not provided'}
        
        CRITICAL REQUIREMENTS:
        1. Use Policy RAG Search tool with the exact query: "{query}"
        2. If customer_email provided, include it in search for filtering
        3. Analyze the query type and provide appropriate response:
        
        QUERY TYPE ANALYSIS:
        - EXCLUSIONS/NOT COVERED: If query asks about "not included", "not covered", "excluded", "what's not", provide exclusions
        - COVERAGE/AMOUNTS: If asking about coverage amounts, sum insured, provide specific monetary values
        - BENEFITS/FEATURES: If asking about benefits, features, what's included, list actual benefits
        - EXPIRY/RENEWAL: If asking about when policies expire, renewal dates, provide specific dates
        - PREMIUMS/COSTS: If asking about premiums, costs, payments, provide exact amounts
        - CLAIMS: If asking about claim process, how to claim, provide claim procedures
        
        RESPONSE FORMAT:
        - Give direct, specific answers using actual policy data
        - For exclusions: List what is NOT covered for each relevant policy
        - For coverage: List exact amounts with currency formatting (₹X,XXX)
        - For benefits: List actual benefits from policy data
        - For dates: Provide specific dates in DD-MM-YYYY format
        - Be conversational but accurate
        - If multiple policies, organize by policy type
        
        NEVER say "I will search" - provide the actual answer using search results.
        Always base answers on the Policy RAG Search tool results.
        """,
        expected_output="Direct, specific answer with actual policy details organized by query type",
        agent=policy_agent
    )
    
    crew = Crew(agents=[policy_agent], tasks=[task], verbose=True)
    return crew.kickoff()

def run_policy_lookup_flow(customer_name: str = None, customer_email: str = None):
    """Policy lookup using single agent"""
    
    task = Task(
        description=f"""
        Find and summarize all policies for customer:
        - Name: {customer_name or 'Not provided'}
        - Email: {customer_email or 'Not provided'}
        
        STEP 1: Use Policy Search tool to find all customer policies
        STEP 2: Create formatted summary
        
        FORMAT EXACTLY AS:
        
        POLICY PORTFOLIO SUMMARY
        ========================
        Customer: [Name]
        Total Active Policies: [count]
        Total Annual Premium: ₹[total]
        Total Coverage: ₹[total]
        
        POLICY DETAILS:
        ---------------
        [For each policy:]
        • [Policy Type] - [Policy Number]
          - Product: [Name]
          - Coverage: ₹[Amount]
          - Premium: ₹[Amount]/year
          - Valid Till: [Date]
          - Status: [Status]
        
        Use actual data from Policy Search tool results.
        """,
        expected_output="Formatted policy portfolio summary with actual customer data",
        agent=policy_agent
    )
    
    crew = Crew(agents=[policy_agent], tasks=[task], verbose=True)
    return crew.kickoff()

def run_enhanced_underwriter_flow(doc_type: str, content: str, filename: str, 
                                 vectordb: UnderwritingVectorDB) -> str:
    """
    Enhanced underwriting flow with vector database support
    
    Args:
        doc_type: Type of insurance document
        content: Document content
        filename: Original filename
        vectordb: UnderwritingVectorDB instance
    
    Returns:
        Comprehensive underwriting analysis
    """
    from src.consolidated_agents import underwriting_agent
    from crewai import Task, Crew
    
    # Index the document
    index_result = vectordb.index_document(content, filename, doc_type)
    
    if "error" in index_result:
        return f"Error indexing document: {index_result['error']}"
    
    # Perform comprehensive analysis using RAG
    key_queries = [
        "What are the applicant's medical conditions and risk factors?",
        "What is the financial profile and income details?",
        "What documents are submitted and what is missing?",
        "What are the key underwriting concerns?",
        "What is the family medical history?"
    ]
    
    rag_results = []
    for query in key_queries:
        result = vectordb.analyze_with_rag(query, doc_type)
        if "relevant_sections" in result:
            rag_results.append(result)
    
    # Combine RAG results for comprehensive context
    combined_context = "\n\n".join([
        f"Query: {r['query']}\nRelevant Information:\n" + 
        "\n".join([s['content'][:500] for s in r['relevant_sections'][:2]])
        for r in rag_results if r.get('relevant_sections')
    ])
    
    # Create enhanced task with RAG context
    task = Task(
        description=f"""
        Perform comprehensive underwriting analysis for {doc_type} insurance application.
        
        ORIGINAL DOCUMENT:
        {content[:3000]}  # First 3000 chars for direct reference
        
        ENHANCED ANALYSIS FROM VECTOR SEARCH:
        {combined_context}
        
        DOCUMENT STATISTICS:
        - Document ID: {index_result['doc_id']}
        - Chunks analyzed: {index_result['chunks_indexed']}
        - Document type: {doc_type}
        
        PROVIDE COMPREHENSIVE UNDERWRITING DECISION INCLUDING:
        1. Applicant profile summary
        2. Risk assessment with specific factors
        3. Medical/Financial analysis (as applicable)
        4. Document compliance status
        5. Clear underwriting recommendation
        6. Premium adjustment if needed
        7. Next steps
        
        Base your analysis on both the original document and the enhanced search results.
        """,
        expected_output="Comprehensive underwriting decision with risk assessment",
        agent=underwriting_agent
    )
    
    crew = Crew(agents=[underwriting_agent], tasks=[task], verbose=True)
    return crew.kickoff()


def run_enhanced_underwriter_chat(query: str, vectordb: UnderwritingVectorDB, 
                                 doc_type: Optional[str] = None) -> str:
    """
    Enhanced chat flow using vector database for specific queries
    
    Args:
        query: User's question
        vectordb: UnderwritingVectorDB instance
        doc_type: Optional document type filter
    
    Returns:
        Detailed answer based on indexed documents
    """
    from src.consolidated_agents import underwriting_agent
    from crewai import Task, Crew
    
    # Perform RAG search for the query
    rag_result = vectordb.analyze_with_rag(query, doc_type)
    
    if "error" in rag_result:
        return f"Unable to find relevant information: {rag_result['error']}"
    
    # Extract relevant sections
    relevant_content = "\n\n".join([
        f"[Document {i+1} - {section['doc_type']} insurance]:\n{section['content']}"
        for i, section in enumerate(rag_result.get('relevant_sections', [])[:3])
    ])
    
    # Create focused task
    task = Task(
        description=f"""
        Answer this specific question based on indexed insurance documents: "{query}"
        
        RELEVANT DOCUMENT SECTIONS (from vector search):
        {relevant_content}
        
        SEARCH STATISTICS:
        - Total matches found: {rag_result.get('total_matches', 0)}
        - Documents analyzed: {len(rag_result.get('relevant_sections', []))}
        
        INSTRUCTIONS:
        1. Answer the question directly using the provided document sections
        2. Be specific and cite actual values/details from the documents
        3. If the information is not in the provided sections, say so clearly
        4. Provide a comprehensive answer with all relevant details
        
        DO NOT make up information not present in the document sections.
        """,
        expected_output="Detailed, specific answer based on document content",
        agent=underwriting_agent
    )
    
    crew = Crew(agents=[underwriting_agent], tasks=[task], verbose=True)
    return crew.kickoff()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_workflow_inputs(mode: str, **kwargs) -> dict:
    """Validate inputs for different workflows"""
    if mode == "customer":
        query = kwargs.get("query")
        category = kwargs.get("category")
        if not query or not category:
            return {"error": "Customer flow requires query and category"}
    elif mode == "sales":
        query = kwargs.get("query")
        category = kwargs.get("category")
        if not query or not category:
            return {"error": "Sales flow requires prospect name and product category"}
    elif mode == "uw":
        category = kwargs.get("category")
        if not category:
            return {"error": "Underwriter flow requires product category"}
    return {"status": "valid"}

def get_workflow_summary(mode: str) -> str:
    """Get summary of what each workflow does"""
    summaries = {
        "customer": "Find prospect → Research products → Generate quotes (1 agent)",
        "sales": "Find prospect → Create CRM contact → Send outreach email (1 agent)", 
        "uw": "Analyze documents → Check compliance → Make decision (1 agent)",
        "policy": "Search policies → Answer questions (1 agent)"
    }
    return summaries.get(mode, "Unknown workflow")

