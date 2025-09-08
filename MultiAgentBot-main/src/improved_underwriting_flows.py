# Create a new file: src/improved_underwriting_flows.py

from crewai import Task, Crew
from src.consolidated_agents import underwriting_agent

def run_underwriter_chat_flow(query: str, document_content: str):
    """Improved chat flow that provides detailed, specific answers"""
    
    task = Task(
        description=f"""
        You are an expert underwriter analyzing an insurance document. Answer this specific question: "{query}"
        
        DOCUMENT CONTENT:
        {document_content}
        
        ANALYSIS INSTRUCTIONS:
        1. Read the document content carefully and extract relevant information
        2. Answer the specific question directly based on the document
        3. Provide detailed, specific information from the document
        4. If the question is about health risk assessment, include:
           - Specific medical conditions mentioned
           - Risk factors identified
           - Health metrics (BMI, blood pressure, etc.)
           - Family history details
           - Lifestyle factors
           - Overall risk level assessment
        
        DO NOT mention "Docs Required tool" or generic responses.
        DO NOT say the tool doesn't have categories.
        
        PROVIDE a detailed answer based on the actual document content.
        
        For health risk assessment specifically, structure your response as:
        
        **HEALTH RISK ASSESSMENT SUMMARY**
        
        **Current Health Status:**
        [Extract specific conditions, medications, vital signs from document]
        
        **Risk Factors Identified:**
        [List specific risk factors from the document]
        
        **Family History:**
        [Extract family medical history details]
        
        **Lifestyle Assessment:**
        [Extract smoking, alcohol, exercise habits]
        
        **Overall Risk Level:**
        [Provide assessment based on above factors]
        
        **Underwriting Implications:**
        [What this means for insurance approval/pricing]
        
        Base your entire response on the actual document content provided.
        """,
        expected_output="Detailed, specific answer based on document content with structured health risk assessment",
        agent=underwriting_agent
    )
    
    crew = Crew(agents=[underwriting_agent], tasks=[task], verbose=True)
    return crew.kickoff()

def run_underwriter_flow(product_category: str, submitted_docs_summary_text: str):
    """Improved underwriting analysis with specific document focus"""
    
    task = Task(
        description=f"""
        Perform comprehensive underwriting analysis for {product_category} insurance application.
        
        DOCUMENT CONTENT TO ANALYZE:
        {submitted_docs_summary_text}
        
        COMPREHENSIVE ANALYSIS REQUIREMENTS:
        
        1. EXTRACT APPLICANT INFORMATION:
        - Name, age, occupation, income
        - Contact details and location
        - Employment details
        
        2. MEDICAL/HEALTH ASSESSMENT (for health/life insurance):
        - Current health conditions and medications
        - Vital signs (blood pressure, BMI, cholesterol, etc.)
        - Family medical history
        - Lifestyle factors (smoking, alcohol, exercise)
        - Previous hospitalizations or treatments
        
        3. FINANCIAL ASSESSMENT (for commercial insurance):
        - Business turnover and profit margins
        - Financial performance trends
        - Asset values and business operations
        
        4. RISK FACTORS IDENTIFICATION:
        - Specific risks based on the document content
        - Industry/occupation-related risks
        - Medical risks (if applicable)
        - Financial risks (if applicable)
        
        5. DOCUMENT COMPLIANCE:
        - List documents mentioned as submitted in the text
        - Identify any documents marked as missing
        - Note compliance status
        
        6. UNDERWRITING RECOMMENDATION:
        Provide clear decision with rationale:
        
        **UNDERWRITING DECISION**
        
        **Recommendation:** [ACCEPT/ACCEPT WITH CONDITIONS/DECLINE/RATE-UP]
        
        **Key Findings:**
        - [List 3-4 most important findings from document analysis]
        
        **Risk Assessment:**
        - [Specific risk factors identified]
        - [Risk level: Low/Medium/High]
        
        **Rationale:**
        - [Clear reasoning based on document content]
        
        **Premium Impact:** [Standard/+X%/Decline]
        
        **Next Steps:**
        - [Specific actions required]
        
        Base your analysis entirely on the actual document content provided.
        Do NOT use generic templates or make assumptions not supported by the document.
        """,
        expected_output="Comprehensive underwriting analysis with specific findings from the document",
        agent=underwriting_agent
    )
    
    crew = Crew(agents=[underwriting_agent], tasks=[task], verbose=True)
    return crew.kickoff()