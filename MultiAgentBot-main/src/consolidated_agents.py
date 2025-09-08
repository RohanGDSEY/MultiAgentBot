from crewai import Agent
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Any, List, Dict, Optional
from src.config import get_llm
from src import tools
import json

llm = get_llm()

# Create input schemas for tools
class LookupPersonInput(BaseModel):
    query: str = Field(..., description="Name, org, email, or phone of person to look up")

class BrochureSearchInput(BaseModel):
    query: str = Field(..., description="Query text to search in brochures")
    k: Optional[int] = Field(3, description="Number of top results to return")

class GetProductsInput(BaseModel):
    category: str = Field(..., description="Interest or product category (e.g., health, auto, life)")

class EstimatePremiumInput(BaseModel):
    person_profile: Dict[str, Any] = Field(..., description="Person details including age and risk_factors")
    product_details: Dict[str, Any] = Field(..., description="Product details including base_rate, factors")
    sum_insured: Optional[int] = Field(500000, description="Requested sum insured")
    riders: Optional[List[str]] = Field([], description="List of rider codes applied")

class CRMContactInput(BaseModel):
    person: Dict[str, Any] = Field(..., description="Person data to create a CRM contact")

class SendEmailInput(BaseModel):
    to: str = Field(..., description="Recipient email")
    subject: str = Field(..., description="Email subject")
    body: str = Field(..., description="Email body content")
    attachments: Optional[List[str]] = Field([], description="List of attachment file paths")

class DocsRequiredInput(BaseModel):
    category: str = Field(..., description="Product category to fetch required docs for")

class PolicySearchInput(BaseModel):
    customer_name: Optional[str] = Field(None, description="Customer name to search")
    customer_email: Optional[str] = Field(None, description="Customer email to search")

class PolicyRAGInput(BaseModel):
    query: str = Field(..., description="Query about policies")
    customer_email: Optional[str] = Field(None, description="Filter by customer email")
    k: Optional[int] = Field(3, description="Number of results to return")

# Create ALL tools needed
class LookupPersonTool(BaseTool):
    name: str = "Lookup Person"
    description: str = "Finds a person/org from local JSON by name or org"
    args_schema: Type[LookupPersonInput] = LookupPersonInput
    def _run(self, query: str) -> str:
        result = tools.lookup_person(query)
        return json.dumps(result, indent=2)

class BrochureSearchTool(BaseTool):
    name: str = "Brochure Search"
    description: str = "Searches brochures for suitable insurance products"
    args_schema: Type[BrochureSearchInput] = BrochureSearchInput
    def _run(self, query: str, k: int = 3) -> str:
        result = tools.brochure_search(query, k)
        return json.dumps(result, indent=2)

class GetProductsTool(BaseTool):
    name: str = "Get Products"
    description: str = "Gets products by interest/category"
    args_schema: Type[GetProductsInput] = GetProductsInput
    def _run(self, category: str) -> str:
        result = tools.get_products_for_interest(category)
        return json.dumps(result, indent=2)

class EstimatePremiumTool(BaseTool):
    name: str = "Estimate Premium"
    description: str = "Computes estimated premium for a product and person profile"
    args_schema: Type[EstimatePremiumInput] = EstimatePremiumInput
    def _run(self, person_profile: Dict[str, Any], product_details: Dict[str, Any],
             sum_insured: int = 500000, riders: List[str] = None) -> str:
        try:
            if not person_profile or "age" not in person_profile:
                return json.dumps({"error": "Person profile must include 'age' field"})
            if not product_details or "base_rate" not in product_details:
                return json.dumps({"error": "Product details must include 'base_rate' field"})
            
            enhanced_product = product_details.copy()
            if "id" not in enhanced_product:
                enhanced_product["id"] = enhanced_product.get("name", "unknown").lower().replace(" ", "_")
            if "age_factor" not in enhanced_product:
                enhanced_product["age_factor"] = [1.0, 1.2, 1.5, 2.0]
            if "industry_factor" not in enhanced_product:
                enhanced_product["industry_factor"] = {"manufacturing": 1.1, "retail": 1.0, "technology": 0.9, "finance": 1.0}
            if "sum_insured_bands" not in enhanced_product:
                enhanced_product["sum_insured_bands"] = [300000, 500000, 1000000, 2000000]
            
            result = tools.estimate_premium(person_profile, enhanced_product, sum_insured, riders or [])
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"error": f"Premium calculation failed: {str(e)}"}, indent=2)

class CRMContactTool(BaseTool):
    name: str = "CRM Contact"
    description: str = "Creates a CRM contact in the mock database"
    args_schema: Type[CRMContactInput] = CRMContactInput
    def _run(self, person: Dict[str, Any]) -> str:
        try:
            result = tools.create_crm_contact(person)
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"error": f"CRM contact creation failed: {str(e)}"})

class SendEmailTool(BaseTool):
    name: str = "Send Email"
    description: str = "Sends an email to the prospect and stores it in outbox"
    args_schema: Type[SendEmailInput] = SendEmailInput
    def _run(self, to: str, subject: str, body: str, attachments: Optional[List[str]] = None) -> str:
        try:
            result = tools.send_email(to, subject, body, attachments or [])
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"error": f"Email sending failed: {str(e)}"})

class DocsRequiredTool(BaseTool):
    name: str = "Docs Required"
    description: str = "Get required and optional docs for a product category"
    args_schema: Type[DocsRequiredInput] = DocsRequiredInput
    def _run(self, category: str) -> str:
        result = tools.required_docs_for(category)
        return json.dumps(result, indent=2)

class PolicySearchTool(BaseTool):
    name: str = "Policy Search"
    description: str = "Search for existing customer policies"
    args_schema: Type[PolicySearchInput] = PolicySearchInput
    def _run(self, customer_name: str = None, customer_email: str = None) -> str:
        from src.tools import search_customer_policies
        result = search_customer_policies(customer_name, customer_email)
        return json.dumps(result, indent=2)

class PolicyRAGTool(BaseTool):
    name: str = "Policy RAG Search"
    description: str = "Search policy information using semantic search"
    args_schema: Type[PolicyRAGInput] = PolicyRAGInput
    def _run(self, query: str, customer_email: str = None, k: int = 3) -> str:
        from src.tools import policy_rag_search
        result = policy_rag_search(query, customer_email, k)
        return json.dumps(result, indent=2)

# Initialize all tools
lookup_tool = LookupPersonTool()
brochure_tool = BrochureSearchTool()
product_tool = GetProductsTool()
quote_tool = EstimatePremiumTool()
crm_tool = CRMContactTool()
email_tool = SendEmailTool()
docs_tool = DocsRequiredTool()
policy_search_tool = PolicySearchTool()
policy_rag_tool = PolicyRAGTool()

# 1. CUSTOMER & SALES AGENT - Handles all customer/sales workflows
customer_sales_agent = Agent(
    role="Customer & Sales Operations Specialist",
    goal="""Handle all customer and sales operations including:
    - Finding prospects and customer information
    - Researching and recommending insurance products
    - Generating accurate premium quotes
    - Creating CRM contacts and sending sales emails
    - Searching customer policies and answering policy questions""",
    backstory="""You are a versatile insurance professional who combines customer service 
    excellence with sales expertise. You can find prospects, research products, calculate 
    accurate premiums, manage CRM data, send professional emails, and provide comprehensive 
    policy support to existing customers. You excel at switching between customer service 
    mode and sales mode based on the task requirements.""",
    llm=llm,
    tools=[
        lookup_tool, brochure_tool, product_tool, quote_tool, 
        crm_tool, email_tool, policy_search_tool, policy_rag_tool
    ],
    verbose=True
)

# 2. UNDERWRITING AGENT - Handles all underwriting and risk assessment
underwriting_agent = Agent(
    role="Expert Insurance Underwriter and Risk Analyst",
    goal="""Provide comprehensive, detailed underwriting analysis by:
    - Extracting specific information from insurance application documents
    - Conducting thorough risk assessments based on actual document content
    - Providing clear, actionable underwriting recommendations
    - Answering specific questions with detailed, document-based responses""",
    backstory="""You are a senior underwriting expert with 20+ years of experience in insurance risk assessment. 
    You excel at:
    
    - Reading and analyzing complex insurance applications
    - Extracting key risk factors from medical records, financial statements, and business documents
    - Providing detailed health risk assessments including medical conditions, family history, and lifestyle factors
    - Making sound underwriting decisions based on quantitative risk analysis
    - Explaining complex underwriting concepts in clear, actionable terms
    
    You always base your analysis on the actual document content provided and give specific, 
    detailed responses rather than generic statements. You understand that each application 
    is unique and requires careful analysis of the specific information presented.
    
    When asked about health risk assessment, you provide comprehensive analysis including:
    current health status, specific medical conditions, medications, family history, 
    lifestyle factors, vital signs, and overall risk implications for insurance coverage.""",
    llm=llm,
    tools=[docs_tool],  # Only include the docs tool since that's what's needed
    verbose=True,
    max_iter=3,  # Allow more iterations for thorough analysis
    max_execution_time=120,  # Allow more time for complex analysis
    allow_delegation=False  # Ensure this agent does the work itself
)

# 3. POLICY MANAGEMENT AGENT - Handles existing policy operations
policy_agent = Agent(
    role="Policy Management & Customer Service Agent",
    goal="""Manage all existing policy operations including:
    - Finding and displaying customer policies
    - Answering policy-related questions using RAG search
    - Providing policy details, benefits, and coverage information
    - Helping with claims processes and policy terms
    - Delivering excellent customer service for policy holders""",
    backstory="""You are a policy management expert who specializes in helping existing 
    customers understand and manage their insurance policies. You have access to the complete 
    policy database and can quickly find policy information, answer coverage questions, 
    explain benefits and terms, and provide guidance on claims and renewals. You excel at 
    making complex insurance information easy to understand.""",
    llm=llm,
    tools=[policy_search_tool, policy_rag_tool],
    verbose=True
)

# Export the consolidated agents
__all__ = [
    'customer_sales_agent',
    'underwriting_agent', 
    'policy_agent'
]