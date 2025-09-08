import streamlit as st
import json
import os
import sys
from datetime import datetime
import pandas as pd
from PyPDF2 import PdfReader
import io
import re
import requests
from typing import Dict, Any, Optional, List

# Add src to path so we can import our modules
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

try:
    from src.consolidated_flows import run_customer_flow, run_customer_email_flow, run_sales_flow, run_enhanced_underwriter_chat,run_enhanced_underwriter_flow
    from src.underwriting_vectordb import UnderwritingVectorDB
    # from src.dynamic_underwriting_flows import run_underwriter_flow, run_underwriter_chat_flow
    from src.improved_underwriting_flows import run_underwriter_flow, run_underwriter_chat_flow
    # from src.consolidated_flows import run_customer_flow, run_customer_email_flow, run_sales_flow, run_underwriter_flow
    from src.tools import lookup_person, get_products_for_interest, required_docs_for, validate_person_category_match, analyze_application_comprehensively, enhanced_compliance_check
    from src.risk_assessment import calculate_comprehensive_risk_score, parse_medical_information, make_underwriting_decision
    from src.consolidated_flows import run_policy_chat_flow
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.error("Make sure to run: `pip install streamlit` and check your project structure")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Smart Insurance Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f2937;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .welcome-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .policy-card {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        margin: 1rem 0;
    }
    .chat-container {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1rem;
        max-height: 400px;
        overflow-y: auto;
    }
    .user-message {
        background: #dbeafe;
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
        text-align: right;
    }
    .assistant-message {
        background: #f3f4f6;
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #3b82f6, #1d4ed8);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Mock API function - replace with actual API call
def fetch_customer_data_from_api(email: str, password: str) -> Optional[Dict[str, Any]]:
    """Mock function to simulate 3rd party REST API call"""
    mock_customers = {
        "rohit.sharma@sharmamanufacturing.com": {
            "customer_id": "CUST001",
            "name": "Rohit Sharma",
            "email": "rohit.sharma@sharmamanufacturing.com",
            "phone": "+91-9000000001",
            "policies": [
                    {
                        "policy_number": "POL-HLT-2024-001",
                        "customer_name": "Rohit Sharma",
                        "customer_email": "rohit.sharma@sharmamanufacturing.com",
                        "customer_phone": "+91-9000000001",
                        "policy_type": "health",
                        "company": "Alpha Insurance",
                        "product_name": "Alpha Health Shield",
                        "insurance_company": "Alpha Insurance",
                        "status": "active",
                        "sum_insured": 1000000,
                        "annual_premium": 15600,
                        "start_date": "2024-01-15",
                        "end_date": "2025-01-14",
                        "benefits": [
                            "Cashless treatment at 5000+ hospitals",
                            "Pre and post hospitalization coverage",
                            "Day care procedures covered",
                            "Annual health checkup",
                            "Emergency ambulance coverage"
                        ],
                        "exclusions": [
                            "Pre-existing diseases (first 3 years)",
                            "Cosmetic surgery and plastic surgery",
                            "Dental treatment (unless due to accident)",
                            "Self-inflicted injuries and suicide attempts",
                            "War and nuclear risks",
                            "Experimental treatments and unproven therapies",
                            "Fertility treatments and surrogacy",
                            "Obesity treatment and weight reduction surgery"
                        ],
                        "riders": ["CRIT", "ACCI"],
                        "claim_history": [],
                        "terms_conditions": "Subject to policy terms. Pre-existing diseases covered after 3 years.",
                        "claim_process": "Notify within 48 hours. Submit documents within 7 days. Cashless or reimbursement available.",
                        "next_renewal": "2025-01-14",
                        "auto_renewal": "true"
                        },
                        {
                        "policy_number": "POL-LIF-2023-045",
                        "customer_name": "Rohit Sharma",
                        "customer_email": "rohit.sharma@sharmamanufacturing.com",
                        "customer_phone": "+91-9000000001",
                        "company": "Gamma Life",
                        "policy_type": "life",
                        "product_name": "Gamma Term Life Pro",
                        "insurance_company": "Gamma Life",
                        "status": "active",
                        "sum_insured": 5000000,
                        "annual_premium": 12000,
                        "start_date": "2023-06-01",
                        "end_date": "2043-05-31",
                        "benefits": [
                            "Death benefit to nominee",
                            "Accidental death additional coverage",
                            "Terminal illness benefit",
                            "Tax benefits under 80C",
                            "Loan facility after 3 years"
                        ],
                        "exclusions": [
                            "Suicide within first year of policy",
                            "Death due to war, invasion, or military operations",
                            "Self-inflicted injuries while sane or insane",
                            "Death while under influence of drugs or alcohol",
                            "Participation in hazardous activities like mountaineering",
                            "Death due to pre-existing diseases not disclosed",
                            "Death during participation in criminal activities"
                        ],
                        "riders": ["ACCI", "DISA"],
                        "nominee_name": "Mrs. Sharma",
                        "nominee_relationship": "Spouse",
                        "terms_conditions": "20-year term policy. Premium payment annual.",
                        "claim_process": "Nominee to submit death certificate and claim form within 90 days.",
                        "premium_paid_till": "2024-05-31",
                        "next_premium_due": "2025-05-31"
                        }
            ]
        },
        "priya.patel@techsolutions.in": {
            "customer_id": "CUST002",
            "name": "Priya Patel",
            "email": "priya.patel@techsolutions.in",
            "phone": "+91-9000000002",
            "policies": [
                {
                    "policy_number": "POL-MOT-2024-789",
                    "customer_name": "Priya Patel",
                    "company": "Delta Motors",
                    "customer_email": "priya.patel@techsolutions.in",
                    "customer_phone": "+91-9000000002",
                    "policy_type": "motor",
                    "product_name": "Delta Comprehensive Auto",
                    "insurance_company": "Delta Motors",
                    "status": "active",
                    "sum_insured": 800000,
                    "annual_premium": 28000,
                    "start_date": "2024-03-10",
                    "end_date": "2025-03-09",
                    "vehicle_details": {
                        "make": "Honda",
                        "model": "City",
                        "year": 2023,
                        "registration": "KA-01-AB-1234"
                    },
                    "benefits": [
                        "Comprehensive coverage",
                        "Zero depreciation",
                        "Cashless garage network",
                        "24x7 roadside assistance",
                        "Personal accident cover for driver"
                    ],
                    "exclusions": [
                        "Driving without valid license",
                        "Drunk driving or under influence of drugs",
                        "Racing, speed trials, or competitive events",
                        "Normal wear and tear",
                        "Consequential losses and loss of use",
                        "Depreciation (unless zero depreciation add-on)",
                        "Damage due to overloading",
                        "War risks and nuclear perils",
                        "Damage to tyres and tubes (unless vehicle damaged)",
                        "Electrical/electronic equipment damage"
                    ],
                    "riders": ["ZERO", "RSA"],
                    "claim_history": [
                        {
                        "claim_date": "2024-07-15",
                        "claim_amount": 15000,
                        "claim_type": "Minor accident repair",
                        "status": "settled"
                        }
                    ],
                    "ncb_percentage": 20,
                    "terms_conditions": "Standard motor insurance terms apply.",
                    "claim_process": "Report within 24 hours. Survey within 48 hours."
                    },
                    {
                    "policy_number": "POL-HLT-2024-234",
                    "customer_name": "Priya Patel",
                    "company": "Beta Health",
                    "customer_email": "priya.patel@techsolutions.in",
                    "customer_phone": "+91-9000000002",
                    "policy_type": "health",
                    "product_name": "Beta Family Care Plus",
                    "insurance_company": "Beta Health",
                    "status": "active",
                    "sum_insured": 1500000,
                    "annual_premium": 22000,
                    "start_date": "2024-04-01",
                    "end_date": "2025-03-31",
                    "family_members": [
                        {"name": "Priya Patel", "age": 28, "relationship": "Self"},
                        {"name": "Spouse", "age": 30, "relationship": "Spouse"}
                    ],
                    "benefits": [
                        "Family floater coverage",
                        "Maternity benefits",
                        "Newborn baby coverage",
                        "Health checkup for all members",
                        "International emergency coverage"
                    ],
                    "exclusions": [
                        "Pre-existing diseases (first 2 years)",
                        "Maternity expenses (first 2 years)",
                        "Cosmetic and plastic surgery",
                        "Dental treatment (unless accident-related)",
                        "Alternative treatments like Ayurveda (unless specified)",
                        "Self-inflicted injuries",
                        "War and nuclear risks",
                        "Congenital diseases and genetic disorders",
                        "Infertility and assisted reproduction treatments"
                    ],
                    "riders": ["MATER"],
                    "claim_history": [],
                    "terms_conditions": "Family floater policy. Maternity after 2 year waiting.",
                    "claim_process": "Cashless through network hospitals or reimbursement.",
                    "wellness_points": 1500
                    }
            ]
        }
    }
    
    import time
    time.sleep(1)
    
    if email in mock_customers and password == "customer123":
        return mock_customers[email]
    return None

# Authentication functions
def authenticate_customer(email: str, password: str) -> bool:
    """Authenticate customer and fetch their data"""
    customer_data = fetch_customer_data_from_api(email, password)
    if customer_data:
        st.session_state.customer_authenticated = True
        st.session_state.customer_data = customer_data
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        return True
    return False

def authenticate_admin(username: str, password: str) -> bool:
    """Authenticate admin user"""
    if username == "admin" and password == "admin123":
        st.session_state.admin_authenticated = True
        if "uploaded_documents" not in st.session_state:
            st.session_state.uploaded_documents = []
        if "analysis_results" not in st.session_state:
            st.session_state.analysis_results = {}
        if "underwriter_chat_history" not in st.session_state:
            st.session_state.underwriter_chat_history = []
        return True
    return False

# Initialize session states
if "customer_authenticated" not in st.session_state:
    st.session_state.customer_authenticated = False
if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False
if "customer_data" not in st.session_state:
    st.session_state.customer_data = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_documents" not in st.session_state:
    st.session_state.uploaded_documents = []
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = {}
if "underwriter_chat_history" not in st.session_state:
    st.session_state.underwriter_chat_history = []
if "document_registry" not in st.session_state:
    st.session_state.document_registry = {}
if "document_store" not in st.session_state:
    st.session_state.document_store = {
        "health": [],
        "life": [],
        "motor": [],
        "commercial": []
    }
if "current_insurance_type" not in st.session_state:
    st.session_state.current_insurance_type = None

def main():
    # Header
    st.markdown('<h1 class="main-header">Smart Insurance Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Insurance Management System</p>', unsafe_allow_html=True)
    
    # Sidebar for authentication and file operations
    with st.sidebar:
        st.markdown("### 🔐 Authentication")
        
        # Role selection
        selected_role = st.selectbox(
            "Select your role:",
            ["", "Customer", "Underwriter", "Sales Agent"],
            help="Choose your role to access relevant features"
        )
        
        if selected_role == "Customer":
            render_customer_auth()
        elif selected_role == "Underwriter":
            render_underwriter_auth()
        elif selected_role == "Sales Agent":
            render_sales_agent_auth()
    
    # Main content area
    if selected_role == "Customer" and st.session_state.customer_authenticated:
        render_customer_dashboard()
    elif selected_role == "Underwriter" and st.session_state.admin_authenticated:
        render_underwriter_dashboard()
    elif selected_role == "Sales Agent":
        render_sales_dashboard()
    else:
        render_welcome_screen()

def render_customer_auth():
    """Customer authentication in sidebar"""
    if st.session_state.customer_authenticated:
        customer = st.session_state.customer_data
        st.success(f"✅ Logged in as {customer['name']}")
        if st.button("🚪 Logout"):
            st.session_state.customer_authenticated = False
            st.session_state.customer_data = None
            st.session_state.chat_history = []
            st.rerun()
    else:
        with st.form("customer_login"):
            st.subheader("Customer Login")
            email = st.text_input("Email", placeholder="rohit.sharma@sharmamanufacturing.com")
            password = st.text_input("Password", type="password", placeholder="customer123")
            submitted = st.form_submit_button("Login", use_container_width=True)
            
            if submitted:
                if email and password:
                    with st.spinner("Authenticating..."):
                        if authenticate_customer(email, password):
                            st.success("✅ Login successful!")
                            st.rerun()
                        else:
                            st.error("❌ Invalid credentials or API error")
                else:
                    st.error("Please enter both email and password")

def render_underwriter_auth():
    """Fixed underwriter authentication with proper document management"""
    if st.session_state.admin_authenticated:
        st.success("✅ Logged in as Admin")
        
        st.markdown("### 📄 Document Upload & Management")
        
        # Insurance type selector
        insurance_type = st.selectbox(
            "Select Insurance Type:",
            ["", "health", "life", "motor", "commercial"],
            key="insurance_type_selector",
            help="Select the type of insurance documents you want to work with"
        )
        
        if insurance_type:
            st.session_state.current_insurance_type = insurance_type
            st.session_state.selected_insurance_type = insurance_type
            
            # Show current type status
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"📋 Working with: **{insurance_type.upper()}** Insurance")
            with col2:
                doc_count = len(st.session_state.document_store.get(insurance_type, []))
                st.info(f"📄 Documents: **{doc_count}**")
            
            # File upload for selected type
            uploaded_files = st.file_uploader(
                f"Upload {insurance_type} insurance documents",
                accept_multiple_files=True,
                type=['pdf', 'txt'],
                key=f"upload_{insurance_type}",
                help=f"Upload documents for {insurance_type} insurance analysis"
            )
            
            if uploaded_files:
                new_docs = []
                for file in uploaded_files:
                    # Check if already uploaded
                    existing = False
                    for doc in st.session_state.document_store[insurance_type]:
                        if doc["filename"] == file.name:
                            existing = True
                            break
                    
                    if not existing:
                        # Register new document
                        doc_entry = {
                            "filename": file.name,
                            "file_obj": file,
                            "insurance_type": insurance_type,
                            "uploaded_at": datetime.now().isoformat(),
                            "size": file.size,
                            "type": file.type,
                            "analyzed": False,
                            "analysis": None,
                            "content": None
                        }
                        st.session_state.document_store[insurance_type].append(doc_entry)
                        new_docs.append(file.name)
                
                if new_docs:
                    st.success(f"✅ Added {len(new_docs)} new {insurance_type} document(s)")
                    for name in new_docs:
                        st.write(f"  📄 {name}")
                else:
                    st.info("All files already uploaded")
            
            # Show all documents for this type
            docs = st.session_state.document_store.get(insurance_type, [])
            if docs:
                st.markdown(f"#### 📚 {insurance_type.title()} Insurance Documents")
                for doc in docs:
                    status = "✅ Analyzed" if doc["analyzed"] else "⏳ Pending"
                    st.write(f"• {doc['filename']} - {status}")
        else:
            st.warning("⚠️ Please select an insurance type to begin")
        
        # Show document summary
        st.markdown("### 📊 Document Summary")
        cols = st.columns(4)
        for i, (ins_type, docs) in enumerate(st.session_state.document_store.items()):
            with cols[i]:
                st.metric(ins_type.title(), len(docs))
        
        if st.button("🚪 Logout"):
            st.session_state.admin_authenticated = False
            st.session_state.document_store = {
                "health": [], "life": [], "motor": [], "commercial": []
            }
            st.session_state.current_insurance_type = None
            st.session_state.analysis_results = {}
            st.session_state.underwriter_chat_history = []
            st.rerun()
    else:
        # Login form (unchanged)
        with st.form("admin_login"):
            st.subheader("Admin Login")
            username = st.text_input("Username", placeholder="admin")
            password = st.text_input("Password", type="password", placeholder="admin123")
            submitted = st.form_submit_button("Login", use_container_width=True)
            
            if submitted:
                if username == "admin" and password == "admin123":
                    st.session_state.admin_authenticated = True
                    st.success("✅ Admin login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials")

def render_sales_agent_auth():
    """Sales agent authentication"""
    pass
    # st.info("🏗️ Sales Agent features coming soon!")

def render_customer_dashboard():
    """Customer dashboard with policy info and chat"""
    customer = st.session_state.customer_data
    
    # Welcome section
    st.markdown(f"""
    <div class="welcome-card">
        <h2>Hi {customer['name']}, Welcome! 👋</h2>
        <p>Manage your policies and get instant answers to your insurance questions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Policy overview
    policies = customer.get('policies', [])
    
    col1, col2, col3= st.columns(3)
    with col1:
        st.metric("Active Policies", len(policies))
    with col2:
        total_coverage = sum(p.get('sum_insured', 0) for p in policies)
        st.metric("Total Coverage", f"₹{total_coverage:,}")
    with col3:
        total_premium = sum(p.get('annual_premium', 0) for p in policies)
        st.metric("Annual Premium", f"₹{total_premium:,}")

    # Policy details
    st.markdown("### 📋 Your Policies")
    
    for policy in policies:
        with st.expander(f"📄 {policy.get('policy_type', 'Unknown').title()} - {policy.get('policy_number', 'N/A')}", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Policy Details**")
                st.write(f"Product: {policy.get('product_name', 'N/A')}")
                
                # Safe access to company field - check both possible field names
                company = policy.get('company') or policy.get('insurance_company', 'N/A')
                st.write(f"Company: {company}")
                
                st.write(f"Status: {policy.get('status', 'N/A').title()}")
            
            with col2:
                st.write("**Coverage**")
                sum_insured = policy.get('sum_insured', 0)
                annual_premium = policy.get('annual_premium', 0)
                st.write(f"Sum Insured: ₹{sum_insured:,}")
                st.write(f"Premium: ₹{annual_premium:,}/year")
            
            with col3:
                st.write("**Validity**")
                st.write(f"Start: {policy.get('start_date', 'N/A')}")
                st.write(f"End: {policy.get('end_date', 'N/A')}")
                
                # Show next renewal if available
                if policy.get('next_renewal'):
                    st.write(f"Next Renewal: {policy['next_renewal']}")
                
                # Show auto-renewal status
                auto_renewal = policy.get('auto_renewal', False)
                if isinstance(auto_renewal, str):
                    auto_renewal = auto_renewal.lower() == 'true'
                st.write(f"Auto Renewal: {'Yes' if auto_renewal else 'No'}")
    
    # Chat interface
    render_customer_chat()

def render_customer_chat():
    """Customer chat interface for policy queries - FIXED VERSION"""
    st.markdown("### 💬 Ask About Your Policies")
    
    # Sample questions
    # with st.expander("💡 Sample Questions", expanded=False):
    #     sample_questions = [
    #         "What is my total coverage amount?",
    #         "When do my policies expire?",
    #         "How do I file a claim?",
    #         "What are the benefits of my health policy?",
    #         "What's my total annual premium?"
    #     ]
        
    #     # Create columns for sample questions
    #     cols = st.columns(len(sample_questions))
    #     for i, question in enumerate(sample_questions):
    #         with cols[i]:
    #             if st.button(f"❓", key=f"sample_{i}", help=question):
    #                 # Add to chat history immediately
    #                 st.session_state.chat_history.append({"role": "user", "content": question})
    #                 # Process the query
    #                 process_customer_query(question)
    #                 # Force rerun to display the result
    #                 st.rerun()
    
    # Chat history display - ALWAYS SHOW FIRST
    if st.session_state.chat_history:
        st.markdown("**Recent Conversations:**")
        
        # Display chat messages in reverse order (newest first)
        for message in (st.session_state.chat_history[-10:]):  # Show last 10 messages
            if message['role'] == 'user':
                st.markdown(f"""
                <div class="user-message">
                    <strong>You:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="assistant-message">
                    <strong>Assistant:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input - FIXED TO PREVENT DISAPPEARING
    user_query = st.text_area(
        "Ask me anything about your policies:",
        placeholder="Type your question here...",
        height=100,
        key="customer_chat_input"
    )
    
    if st.button("Send 💬", use_container_width=True):
        if user_query:
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            # Process the query
            process_customer_query(user_query)
            # Clear the input and rerun
            # st.session_state.customer_chat_input = ""
            st.rerun()

def process_customer_query(query: str):
    """Process customer query using policy chat flow - FIXED VERSION"""
    customer = st.session_state.customer_data
    
    try:
        # Use the existing policy chat flow
        response = run_policy_chat_flow(
            query=query,
            customer_email=customer['email'],
            customer_name=customer['name']
        )
        
        # Add response to chat history immediately
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": str(response)
        })
        
    except Exception as e:
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": error_msg
        })

def format_underwriter_response(response: str, document_type: str, query: str) -> str:
    """Format underwriter responses with better visual appeal"""
    
    # Create header based on document type and query
    header_emoji = {
        'commercial': '🏢',
        'health': '🏥', 
        'life': '👨‍👩‍👧‍👦',
        'motor': '🚗',
        'general': '📋'
    }
    
    emoji = header_emoji.get(document_type, '📋')
    
    formatted_response = f"""
## {emoji} {document_type.title()} Insurance Analysis

### 📝 Query: "{query}"

---

{response}

---
💡 **Tip:** Ask more specific questions like:
- "What is the exact premium recommendation?"
- "List all missing documents"
- "What are the top 3 risk concerns?"
"""
    
    return formatted_response

def render_underwriter_dashboard():
    """Fixed dashboard that properly handles insurance types"""
    st.markdown("### 🔍 Underwriting Analysis Dashboard")
    
    current_type = st.session_state.get("current_insurance_type")
    
    if not current_type:
        st.warning("⚠️ Please select an insurance type from the sidebar first")
        return
    
    st.info(f"🎯 Currently working with: **{current_type.upper()}** Insurance")
    
    # Get documents for current type
    current_docs = st.session_state.document_store.get(current_type, [])
    
    # Stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(f"📄 {current_type.title()} Docs", len(current_docs))
    with col2:
        analyzed = sum(1 for d in current_docs if d.get("analyzed", False))
        st.metric("✅ Analyzed", analyzed)
    with col3:
        pending = len(current_docs) - analyzed
        st.metric("⏳ Pending", pending)
    with col4:
        st.metric("🤖 Status", "Ready")
    
    if not current_docs:
        st.info(f"No {current_type} insurance documents uploaded yet. Please upload from sidebar.")
        return
    
    # Document analysis section
    st.markdown(f"### 📄 {current_type.title()} Insurance Documents")
    
    for i, doc in enumerate(current_docs):
        with st.expander(f"📄 {doc['filename']}", expanded=False):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**Type:** {current_type} insurance")
                st.write(f"**Size:** {doc['size']} bytes")
                st.write(f"**Uploaded:** {doc['uploaded_at'][:19]}")
                if doc['analyzed']:
                    st.write("**Status:** ✅ Analyzed")
            
            with col2:
                if not doc['analyzed']:
                    if st.button(f"🔍 Analyze", key=f"analyze_{current_type}_{i}"):
                        with st.spinner(f"Analyzing {current_type} document..."):
                            analyze_document_for_type(doc, current_type)
                            st.rerun()
                else:
                    if st.button(f"🔄 Re-analyze", key=f"reanalyze_{current_type}_{i}"):
                        with st.spinner(f"Re-analyzing {current_type} document..."):
                            analyze_document_for_type(doc, current_type)
                            st.rerun()
            
            # Show analysis if available
            if doc.get('analyzed') and doc.get('analysis'):
                st.markdown("#### 📊 Analysis Result")
                st.text_area("", value=doc['analysis'], height=200, key=f"analysis_{current_type}_{i}")
    
    # Chat interface
    render_type_specific_chat()

def analyze_document_for_type(doc: Dict[str, Any], insurance_type: str):
    """Analyze a document with proper type context"""
    try:
        # Extract content if not already done
        if not doc.get("content"):
            file_obj = doc["file_obj"]
            if file_obj.type == "text/plain":
                doc["content"] = str(file_obj.read(), "utf-8")
            elif file_obj.type == "application/pdf":
                pdf_reader = PdfReader(io.BytesIO(file_obj.read()))
                doc["content"] = "\n".join([p.extract_text() or "" for p in pdf_reader.pages])
            else:
                doc["content"] = "Unable to extract content"
        
        # Run analysis with proper type
        from src.improved_underwriting_flows import run_underwriter_flow
        result = run_underwriter_flow(insurance_type, doc["content"])
        
        # Update document
        doc["analyzed"] = True
        doc["analysis"] = str(result)
        
        # Store in analysis results
        if "analysis_results" not in st.session_state:
            st.session_state.analysis_results = {}
        
        st.session_state.analysis_results[doc["filename"]] = {
            "content": doc["content"],
            "analysis": str(result),
            "document_type": insurance_type,
            "timestamp": datetime.now().isoformat()
        }
        
        st.success(f"✅ Analysis complete for {doc['filename']}")
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        doc["analyzed"] = False
        doc["analysis"] = f"Error: {str(e)}"

def render_type_specific_chat():
    """Chat interface that respects insurance type"""
    current_type = st.session_state.get("current_insurance_type")
    
    if not current_type:
        st.warning("Please select an insurance type first")
        return
    
    st.markdown(f"### 💬 Ask About {current_type.title()} Insurance Documents")
    
    # Check if we have analyzed documents
    current_docs = st.session_state.document_store.get(current_type, [])
    analyzed_docs = [d for d in current_docs if d.get("analyzed")]
    
    if not analyzed_docs:
        st.info(f"Please analyze at least one {current_type} document first")
        return
    
    # Quick questions based on type
    # st.markdown("**Quick Questions:**")
    
    # if current_type == "health":
    #     cols = st.columns(3)
    #     with cols[0]:
    #         if st.button("🏥 Health Risks"):
    #             process_type_specific_query("What are the health risks and medical conditions?", current_type)
    #     with cols[1]:
    #         if st.button("💊 Medications"):
    #             process_type_specific_query("What medications are mentioned?", current_type)
    #     with cols[2]:
    #         if st.button("📊 Risk Score"):
    #             process_type_specific_query("What is the risk assessment?", current_type)
    
    # elif current_type == "life":
    #     cols = st.columns(3)
    #     with cols[0]:
    #         if st.button("👤 Beneficiaries"):
    #             process_type_specific_query("Who are the beneficiaries?", current_type)
    #     with cols[1]:
    #         if st.button("💰 Sum Assured"):
    #             process_type_specific_query("What is the sum assured?", current_type)
    #     with cols[2]:
    #         if st.button("⚠️ Risk Factors"):
    #             process_type_specific_query("What are the mortality risk factors?", current_type)
    
    # elif current_type == "motor":
    #     cols = st.columns(3)
    #     with cols[0]:
    #         if st.button("🚗 Vehicle"):
    #             process_type_specific_query("What are the vehicle details?", current_type)
    #     with cols[1]:
    #         if st.button("👤 Driver"):
    #             process_type_specific_query("What is the driver history?", current_type)
    #     with cols[2]:
    #         if st.button("📋 Coverage"):
    #             process_type_specific_query("What coverage is requested?", current_type)
    
    # elif current_type == "commercial":
    #     cols = st.columns(3)
    #     with cols[0]:
    #         if st.button("💼 Business"):
    #             process_type_specific_query("What is the business profile?", current_type)
    #     with cols[1]:
    #         if st.button("💰 Financials"):
    #             process_type_specific_query("What are the financials?", current_type)
    #     with cols[2]:
    #         if st.button("⚠️ Risks"):
    #             process_type_specific_query("What are the risk exposures?", current_type)
    
    # Chat history
    if "underwriter_chat_history" not in st.session_state:
        st.session_state.underwriter_chat_history = []
    
    if st.session_state.underwriter_chat_history:
        st.markdown("**Recent Q&A:**")
        for msg in st.session_state.underwriter_chat_history[-4:]:
            if msg['role'] == 'user':
                st.markdown(f"**❓ Q:** {msg['content']}")
            else:
                st.markdown(f"**🤖 A:** {msg['content']}")
                st.markdown("---")
    
    # Chat input
    with st.form("chat_form", clear_on_submit=True):
        query = st.text_area(
            f"Ask about {current_type} insurance documents:",
            placeholder=f"Type your question about {current_type} insurance...",
            height=100
        )
        submitted = st.form_submit_button(f"Ask about {current_type.title()}", use_container_width=True)
        
        if submitted and query:
            process_type_specific_query(query, current_type)
            st.rerun()

def process_type_specific_query(query: str, insurance_type: str):
    """Process queries for specific insurance type"""
    try:
        # Get analyzed documents for this type
        docs = st.session_state.document_store.get(insurance_type, [])
        analyzed_docs = [d for d in docs if d.get("analyzed") and d.get("content")]
        
        if not analyzed_docs:
            response = f"No analyzed {insurance_type} documents found. Please analyze documents first."
            st.session_state.underwriter_chat_history.append({"role": "assistant", "content": response})
            return
        
        # Combine content from all analyzed docs of this type
        combined_content = f"=== {insurance_type.upper()} INSURANCE DOCUMENTS ===\n\n"
        for doc in analyzed_docs:
            combined_content += f"--- Document: {doc['filename']} ---\n"
            combined_content += doc['content'][:3000] + "\n\n"
        
        # Add user query to history
        st.session_state.underwriter_chat_history.append({"role": "user", "content": query})
        
        # Process with type context
        with st.spinner(f"Analyzing {insurance_type} documents..."):
            from src.improved_underwriting_flows import run_underwriter_chat_flow
            response = run_underwriter_chat_flow(query, combined_content)
            formatted_response = f"**{insurance_type.title()} Insurance Analysis:**\n\n{str(response)}"
        
        st.session_state.underwriter_chat_history.append({"role": "assistant", "content": formatted_response})
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        st.session_state.underwriter_chat_history.append({"role": "assistant", "content": error_msg})

def render_simplified_underwriter_chat():
    """Simplified underwriter chat interface with error handling"""
    st.markdown("### 💬 Ask Questions About Your Document")
    
    # Show current document type clearly with None check
    if ("selected_insurance_type" in st.session_state and 
        st.session_state.selected_insurance_type and 
        st.session_state.uploaded_documents):
        
        insurance_type = st.session_state.selected_insurance_type
        st.info(f"🔍 **Analyzing:** {insurance_type.title()} Insurance Documents")
        
        # Simplified quick questions based on insurance type
        st.markdown("**Quick Questions:**")
        
        if insurance_type == 'commercial':
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("💰 Profit Margin", use_container_width=True):
                    add_and_process_simple_query("What is the profit margin?")
            with col2:
                if st.button("⚠️ Risk Factors", use_container_width=True):
                    add_and_process_simple_query("What are the risk factors?")
            with col3:
                if st.button("📋 Missing Docs", use_container_width=True):
                    add_and_process_simple_query("What documents are missing?")
        
        elif insurance_type == 'health':
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🏥 Medical Conditions", use_container_width=True):
                    add_and_process_simple_query("What medical conditions are mentioned?")
            with col2:
                if st.button("📊 Risk Assessment", use_container_width=True):
                    add_and_process_simple_query("What is the health risk assessment?")
            with col3:
                if st.button("💊 Medications", use_container_width=True):
                    add_and_process_simple_query("What medications is the applicant taking?")
        
        elif insurance_type == 'motor':
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🚗 Vehicle Details", use_container_width=True):
                    add_and_process_simple_query("What are the vehicle details?")
            with col2:
                if st.button("👤 Driver Profile", use_container_width=True):
                    add_and_process_simple_query("What is the driver's profile and experience?")
            with col3:
                if st.button("📋 Coverage Details", use_container_width=True):
                    add_and_process_simple_query("What coverage is being requested?")
        
        elif insurance_type == 'life':
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("👤 Applicant Profile", use_container_width=True):
                    add_and_process_simple_query("What is the applicant's profile?")
            with col2:
                if st.button("🏥 Health Assessment", use_container_width=True):
                    add_and_process_simple_query("What is the health and medical assessment?")
            with col3:
                if st.button("💰 Sum Assured", use_container_width=True):
                    add_and_process_simple_query("What is the sum assured and premium details?")
    
    elif st.session_state.uploaded_documents and not st.session_state.get("selected_insurance_type"):
        st.warning("⚠️ Please select an insurance type from the dropdown in the sidebar to enable analysis features.")
    
    # Show chat history
    if st.session_state.underwriter_chat_history:
        st.markdown("**Recent Questions & Answers:**")
        for message in st.session_state.underwriter_chat_history[-4:]:
            if message['role'] == 'user':
                st.markdown(f"**❓ Question:** {message['content']}")
            else:
                st.markdown(f"**🤖 Answer:** {message['content']}")
                st.markdown("---")
    
    # Simple chat input
    with st.form("simple_chat_form", clear_on_submit=True):
        query = st.text_area(
            "Type your question:",
            placeholder="Ask anything about the uploaded document...",
            height=100
        )
        submitted = st.form_submit_button("Ask Question", use_container_width=True)
        
        if submitted and query:
            # Check if insurance type is selected
            if not st.session_state.get("selected_insurance_type"):
                st.error("Please select an insurance type from the dropdown in the sidebar first.")
            else:
                add_and_process_simple_query(query)
                st.rerun()

def add_and_process_simple_query(query: str):
    """Process simple query with type awareness"""
    current_type = st.session_state.get("current_insurance_type")
    if current_type:
        process_type_specific_query(query, current_type)
    else:
        st.error("Please select an insurance type first")

def process_simple_underwriter_query(query: str):
    """Simple query processing with type context"""
    current_type = st.session_state.get("current_insurance_type")
    if current_type:
        process_type_specific_query(query, current_type)
    else:
        response = "Please select an insurance type from the sidebar."
        st.session_state.underwriter_chat_history.append({"role": "assistant", "content": response})

# Initialize session state for insurance type selection
if "selected_insurance_type" not in st.session_state:
    st.session_state.selected_insurance_type = None

def extract_key_points(analysis_text: str) -> str:
    """Extract key points from analysis for summary view"""
    text = str(analysis_text)
    
    # Look for key sections
    key_points = []
    
    if "RECOMMENDATION:" in text:
        decision_match = text.split("RECOMMENDATION:")[1].split("\n")[0] if "RECOMMENDATION:" in text else ""
        if decision_match:
            key_points.append(f"**Decision:** {decision_match.strip()}")
    
    if "RISK SCORE:" in text:
        risk_match = text.split("RISK SCORE:")[1].split("\n")[0] if "RISK SCORE:" in text else ""
        if risk_match:
            key_points.append(f"**Risk Assessment:** {risk_match.strip()}")
    
    if "PREMIUM IMPACT:" in text:
        premium_match = text.split("PREMIUM IMPACT:")[1].split("\n")[0] if "PREMIUM IMPACT:" in text else ""
        if premium_match:
            key_points.append(f"**Premium Impact:** {premium_match.strip()}")
    
    if not key_points:
        # Fallback to first few lines
        lines = text.split('\n')[:3]
        key_points = [line.strip() for line in lines if line.strip()]
    
    return "\n".join([f"• {point}" for point in key_points[:5]])

def render_enhanced_underwriter_chat():
    """Enhanced underwriter chat with better UX"""
    st.markdown("### 💬 AI Underwriter Assistant")
    
    # Enhanced quick analysis buttons
    if st.session_state.uploaded_documents and st.session_state.analysis_results:
        latest_file = st.session_state.uploaded_documents[-1]
        if latest_file.name in st.session_state.analysis_results:
            doc_type = st.session_state.analysis_results[latest_file.name].get("document_type", "general")
            
            # Create a nice info box
            st.info(f"🔍 **Analyzing:** {latest_file.name} ({doc_type.title()} Insurance)")
            
            # Enhanced quick questions with emojis and descriptions
            st.markdown("**🚀 Quick Analysis Options:**")
            
            if doc_type == 'commercial':
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("💰 Financial Health", use_container_width=True, help="Analyze profit margins, revenue trends, and financial stability"):
                        query = "What is the financial performance including profit margins, revenue trends, and financial stability?"
                        add_and_process_query(query)
                    
                    if st.button("⚠️ Risk Assessment", use_container_width=True, help="Identify all risk factors and exposures"):
                        query = "What are all the risk factors, exposures, and potential threats identified?"
                        add_and_process_query(query)
                
                with col2:
                    if st.button("📋 Document Review", use_container_width=True, help="Check document completeness and compliance"):
                        query = "What documents are submitted vs missing? What compliance issues exist?"
                        add_and_process_query(query)
                    
                    if st.button("🎯 Final Decision", use_container_width=True, help="Get the underwriting recommendation"):
                        query = "What is your final underwriting decision and detailed rationale?"
                        add_and_process_query(query)
            
            elif doc_type == 'health':
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🏥 Medical Profile", use_container_width=True, help="Analyze health conditions and medical history"):
                        query = "What medical conditions, medications, and health risks are identified?"
                        add_and_process_query(query)
                    
                    if st.button("📊 Risk Scoring", use_container_width=True, help="Calculate and explain risk score"):
                        query = "What is the calculated health risk score and how was it determined?"
                        add_and_process_query(query)
                
                with col2:
                    if st.button("👨‍⚕️ Family History", use_container_width=True, help="Review family medical history impact"):
                        query = "What family medical history factors affect the risk assessment?"
                        add_and_process_query(query)
                    
                    if st.button("💊 Medication Impact", use_container_width=True, help="Assess medication-related risks"):
                        query = "What medications is the applicant taking and what are the underwriting implications?"
                        add_and_process_query(query)
            
            else:
                # Generic enhanced options
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("📊 Key Insights", use_container_width=True):
                        query = "What are the most important findings and insights?"
                        add_and_process_query(query)
                with col2:
                    if st.button("⚠️ Risk Analysis", use_container_width=True):
                        query = "What are the primary risk factors and concerns?"
                        add_and_process_query(query)
                with col3:
                    if st.button("✅ Recommendation", use_container_width=True):
                        query = "What is your underwriting recommendation?"
                        add_and_process_query(query)
    
    # Enhanced chat history display
    if st.session_state.underwriter_chat_history:
        st.markdown("### 📚 Analysis History")
        
        # Show recent conversations with better formatting
        for i, message in enumerate(st.session_state.underwriter_chat_history[-4:], 1):
            if message['role'] == 'user':
                st.markdown(f"""
                <div style="background: linear-gradient(90deg, #e3f2fd, #bbdefb); padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #2196f3;">
                    <strong>🤔 Your Question #{i}:</strong><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: linear-gradient(90deg, #f3e5f5, #e1bee7); padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #9c27b0;">
                    <strong>🤖 AI Analysis #{i}:</strong><br>
                </div>
                """, unsafe_allow_html=True)
                
                # Display the actual response with proper formatting
                st.markdown(message['content'])
    
    # Enhanced chat input with better styling
    st.markdown("### ✍️ Ask Your Question")
    
    with st.form("underwriter_chat_form", clear_on_submit=True):
        # Create columns for better layout
        col1, col2 = st.columns([4, 1])
        
        with col1:
            query = st.text_area(
                "Type your question about the document:",
                placeholder="Examples:\n• What is the exact profit margin?\n• Are there any red flags in this application?\n• What premium adjustment would you recommend?\n• How does this compare to industry standards?",
                height=120
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
            submitted = st.form_submit_button("🔍 Analyze", use_container_width=True, type="primary")
            
            if st.form_submit_button("💡 Get Suggestions", use_container_width=True):
                show_query_suggestions()
        
        if submitted and query:
            add_and_process_query(query)
            st.rerun()

def add_and_process_query(query: str):
    """Helper function to add query and process it"""
    st.session_state.underwriter_chat_history.append({"role": "user", "content": query})
    process_underwriter_query(query)

def show_query_suggestions():
    """Show helpful query suggestions"""
    suggestions = [
        "What is the applicant's risk profile?",
        "What documents are missing?", 
        "What is your underwriting decision?",
        "What are the key risk factors?",
        "What premium adjustment is needed?",
        "Are there any red flags?",
        "How does this compare to standards?",
        "What are the next steps?"
    ]
    
    st.info("💡 **Suggested Questions:**\n" + "\n".join([f"• {s}" for s in suggestions]))

def analyze_document(uploaded_file):
    # from src.underwriting_vectordb import enhanced_analyze_document
    enhanced_analyze_document(uploaded_file, st.session_state.selected_insurance_type)

def render_underwriter_chat():
    """Enhanced underwriter chat interface with better quick actions"""
    st.markdown("### 💬 Underwriter Assistant")
    
    # Enhanced quick analysis buttons with document-specific queries
    if st.session_state.uploaded_documents and st.session_state.analysis_results:
        latest_file = st.session_state.uploaded_documents[-1]
        if latest_file.name in st.session_state.analysis_results:
            doc_type = st.session_state.analysis_results[latest_file.name].get("document_type", "general")
            
            st.markdown(f"**Quick Analysis for {doc_type.title()} Insurance:**")
            
            # Dynamic quick questions based on document type
            if doc_type == 'commercial':
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("💰 Financial Analysis", use_container_width=True):
                        query = "What is the financial performance and profit margins of this business?"
                        st.session_state.underwriter_chat_history.append({"role": "user", "content": query})
                        process_underwriter_query(query)
                        st.rerun()
                with col2:
                    if st.button("⚠️ Risk Factors", use_container_width=True):
                        query = "What are the key risk factors and exposures for this commercial insurance application?"
                        st.session_state.underwriter_chat_history.append({"role": "user", "content": query})
                        process_underwriter_query(query)
                        st.rerun()
                with col3:
                    if st.button("📋 Missing Documents", use_container_width=True):
                        query = "What documents are missing from this commercial insurance application?"
                        st.session_state.underwriter_chat_history.append({"role": "user", "content": query})
                        process_underwriter_query(query)
                        st.rerun()
            
            elif doc_type == 'health':
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("🏥 Medical Assessment", use_container_width=True):
                        query = "What medical conditions and health risks are identified?"
                        st.session_state.underwriter_chat_history.append({"role": "user", "content": query})
                        process_underwriter_query(query)
                        st.rerun()
                with col2:
                    if st.button("📊 Risk Score", use_container_width=True):
                        query = "What is the calculated risk score and how was it determined?"
                        st.session_state.underwriter_chat_history.append({"role": "user", "content": query})
                        process_underwriter_query(query)
                        st.rerun()
                with col3:
                    if st.button("💊 Medications", use_container_width=True):
                        query = "What medications is the applicant taking and what are the implications?"
                        st.session_state.underwriter_chat_history.append({"role": "user", "content": query})
                        process_underwriter_query(query)
                        st.rerun()
            
            else:
                # Generic quick actions for other document types
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("📊 Key Findings", use_container_width=True):
                        query = "What are the key findings from this insurance application?"
                        st.session_state.underwriter_chat_history.append({"role": "user", "content": query})
                        process_underwriter_query(query)
                        st.rerun()
                with col2:
                    if st.button("⚠️ Risk Assessment", use_container_width=True):
                        query = "What are the main risk factors identified?"
                        st.session_state.underwriter_chat_history.append({"role": "user", "content": query})
                        process_underwriter_query(query)
                        st.rerun()
                with col3:
                    if st.button("✅ Decision Summary", use_container_width=True):
                        query = "What is your underwriting recommendation and rationale?"
                        st.session_state.underwriter_chat_history.append({"role": "user", "content": query})
                        process_underwriter_query(query)
                        st.rerun()
    
    # Show chat history first
    if st.session_state.underwriter_chat_history:
        st.markdown("**Analysis History:**")
        for message in st.session_state.underwriter_chat_history[-6:]:
            if message['role'] == 'user':
                st.info(f"**Query:** {message['content']}")
            else:
                # Display assistant responses with proper formatting
                st.success(message['content'])
    
    # Chat input with form to handle clearing properly
    with st.form("underwriter_chat_form", clear_on_submit=True):
        query = st.text_area(
            "Ask specific questions about the uploaded document:",
            placeholder="Examples:\n- What is the profit margin?\n- What are the risk factors?\n- What documents are missing?\n- What is your underwriting decision?",
            height=100
        )
        submitted = st.form_submit_button("Analyze 🔍", use_container_width=True)
        
        if submitted and query:
            st.session_state.underwriter_chat_history.append({"role": "user", "content": query})
            process_underwriter_query(query)
            st.rerun()

def process_underwriter_query(query: str):
    # from src.underwriting_vectordb import enhanced_process_underwriter_query
    enhanced_process_underwriter_query(query)

# Also update the simple detection function for consistency
def detect_document_type_simple(content: str) -> str:
    """Enhanced document type detection for Streamlit"""
    content_lower = content.lower()
    
    # Score-based detection
    scores = {
        'commercial': 0,
        'health': 0,
        'life': 0,
        'motor': 0
    }
    
    # Commercial keywords
    if any(term in content_lower for term in ['commercial insurance', 'business', 'company', 'turnover', 'gst', 'profit margin']):
        scores['commercial'] += 3
    if any(term in content_lower for term in ['incorporation', 'audited financial', 'employees', 'office area']):
        scores['commercial'] += 2
    
    # Health keywords  
    if any(term in content_lower for term in ['health insurance', 'medical report', 'blood pressure', 'bmi', 'hypertension']):
        scores['health'] += 3
    if any(term in content_lower for term in ['medical test', 'cholesterol', 'family history', 'pre-existing', 'medication']):
        scores['health'] += 2
    
    # Life keywords
    if any(term in content_lower for term in ['life insurance', 'nominee', 'death benefit', 'sum assured']):
        scores['life'] += 3
    
    # Motor keywords
    if any(term in content_lower for term in ['motor', 'vehicle', 'driving license', 'rc book']):
        scores['motor'] += 3
    
    # Return type with highest score
    max_score_type = max(scores.items(), key=lambda x: x[1])
    return max_score_type[0] if max_score_type[1] > 0 else 'general'

def render_sales_dashboard():
    pass

def render_welcome_screen():
    """Welcome screen when no role is selected"""
    st.markdown("### 👋 Welcome to Smart Insurance Assistant")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="policy-card">
            <h3>🏥 Customer Portal</h3>
            <p><strong>For Policyholders</strong></p>
            <ul>
                <li>View all your policies</li>
                <li>Ask questions about coverage</li>
                <li>Check renewal dates</li>
                <li>Get instant answers</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="policy-card">
            <h3>🔍 Underwriter Portal</h3>
            <p><strong>For Risk Assessment</strong></p>
            <ul>
                <li>Upload policy documents</li>
                <li>AI-powered analysis</li>
                <li>Risk scoring</li>
                <li>Compliance checking</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="policy-card">
            <h3>💼 Sales Portal</h3>
            <p><strong>For Sales Agents</strong></p>
            <ul>
                <li>Lead management</li>
                <li>Quote generation</li>
                <li>Customer outreach</li>
                <li>Pipeline tracking</li>
            </ul>
            <small><em>Coming soon...</em></small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.info("👆 Select your role from the sidebar to get started")

def initialize_underwriting_vectordb():
    """Initialize vector DB for Streamlit session"""
    if "underwriting_vectordb" not in st.session_state:
        st.session_state.underwriting_vectordb = UnderwritingVectorDB()
    return st.session_state.underwriting_vectordb


def enhanced_analyze_document(uploaded_file, doc_type: str):
    """Enhanced document analysis with vector DB support for Streamlit"""
    import streamlit as st
    from PyPDF2 import PdfReader
    import io
    
    try:
        # Extract file content
        if uploaded_file.type == "text/plain":
            content = str(uploaded_file.read(), "utf-8")
        elif uploaded_file.type == "application/pdf":
            pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
            content = "\n".join([p.extract_text() or "" for p in pdf_reader.pages])
        else:
            st.error(f"Unsupported file type: {uploaded_file.type}")
            return
        
        # Initialize vector DB
        vectordb = initialize_underwriting_vectordb()
        
        with st.spinner(f"🤖 Indexing and analyzing {doc_type} insurance document..."):
            # Run enhanced analysis
            result = run_enhanced_underwriter_flow(
                doc_type, content, uploaded_file.name, vectordb
            )
            
            # Store results
            st.session_state.analysis_results[uploaded_file.name] = {
                "content": content,
                "analysis": str(result),
                "document_type": doc_type,
                "timestamp": datetime.now().isoformat(),
                "vectordb_indexed": True
            }
            
            # Show statistics
            stats = vectordb.get_collection_stats()
            st.success(f"""
            ✅ Analysis complete for {uploaded_file.name}
            📊 Vector DB Stats: {stats['total_chunks']} chunks from {stats['unique_documents']} documents
            """)
    
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")


def enhanced_process_underwriter_query(query: str):
    """Enhanced query processing with vector DB for Streamlit"""
    import streamlit as st
    
    if "underwriting_vectordb" not in st.session_state:
        response = "Please upload and index a document first."
        st.session_state.underwriter_chat_history.append({"role": "assistant", "content": response})
        return
    
    vectordb = st.session_state.underwriting_vectordb
    doc_type = st.session_state.get("selected_insurance_type")
    
    try:
        with st.spinner("🔍 Searching indexed documents..."):
            response = run_enhanced_underwriter_chat(query, vectordb, doc_type)
            formatted_response = f"**AI Analysis:**\n\n{str(response)}"
        
        st.session_state.underwriter_chat_history.append({"role": "assistant", "content": formatted_response})
        
    except Exception as e:
        error_response = f"Error: {str(e)}"
        st.session_state.underwriter_chat_history.append({"role": "assistant", "content": error_response})

if __name__ == "__main__":
    main()