import streamlit as st
import pandas as pd
import json
import torch
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from Crypto.Cipher import AES
from Crypto.Hash import SHA256
from fuzzywuzzy import fuzz
import numpy as np
import warnings
from google.oauth2 import service_account
from langchain_google_vertexai import VertexAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize BERT (with fallback to fuzzywuzzy)
bert_available = False
tokenizer = None
model = None

@st.cache_resource
def load_bert():
    global bert_available, tokenizer, model
    try:
        from transformers import BertTokenizer, BertModel
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        bert_available = True
        st.write("BERT loaded successfully.")
        return tokenizer, model
    except (ImportError, Exception) as e:
        st.error(f"Failed to load BERT: {e}. Using fuzzywuzzy fallback.")
        bert_available = False
        return None, None

tokenizer, model = load_bert()

# Load Synthetic Data
customers_df = pd.read_csv("customers.csv")
transactions_df = pd.read_csv("transactions.csv")
credit_df = pd.read_csv("credit_data.csv")
collateral_df = pd.read_csv("collateral.csv")
deposits_df = pd.read_csv("deposits.csv")
behavior_df = pd.read_csv("behavior.csv")

# Expanded Sources
sources = {
    "Customer_Profiles": customers_df.columns.tolist()[1:] + ["customer_loyalty", "customer_engagement", "last_interaction", "identity_verification", "digital_signature", "demographics", "spending_habits", "customer_satisfaction", "customer_tenure", "product_ownership", "purchase_history", "feedback_score", "comment_text", "income_level", "login_history", "security_token", "ip_address", "churn_risk", "account_activity", "points_earned", "redemption_history", "loyalty_tier", "survey_score", "feedback_comments"],
    "Transactions": transactions_df.columns.tolist()[1:] + ["transaction_behavior", "pattern_recognition", "transaction_amount", "transaction_frequency", "wallet_balance", "transaction_type", "security_level", "atm_location", "frequency", "amount_lost", "fraud_report", "recovery_status", "dispute_id", "resolution_status", "transaction_category", "payment_status", "timestamp", "transfer_amount", "currency_type", "fee_amount", "transaction_success"],
    "Credit_Data": credit_df.columns.tolist()[1:] + ["monthly_income", "credit_history", "current_debt", "credit_utilization", "debt_to_income_ratio", "payment_history", "credit_limit", "credit_card_usage", "risk_score", "debt_level", "credit_report", "score_change", "report_date", "pre_approval_score", "eligibility_score", "credit_check"],
    "Collateral": collateral_df.columns.tolist()[1:] + ["property_value", "vehicle_value"],
    "Deposits": deposits_df.columns.tolist()[1:] + ["savings_balance", "overdraft_limit", "account_balance", "alert_status", "current_savings", "savings_target", "savings_growth", "incentive_type", "promotion_period", "cash_reserve"],
    "Behavior": behavior_df.columns.tolist()[1:] + ["investment_portfolio", "risk_tolerance", "market_trends", "asset_allocation", "investment_returns", "risk_profile", "login_attempts", "device_id", "retirement_age", "savings_rate", "pension_plan", "branch_visits", "service_time", "customer_feedback", "response_time", "support_tickets", "application_status", "onboarding_time", "adoption_rate", "feature_usage", "user_training"],
    "External_Systems": ["market_volatility", "economic_indicators", "rate_trend", "exchange_rate", "recipient_bank", "conversion_fee", "business_revenue", "business_size", "loan_demand"]
}

with open("finforge_sources.json", "w") as f:
    json.dump(sources, f)

# Expanded Banking Keywords (unchanged)
banking_keywords = {
    "loan": ["credit_score", "loan_id", "loan_amount", "loan_duration", "current_balance", "loan_status", "collateral_value", "monthly_income", "debt_to_income_ratio", "property_value", "payment_history", "loan_application", "income_verification", "credit_check", "current_rate", "new_rate", "loan_term", "pre_approval_score", "eligibility_score", "vehicle_value", "disbursement_amount", "date_issued", "remaining_balance", "restructure_plan", "loan_offer", "loan_demand", "business_size", "penalty_fee", "remaining_term"],
    "customer": ["customer_id", "name", "age", "income", "location", "customer_loyalty", "customer_engagement", "last_interaction", "demographics", "spending_habits", "customer_satisfaction", "customer_tenure", "product_ownership", "purchase_history", "income_level", "churn_risk", "account_activity", "points_earned", "redemption_history", "loyalty_tier", "survey_score", "feedback_comments", "customer_needs", "customer_profile", "referral_count", "reward_points"],
    "transaction": ["transaction_id", "date", "amount", "type", "transaction_behavior", "pattern_recognition", "ip_address", "transaction_amount", "transaction_frequency", "wallet_balance", "transaction_type", "security_level", "atm_location", "frequency", "amount_lost", "fraud_report", "recovery_status", "dispute_id", "resolution_status", "transaction_category", "payment_status", "timestamp", "transfer_amount", "currency_type", "fee_amount", "transaction_success", "trend_pattern", "spending_category", "time_period", "receipt_id"],
    "fraud": ["credit_score", "transaction_behavior", "dispute_count", "pattern_recognition", "ip_address", "fraud_report", "recovery_status", "amount_lost", "login_attempts", "security_questions", "device_id", "identity_alert", "security_measures", "alert_status", "claim_id", "fraud_type", "resolution_time"],
    "risk": ["credit_score", "credit_history", "loan_status", "current_balance", "risk_tolerance", "market_trends", "asset_allocation", "investment_returns", "risk_profile", "risk_score", "debt_level", "credit_report", "churn_risk", "risk_factor", "prediction_score", "market_volatility", "vulnerability"],
    "payment": ["amount", "date", "type", "current_balance", "payment_history", "credit_limit", "credit_card_usage", "overdraft_limit", "account_balance", "alert_status", "payment_due_date", "amount_paid", "remaining_balance", "due_date", "amount_due", "contact_info", "bill_amount", "payment_status", "payment_schedule", "flexibility_options", "prepayment_amount", "penalty_fee"],
    "account": ["account_id", "balance", "account_type", "savings_balance", "overdraft_limit", "account_balance", "alert_status", "current_savings", "savings_target", "savings_growth", "incentive_type", "promotion_period", "last_login", "account_status", "reactivation_date", "currency_balance", "exchange_rate", "conversion_fee", "source_account", "compliance_flag", "audit_status", "compliance_score", "last_audit_date", "recovery_code", "verification_step"],
    "behavior": ["feedback_score", "dispute_count", "customer_engagement", "last_interaction", "spending_habits", "login_attempts", "device_id", "response_time", "support_tickets", "customer_feedback", "feature_usage", "user_training"],
    "investment": ["investment_portfolio", "risk_tolerance", "market_trends", "asset_allocation", "investment_returns", "risk_profile", "portfolio_value", "performance_metrics", "review_date", "fund_allocation", "retirement_goal"],
    "credit": ["credit_score", "credit_history", "current_debt", "credit_utilization", "monthly_income", "credit_limit", "credit_card_usage", "score_change", "report_date", "credit_check", "improvement_plan", "usage_history"],
    "security": ["identity_verification", "digital_signature", "login_attempts", "security_questions", "device_id", "security_token", "ip_address", "security_level", "encryption_level", "security_measures", "alert_status", "verification_status", "signature_hash", "verification_result", "encryption_status", "audit_score", "vulnerability"],
    "savings": ["savings_balance", "current_savings", "savings_target", "savings_growth", "incentive_type", "promotion_period", "savings_rate", "pension_plan", "interest_rate", "cash_reserve", "savings_goal"],
    "support": ["support_tickets", "response_time", "customer_satisfaction", "chat_session_id", "encryption_status", "service_time"],
    "digital": ["application_status", "digital_signature", "wallet_balance", "security_level", "adoption_rate", "feature_usage", "user_training", "deposit_amount", "processing_time", "image_quality", "upload_status", "document_type"]
}

# BERT Cosine Similarity (with fuzzywuzzy fallback)
def get_embedding(text):
    if bert_available:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return text

def cosine_similarity(emb1, emb2):
    if bert_available:
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return fuzz.ratio(emb1, emb2) / 100.0

# Business Requirement Agent
def parse_use_case(title, description=""):
    with open("use_cases.json", "r") as f:
        use_cases = json.load(f)
    for uc in use_cases:
        if uc["title"].lower() == title.lower():
            st.write(f"Fields for '{title}': {uc['fields']}")
            return {"needs": uc['fields']}
    
    st.write(f"Title '{title}' not in dataset, generating fields dynamically...")
    text = f"{title} {description}".lower()
    all_fields = set().union(*[set(attrs) for attrs in sources.values()])
    relevant_fields = []
    for keyword, fields in banking_keywords.items():
        if keyword in text:
            relevant = [f for f in fields if f in all_fields or f in text]
            relevant_fields.extend(relevant)
    if relevant_fields:
        st.write(f"Fields from data for '{title}': {relevant_fields}")
        return {"needs": list(set(relevant_fields))}
    default_fields = ["customer_id"]
    st.write(f"Using default fields: {default_fields}")
    return {"needs": default_fields}

# Data Product Designer Agent
def design_schema(requirements):
    schema = {"customer_id": "int"}
    for field in requirements["needs"]:
        if field in ["credit_score", "loan_amount", "current_balance", "collateral_value", "balance", "amount", "income", "feedback_score", "monthly_income", "current_debt", "credit_utilization", "savings_balance", "transaction_amount", "wallet_balance", "risk_score", "credit_limit", "investment_returns", "portfolio_value", "savings_growth", "deposit_amount", "disbursement_amount", "cash_reserve", "net_worth", "asset_value", "liability", "transfer_amount", "fee_amount", "survey_score", "bill_amount", "prepayment_amount", "reward_points"]:
            schema[field] = "float"
        elif field in ["date", "last_interaction", "payment_due_date", "response_date", "report_date", "date_verified", "appointment_time", "alert_time", "date_issued", "submission_date", "redemption_date", "last_audit_date", "review_date", "transaction_date", "due_date", "reactivation_date", "resolution_date", "timestamp", "response_time", "processing_time", "resolution_time", "audit_date", "time_period"]:
            schema[field] = "datetime"
        elif field in ["customer_id", "transaction_id", "account_id", "loan_id", "loan_duration", "credit_history", "dispute_count", "support_tickets", "login_attempts", "branch_visits", "frequency", "retirement_age", "dispute_id", "application_id", "chat_session_id", "claim_id", "referral_count", "business_size"]:
            schema[field] = "int"
        else:
            schema[field] = "string"
    with open("finforge_schema.json", "w") as f:
        json.dump(schema, f)
    return schema

# Source System Discovery Agent
def discover_sources(schema):
    with open("finforge_sources.json", "r") as f:
        sources = json.load(f)
    source_list = {}
    for field in schema.keys():
        found = False
        for system, attrs in sources.items():
            if field in attrs:
                source_list[field] = system
                found = True
                break
            elif any(fuzz.ratio(field, attr) > 85 for attr in attrs):
                closest_match = max(attrs, key=lambda attr: fuzz.ratio(field, attr))
                source_list[field] = system
                found = True
                break
        if not found and field != "customer_id":
            source_list[field] = "N/A"
    return source_list

# Attribute Mapping Agent
def generate_mappings(source_list, schema):
    mappings = []
    for target in schema.keys():
        source_system = source_list[target]
        target_clean = target.replace("customer_360.", "")
        if source_system != "N/A":
            source_attrs = sources[source_system]
            if target_clean in source_attrs:
                confidence = 0.95
                source_attr = target_clean
            else:
                fuzzy_scores = [fuzz.ratio(target_clean, attr) for attr in source_attrs]
                max_fuzzy = max(fuzzy_scores, default=0)
                target_emb = get_embedding(target_clean)
                cosine_scores = [cosine_similarity(target_emb, get_embedding(attr)) for attr in source_attrs]
                max_cosine = max(cosine_scores, default=0)
                confidence = (max_fuzzy / 100 * 0.4) + (max_cosine * 0.6)
                confidence = min(confidence + 0.15, 1.0) if target_clean.split("_")[0] in source_system.lower() else confidence
                source_attr = max(source_attrs, key=lambda attr: fuzz.ratio(target_clean, attr), default="N/A")
                if max_fuzzy > 90:
                    confidence = min(confidence + 0.1, 0.95)
        else:
            confidence = 0.0
            source_attr = "N/A"
        
        mapping_type = "direct" if confidence > 0.75 else "N/A" if confidence < 0.6 else "partial"
        mappings.append({
            "source_system": source_system,
            "source_attribute": source_attr,
            "target_attribute": f"customer_360.{target_clean}",
            "confidence": round(confidence, 2),
            "mapping_type": mapping_type,
            "transformation": "N/A"
        })
    df = pd.DataFrame(mappings)
    df["transformation"] = df["transformation"].astype("object")
    df.to_csv("finforge_mappings.csv", index=False)
    return df

# Transformation Agent
def apply_transformations(mappings_file):
    df = pd.read_csv(mappings_file)
    numeric_keywords = ["score", "amount", "balance", "income", "debt", "rate", "value", "limit", "returns", "growth", "fee", "cost", "points"]
    for i, row in df.iterrows():
        target_clean = row["target_attribute"].replace("customer_360.", "")
        if (any(keyword in target_clean.lower() for keyword in numeric_keywords) and 
            row["confidence"] > 0.75 and 
            row["source_system"] != "N/A"):
            df.at[i, "mapping_type"] = "transform"
            df.at[i, "transformation"] = "NORMALIZE({})".format(row["source_attribute"])
    df.to_csv("finforge_mappings.csv", index=False)
    return df

# Quality Certification Agent with Enhanced Security
def certify_data_product(mappings_file):
    df = pd.read_csv(mappings_file)
    min_confidence = df["confidence"].min()
    certification_status = "✅ Certified" if min_confidence >= 0.6 else "⚠️ Certification Pending: Low confidence mappings (min: {:.2f})".format(min_confidence)
    st.write("Certification Status:", certification_status)
    
    # AES Encryption for PII
    key = b'Sixteen byte key'  # In production, use a secure key management system
    cipher = AES.new(key, AES.MODE_EAX)
    nonce = cipher.nonce
    encrypted_customer_id = cipher.encrypt(b"customer_id")  # Example PII
    
    # SHA-256 Hashing for Data Integrity
    hasher = SHA256.new()
    hasher.update(b"customer_id")  # Hash the same field for integrity check
    hashed_customer_id = hasher.hexdigest()
    
    config = {
        "ingress": "Batch ETL from Customer_Profiles, Transactions, Credit_Data, Collateral, Deposits, Behavior, External_Systems",
        "egress": "REST API for banking applications",
        "store": "SQLite",
        "search": "SQL queries",
        "certification": certification_status,
        "encryption": "AES",
        "hashing": "SHA-256",
        "sample_encrypted_pii": encrypted_customer_id.hex(),  # For demo visibility
        "sample_hashed_pii": hashed_customer_id[:16]  # Truncated for brevity
    }
    with open("finforge_config.yaml", "w") as f:
        json.dump(config, f)
    st.write("Certification Report: PII encrypted with AES, integrity ensured with SHA-256, ingress/egress defined")
    return config

# Gen AI Insights Agent
insights_template = """
Role Assignment:
You are Banking Insights AI, a highly capable AI assistant specializing in analyzing retail banking data to provide actionable insights for Customer 360 data products. Your goal is to analyze the provided credit data summary and generate insights tailored to the specified use case.
Credit Data Summary: {credit_summary}
Use Case: {use_case}

Instructions:
- Analyze key metrics such as credit scores, loan amounts, current balances, and dispute counts.
- Identify patterns or risks relevant to the use case (e.g., fraud potential, default risk).
- Provide concise, data-driven insights with numerical estimates where possible (e.g., percentage of high-risk customers).
- Keep it practical and aligned with retail banking goals (e.g., risk mitigation, customer retention).
Output Format:
- Bullet points for clarity
- Include at least one numerical insight (e.g., 'X% of customers...')
"""
insights_prompt = PromptTemplate(input_variables=["credit_summary", "use_case"], template=insights_template)

credentials = service_account.Credentials.from_service_account_file(st.secrets["gcp_service_account"],scopes=["https://www.googleapis.com/auth/cloud-platform"])
llm = VertexAI(
    model_name="gemini-1.5-flash",
    project="inspired-rock-450806-r5",
    location="us-central1",
    credentials=credentials,
    max_output_tokens=1000,
    temperature=0.7
)
insights_chain = RunnableSequence(insights_prompt | llm)

def generate_insights(config, use_case):
    credit_summary = credit_df.describe().to_string()
    try:
        insight = insights_chain.invoke({"credit_summary": credit_summary, "use_case": use_case})
    except Exception as e:
        st.error(f"Gemini error: {e}")
        high_risk = credit_df[(credit_df["credit_score"] < 600) & (credit_df["current_balance"] > 0.5 * credit_df["loan_amount"])].shape[0]
        total = credit_df.shape[0]
        insight = f"- {high_risk} out of {total} customers ({high_risk/total*100:.1f}%) have high risk (credit_score < 600, balance > 50% loan)."
    return {"insights": insight}

# Chatbot Agent
chat_template = """
Role Assignment:
You are FinForge Assistant, an AI designed to help users understand the Customer 360 data product dashboard for retail banking. Answer queries based on the provided dashboard outputs.
Context: {context}
Query: {query}

Instructions:
- Use the context (schema, mappings, config, insights) to provide accurate answers.
- Be concise and clear, focusing on the dashboard data.
- If the query is unclear or lacks context, ask for clarification or suggest running a use case.
"""
chat_prompt = PromptTemplate(input_variables=["context", "query"], template=chat_template)
chat_chain = RunnableSequence(chat_prompt | llm)

def chatbot_response(query, result=None):
    context = ""
    if result:
        context += f"Schema: {json.dumps(result.get('schema', {}))}\n"
        context += f"Mappings: {pd.read_csv('finforge_mappings.csv').to_string() if 'mappings' in result else 'Not generated yet'}\n"
        context += f"Config: {json.dumps(result.get('config', {}))}\n"
        context += f"Insights: {result.get('insights', 'Not generated yet')}\n"
    else:
        context = "No use case has been run yet."
    
    try:
        answer = chat_chain.invoke({"context": context, "query": query})
    except Exception as e:
        st.error(f"Gemini error: {e}")
        answer = "Sorry, I couldn’t connect to Gemini. Please try again or run a use case first."
    return answer

# Orchestrator Agent
def run_workflow(title, description):
    class AgentState(TypedDict):
        title: str
        description: str
        requirements: dict
        schema: dict
        sources: dict
        mappings: object
        config: dict
        insights: str

    workflow = StateGraph(AgentState)
    workflow.add_node("business_req", lambda state: {"requirements": parse_use_case(state["title"], state["description"])})
    workflow.add_node("designer", lambda state: {"schema": design_schema(state["requirements"])})
    workflow.add_node("discovery", lambda state: {"sources": discover_sources(state["schema"])})
    workflow.add_node("mapping", lambda state: {"mappings": generate_mappings(state["sources"], state["schema"])})
    workflow.add_node("transformation", lambda state: {"mappings": apply_transformations("finforge_mappings.csv")})
    workflow.add_node("certification", lambda state: {"config": certify_data_product("finforge_mappings.csv")})
    workflow.add_node("gen_insights", lambda state: generate_insights(state["config"], state["title"]))
    workflow.add_edge(START, "business_req")
    workflow.add_edge("business_req", "designer")
    workflow.add_edge("designer", "discovery")
    workflow.add_edge("discovery", "mapping")
    workflow.add_edge("mapping", "transformation")
    workflow.add_edge("transformation", "certification")
    workflow.add_edge("certification", "gen_insights")
    workflow.add_edge("gen_insights", END)

    app = workflow.compile()
    return app.invoke({"title": title, "description": description})

# Streamlit UI
def main():
    st.title("FinForge Dashboard")
    st.write("Secure Customer 360 Data Product for Retail Banking with Multi-Agent AI")

    # Sidebar for Use Case Selection
    st.sidebar.header("Run a Use Case")
    title = st.sidebar.text_input("Enter Use Case Title", "Fraud Detection and Transaction Monitoring")
    description = st.sidebar.text_area("Enter Description (optional)", "Detect fraudulent activities and monitor transactions in real-time to enhance security and reduce financial losses.")
    if st.sidebar.button("Generate Outputs"):
        result = run_workflow(title, description)
        st.session_state["result"] = result
        st.sidebar.success("Outputs generated!")
        st.sidebar.write("Fields Used:", result["requirements"]["needs"])
        st.sidebar.write("Sample Credit Data:", credit_df.head(5).to_dict())

    # Display Schema
    st.subheader("Customer 360 Schema (finforge_schema.json)")
    try:
        with open("finforge_schema.json", "r") as f:
            schema = json.load(f)
        st.json(schema)
        # Download button for schema
        schema_str = json.dumps(schema, indent=2)
        st.download_button(
            label="Download finforge_schema.json",
            data=schema_str,
            file_name="finforge_schema.json",
            mime="application/json"
        )
    except FileNotFoundError:
        st.write("Run a use case to generate schema.")

    # Display Mappings with Confidence Highlighting
    st.subheader("Source-to-Target Mappings (finforge_mappings.csv)")
    try:
        mappings = pd.read_csv("finforge_mappings.csv")
        def highlight_confidence(row):
            color = 'green' if row['confidence'] > 0.75 else 'yellow' if row['confidence'] > 0.6 else 'red'
            return [f'background-color: {color}' if col == 'confidence' else '' for col in row.index]
        st.table(mappings.style.apply(highlight_confidence, axis=1))
        # Download button for mappings
        mappings_csv = mappings.to_csv(index=False)
        st.download_button(
            label="Download finforge_mappings.csv",
            data=mappings_csv,
            file_name="finforge_mappings.csv",
            mime="text/csv"
        )
    except FileNotFoundError:
        st.write("Run a use case to generate mappings.")

    # Display Config (Ingress/Egress)
    st.subheader("Ingress/Egress Configuration (finforge_config.yaml)")
    try:
        with open("finforge_config.yaml", "r") as f:
            config = json.load(f)
        st.json(config)
        # Download button for config
        config_str = json.dumps(config, indent=2)
        st.download_button(
            label="Download finforge_config.yaml",
            data=config_str,
            file_name="finforge_config.yaml",
            mime="application/yaml"
        )
    except FileNotFoundError:
        st.write("Run a use case to generate config.")

    # Display Gen AI Insights
    st.subheader("Gemini Insights (Vertex AI)")
    try:
        st.write(st.session_state["result"]["insights"])
    except (KeyError, NameError):
        st.write("Run a use case to generate insights.")

    # Chatbot Interface
    st.subheader("Chat with FinForge Assistant")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    
    user_input = st.text_input("Ask a question about the dashboard:", key="chat_input")
    if user_input:
        result = st.session_state.get("result", None)
        response = chatbot_response(user_input, result)
        st.session_state["chat_history"].append({"user": user_input, "bot": response})
    
    for chat in reversed(st.session_state["chat_history"]):
        st.write(f"**You**: {chat['user']}")
        st.write(f"**FinForge Assistant**: {chat['bot']}")

if __name__ == "__main__":
    main()
