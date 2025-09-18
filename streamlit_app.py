#Javeria
import os
import streamlit as st
from qdrant_client import models
import qdrant_client
import xml.etree.ElementTree as ET
from tabulate import tabulate
import requests
from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
import json
import pandas as pd
load_dotenv()

llm = ChatOpenAI(model="gpt-4.1", temperature=0,api_key=os.getenv("OPENAI_API_KEY"))
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-large")


if "client" not in st.session_state:
    st.session_state.client = qdrant_client.QdrantClient(url=os.getenv("QDRANT_URL"),api_key=os.getenv("QDRANT_API_KEY"),port=None)

def detect_language(text: str) -> str:
    """Detect if text is Korean ('ko') or English ('en') based on Hangul ratio."""
    no_space = text.replace(" ", "")
    total_chars = len(no_space)
    korean_chars = sum(1 for ch in no_space if ord('가') <= ord(ch) <= ord('힣'))
    if total_chars > 0 and (korean_chars / total_chars) > 0.3:
        return 'ko'
    return 'en'

def xml_to_table(xml_data):
    """
    Parses XML data and returns a list of dictionaries, where each
    dictionary represents a row in the table.
    """
    root = ET.fromstring(xml_data)
    items = root.findall(".//item")

    table_data = []

    # Get all unique column names from the first item to ensure a consistent header
    if items:
        headers = [child.tag for child in items[0]]
        table_data.append(headers)

        for item in items:
            row = []
            for header in headers:
                # Find the element and get its text, or an empty string if not found
                element = item.find(header)
                row.append(element.text if element is not None else "")
            table_data.append(row)

    return table_data

def validate_steel_factory_data(accident_docs, query):
    """Validate and filter steel factory accidents specifically"""
    
    # Steel factory department names in Korean data
    steel_factory_keywords = ['제강공장', 'steel factory', 'steelmaking']
    
    # Extract steel factory specific accidents
    steel_accidents = []
    
    if hasattr(accident_docs, '__iter__') and not isinstance(accident_docs, str):
        for doc in accident_docs:
            if hasattr(doc, 'page_content'):
                content = doc.page_content.lower()
                if any(keyword in content for keyword in steel_factory_keywords):
                    steel_accidents.append(doc)
    
    # Count validation
    steel_count = len(steel_accidents)
    
    validation_note = f"""
    VALIDATION CHECK:
    - Steel factory accidents found: {steel_count}
    - Department filter applied: 제강공장 only
    - Excluded: 원자력공장, 단조공장, 터빈공장, etc.
    """
    
    return steel_accidents, validation_note

def anonymize_employee_names(text):
    """Anonymize employee names while maintaining data structure"""
    import re
    
    # Korean name patterns ( 2-3 characters typically)
    korean_names = re.findall(r'[가-힣]{2,3}(?=\s|$|,|\)|\()', text)
    
    # Create anonymization mapping
    name_mapping = {}
    for i, name in enumerate(set(korean_names)):
        # Create generic identifiers
        name_mapping[name] = f"Employee_{chr(65+i)}"  # Employee_A, Employee_B, etc.
    
    # Replace names
    anonymized_text = text
    for real_name, anon_name in name_mapping.items():
        anonymized_text = anonymized_text.replace(real_name, anon_name)
    
    return anonymized_text, name_mapping

def validate_data_quality(docs):
    """Check if retrieved documents contain actual accident data"""
    
    if not docs or len(docs) == 0:
        return False, "No documents retrieved"
    
    # Check for accident-related content
    accident_indicators = ['사고', '재해', 'accident', 'injury', '부상', 'incident']
    has_accident_data = False
    
    for doc in docs:
        if hasattr(doc, 'page_content'):
            content = doc.page_content.lower()
            if any(indicator in content for indicator in accident_indicators):
                has_accident_data = True
                break
    
    if not has_accident_data:
        return False, "Documents don't contain accident data"
    
    return True, f"Valid accident data found in {len(docs)} documents"

def validate_steel_plant_chemical_data(chemical_docs, query):
    """Validate and filter steel plant chemicals specifically"""
    
    # Steel plant section markers in Korean data
    steel_plant_keywords = ['■ Steel Plant', '■ 제강공장', 'steel plant', 'steelmaking']
    
    # Extract steel plant specific chemicals
    steel_chemicals = []
    
    if hasattr(chemical_docs, '__iter__') and not isinstance(chemical_docs, str):
        for doc in chemical_docs:
            if hasattr(doc, 'page_content'):
                content = doc.page_content
                # Check for steel plant section markers
                if any(keyword in content for keyword in steel_plant_keywords):
                    steel_chemicals.append(doc)
    
    # Count validation
    steel_chemical_count = len(steel_chemicals)
    
    validation_note = f"""
    CHEMICAL DATA VALIDATION:
    - Steel plant chemicals found: {steel_chemical_count}
    - Department filter applied: ■ Steel Plant only
    - Excluded: Nuclear plant, Forging plant, etc.
    """
    
    return steel_chemicals, validation_note

def validate_chemical_data_quality(docs):
    """Check if retrieved documents contain actual chemical data"""
    
    if not docs or len(docs) == 0:
        return False, "No chemical documents retrieved"
    
    # Check for chemical-related content
    chemical_indicators = ['CAS', 'Chemical', '화학', 'MSDS', 'Monthly Handling', '취급량']
    has_chemical_data = False
    
    for doc in docs:
        if hasattr(doc, 'page_content'):
            content = doc.page_content
            if any(indicator in content for indicator in chemical_indicators):
                has_chemical_data = True
                break
    
    if not has_chemical_data:
        return False, "Documents don't contain chemical data"
    
    return True, f"Valid chemical data found in {len(docs)} documents"

def extract_special_management_substances(chemical_docs):
    """Extract only chemicals marked as Special Management Substances"""
    
    special_substances = []
    
    for doc in chemical_docs:
        if hasattr(doc, 'page_content'):
            content = doc.page_content
            # Look for specific marking pattern
            if 'Special Management Substance: ○' in content or 'Special Management Substance": "○"' in content:
                special_substances.append(doc)
    
    return special_substances

@tool
def get_accident_records(query: str) -> str:
    """
    Retrieves accident records and safety incident information from company documents.
    Use this tool for questions about workplace accidents, safety incidents, injuries, or safety violations.
    Supports both English and Korean queries.
    Example queries: '사고 기록', 'accident records', '안전 사고', 'workplace injuries', '사고 보고서', 'safety incidents', 'accident analysis', ''.
    """
    qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="doosan-accidents-new",
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    port=None,
    timeout=120
    )
    retriever = qdrant.as_retriever(search_kwargs={"k": 30})
    retrieved_docs = retriever.invoke(query)
    print("retrieved_docs",retrieved_docs)
    return retrieved_docs


def accident_output(accident_docs, query):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    lang = detect_language(query)
    
    # Apply steel factory validation if relevant
    if 'steel factory' in query.lower() or '제강공장' in query:
        filtered_docs, validation_note = validate_steel_factory_data(accident_docs, query)
        accident_docs = filtered_docs
    else:
        validation_note = ""

    language_instruction = (
        "MANDATORY: Respond ONLY in Korean language."
        if lang == 'ko' else
        "MANDATORY: Respond ONLY in English language."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": (
                    f"{language_instruction}\n\n"
                    "CRITICAL INSTRUCTIONS FOR ACCURACY:\n"
                    "1. Count accidents EXACTLY from provided data only\n"
                    "2. Use Employee_A, Employee_B format instead of real names\n"
                    "3. State exact count: 'Based on X accidents in steel factory'\n"
                    "4. Filter departments precisely - only 제강공장 for steel factory queries\n"
                    "5. Verify each accident belongs to requested department\n"
                    f"\n{validation_note}\n\n"
                    f"Accident data to analyze:\n{accident_docs}\n\n"
                    "Response format:\n"
                    "- Start with: 'Based on [exact count] steel factory accidents...'\n"
                    "- Use Employee_A, Employee_B for worker identification\n"
                    "- Include specific dates and injury types\n"
                    "- Identify real patterns from this exact dataset\n"
                )
            },
            {"role": "user", "content": query},
        ],
    )
    
    # Anonymize the response
    raw_response = response.choices[0].message.content
    anonymized_response, name_mapping = anonymize_employee_names(raw_response)
    
    return anonymized_response

@tool
def get_chemical_usage(query: str) -> str:
    """
    Retrieves chemical usage and safety information from company documents.
    Use this tool for questions about chemical substances, chemical safety, MSDS, chemical handling, or chemical exposure.
    Supports both English and Korean queries.
    Example queries: '화학 물질 사용', 'chemical usage', '화학 안전', 'chemical safety', 'MSDS', '화학 물질 안전보건자료'.
    """
    # retrieved_docs = st.session_state.client.query_points(
    #     collection_name="doosan-chemical-openai",
    #     query=models.Document(
    #         text=query,
    #         model=embeddings  
    #     ),
    #     limit=5
    # )
    qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="doosan-chemical-openai",
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    port=None,
    timeout=120
    )
    retriever = qdrant.as_retriever(search_kwargs={"k": 30})
    retrieved_docs = retriever.invoke(query)
    print("retrieved_docs",retrieved_docs)

    # docs = [doc.payload['text'] for doc in retrieved_docs.points]
    # print("docs",docs)
    # return "\n\n".join(docs)
    return retrieved_docs


@tool
def get_risk_assessment(query: str) -> str:
    """
    Retrieves risk assessment and safety evaluation information from company documents.
    Use this tool for questions about risk analysis, safety assessments, hazard identification, risk levels, or safety measures.
    Supports both English and Korean queries.
    Example queries: '위험 평가', 'risk assessment', '안전 평가', 'safety evaluation', '위험 분석', 'hazard analysis', '위험도 평가', 'risk assessment table'.
    # """

    # retrieved_docs = st.session_state.client.query_points(
    #     collection_name="doosan-risk-new",
    #     query=models.Document(
    #         text=query,
    #         model=embeddings
    #     ),
    #     limit=5
    # )
    qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="doosan-risk-new",
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    port=None,
    timeout=120
    )
    retriever = qdrant.as_retriever(search_kwargs={"k": 30})
    retrieved_docs = retriever.invoke(query)
    print("retrieved_docs",retrieved_docs)
    # docs = [doc.payload['text'] for doc in retrieved_docs.points]
    return retrieved_docs

def risk_assessment_table_output(risk_assessment_docs, query):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    lang = detect_language(query)
    
    # Create structured prompt for JSON output
    json_instructions = f"""
MANDATORY: All field names and values MUST be in {'Korean' if lang == 'ko' else 'English'}.

You are a safety engineering expert specializing in industrial risk assessments. Your task is to analyze the provided risk assessment documents: {risk_assessment_docs}

Instructions:
- Always return the output in strict JSON format.
- Use the predefined field names exactly as listed below.
- If data cannot be found, set the field value to "N/A".
- If no structured data is available at all, return a JSON object with all fields set to "N/A".
- Do not include explanations, extra text, or markdown—only JSON.
- Provide whole table 
- Use {'Korean' if lang == 'ko' else 'English'} for all headers and values
- Extract only the information relevant to the user query
- Generate a comprehensive risk assessment with the specified format

Output Format:
[
  {{
    "Task/Process Name": "Specific work being performed or N/A",
    "Hazard Factor": "Description of the hazard",
    "Hazard Identification": "List of 3-5 specific hazards or N/A",
    "Current Frequency": "Low/Medium/High or 1-5 scale or N/A",
    "Current Severity": "Low/Medium/High or 1-4 scale or N/A",
    "Current Risk": "Low/Medium/High or Risk Score or N/A",
    "Current Risk Level": "Frequency (1-5) x Severity (1-4) = Risk Score or N/A",
    "Root Causes": "Underlying reasons for each hazard or N/A",
    "Current Measures": "Existing safety measures or N/A",
    "Control Measures": "Specific preventive/protective actions or N/A",
    "Reduction Measures": "Proposed reduction measures or N/A",
    "Improved Frequency": "Low/Medium/High or 1-5 scale or N/A",
    "Improved Severity": "Low/Medium/High or 1-4 scale or N/A",
    "Improved Risk": "Low/Medium/High or Risk Score or N/A",
    "Residual Risk After Controls": "New Frequency x New Severity = New Risk Score or N/A"
  }}
]"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.2,
            messages=[{"role": "system", "content": json_instructions},
                      {"role": "user", "content": query}],
            response_format={"type": "json_object"}
        )

        parsed_json = json.loads(response.choices[0].message.content)
        return parsed_json
        
    except (json.JSONDecodeError, KeyError, Exception) as e:
        print(f"Error in risk_assessment_output: {e}")
        # Return error structure if JSON parsing fails
        return [{
            "Task/Process Name": "Error parsing data",
            "Hazard Factor": "Error parsing data",
            "Hazard Identification": "N/A",
            "Current Frequency": "N/A",
            "Current Severity": "N/A",
            "Current Risk": "N/A",
            "Current Risk Level": "N/A",
            "Root Causes": "N/A",
            "Current Measures": "N/A",
            "Control Measures": "N/A",
            "Reduction Measures": "N/A",
            "Improved Frequency": "N/A",
            "Improved Severity": "N/A",
            "Improved Risk": "N/A",
            "Residual Risk After Controls": "N/A"
        }]

def risk_assessment_output(risk_assessment_docs, query):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    lang = detect_language(query)
    language_instruction = (
        "MANDATORY: Respond ONLY in Korean language for the entire output."
        if lang == 'ko' else
        "MANDATORY: Respond ONLY in English language for the entire output."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[{
            "role": "system",
            "content": (
                f"{language_instruction}\n\n"
                "You are a safety engineering expert specializing in industrial risk assessments. "
                f"From the provided {risk_assessment_docs}, extract only the information relevant to the user query. "
                "Then, generate a comprehensive risk assessment in the following exact format:\n\n"
                "- Task/Process Name: [specific work being performed]\n"
                "- Hazard Identification: [list 3–5 specific hazards]\n"
                "- Current Risk Level: [Frequency (1–5)] x [Severity (1–4)] = [Risk Score]\n"
                "- Root Causes: [underlying reasons for each hazard]\n"
                "- Control Measures: [specific preventive/protective actions]\n"
                "- Residual Risk After Controls: [new Frequency] x [new Severity] = [new Risk Score]\n\n"
                "Requirements:\n"
                "• Always cite relevant safety regulations (e.g., KOSHA, ISO 45001).\n"
                "• Include specific MSDS data when chemicals are involved.\n"
                "• Ensure outputs are technically accurate, concise, and actionable."
            )
        }, {"role": "user", "content": query}],
    )
    return response.choices[0].message.content

@tool
def get_chemical_details(query:str):
    """
    This tool should be used when user want to extract any information about chemicals
    Example query: 'What is cas no of oxygen', 'what is chemical ID of nitrogen'.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful chemical identifier assistant. Identify the chemical name only and return a single word. If no chemical name is available then return 'none'."
            },
            {"role": "user", "content": query},
        ],
    )
    chemical_name = response.choices[0].message.content.strip()
    if chemical_name=="none":
        return None
    else:
        url = "https://msds.kosha.or.kr/openapi/service/msdschem/chemlist"
        params = {'serviceKey': os.getenv("KOSHA_API_KEY"), 'searchWrd': chemical_name, 'searchCnd': '0', 'numOfRows': '10'}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.text
            table_data=xml_to_table(xml_data=data)
            return table_data
        else:
            return None

def chemical_output(table_data, query):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    lang = detect_language(query)
    
    # Apply steel plant validation if relevant
    if 'steel plant' in query.lower() or '제강공장' in query.lower() or 'steel factory' in query.lower():
        filtered_docs, validation_note = validate_steel_plant_chemical_data(table_data, query)
        table_data = filtered_docs
    else:
        validation_note = ""

    language_instruction = (
        "MANDATORY: Respond ONLY in Korean language for the entire output."
        if lang == 'ko' else
        "MANDATORY: Respond ONLY in English language for the entire output."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        messages=[{
            "role": "system",
            "content": (
                f"{language_instruction}\n\n"
                "CRITICAL INSTRUCTIONS FOR CHEMICAL DATA ACCURACY:\n"
                "1. Extract chemicals ONLY from ■ Steel Plant section\n"
                "2. Look for 'Special Management Substance: ○' markings\n"
                "3. State exact monthly handling amounts from data\n"
                "4. Include CAS numbers and chemical names exactly as shown\n"
                "5. Identify measurement targets marked with '측정대상: ○'\n"
                f"\n{validation_note}\n\n"
                f"Chemical data to analyze:\n{table_data}\n\n"
                "Response format:\n"
                "- Start with: 'Based on steel plant chemical data analysis...'\n"
                "- List actual chemicals with CAS numbers\n"
                "- Include real monthly handling amounts (kg/ℓ)\n"
                "- Identify Special Management Substances specifically\n"
                "- Reference actual usage purposes from data\n"
            )
        }, {"role": "user", "content": query}],
    )
    return response.choices[0].message.content


@tool
def get_regulations_data(query: str) -> str:
    """
    Retrieves regulations, compliance, and legal requirements information from company documents.
    Use this tool for questions about safety regulations, compliance requirements, legal standards, or regulatory procedures.
    Supports both English and Korean queries.
    Example queries: '규정', 'regulations', '법규', 'legal requirements', '준수 사항', 'compliance', '컴플라이언스', 'safety standards'.
    """
    # retrieved_docs = st.session_state.client.query_points(
    #     collection_name="rules-and-regulations-new",
    #     query=models.Document(
    #         text=query,
    #         model=embeddings
    #     ),
    #     limit=5
    # )
    # print("retrieved_docs",retrieved_docs.points)
    # docs = [doc.payload['text'] for doc in retrieved_docs.points]
    # print("docs",docs)
    qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="rules-and-regulations-new",
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    port=None,
    timeout=120
    )
    retriever = qdrant.as_retriever(search_kwargs={"k": 30})
    retrieved_docs = retriever.invoke(query)
    return retrieved_docs

@tool
def dynamic_risk_assessment(query: str) -> str:
    """
    Performs dynamic risk assessment using historical risk assessment data and provides comprehensive risk scoring.
    Use this tool for dynamic risk analysis, risk scoring, hazard assessment, or safety evaluation of new risks.
    Supports both English and Korean queries.
    Example queries: 'dynamic risk assessment','dynamic risk analysis','빈도×심각도', today risk analysis', 'risk scoring','위험도 점수', 'yesterday risk analysis', 'dynamic hazard analysis', 'safety evaluation', '위험도 평가', '동적 위험 평가', '안전 평가', '계산해주세요'.
    """
    # retrieved_docs = st.session_state.client.query_points(
    #     collection_name="doosan-risk-new",
    #     query=models.Document(
    #         text=query,
    #         model=embeddings
    #     ),  
    #     limit=5
    # )
    qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="doosan-risk-new",
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    port=None,
    timeout=120
    )
    retriever = qdrant.as_retriever(search_kwargs={"k": 30})
    retrieved_docs = retriever.invoke(query)
    print("retrieved_docs",retrieved_docs)
    return retrieved_docs

def regulations_output(regulations_docs,query):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    lang = detect_language(query)
    language_instruction = (
        "MANDATORY: Respond ONLY in Korean language for the entire output."
        if lang == 'ko' else
        "MANDATORY: Respond ONLY in English language for the entire output."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[{
            "role": "system",
            "content": (
                f"{language_instruction}\n\n"
                "You are a safety compliance expert familiar with Korean industrial safety regulations. "
                f"From the provided data {regulations_docs}, extract only the information relevant to the user query. "
                "Then, generate a comprehensive compliance assessment in the following exact format:\n\n"
                "COMPLIANCE ASSESSMENT OUTPUT:\n"
                "1. Applicable Regulations:\n"
                "   - Primary: [specific act/regulation with article numbers]\n"
                "   - Secondary: [related standards and codes]\n"
                "2. Mandatory Requirements:\n"
                "   - Documentation needed\n"
                "   - Permits/certifications required\n"
                "   - Training prerequisites\n"
                "   - Safety equipment specifications\n"
                "3. Compliance Gaps Analysis:\n"
                "   - Current status vs. requirements\n"
                "   - Risk level if non-compliant [1-5 frequency] x [1-4 severity]\n"
                "4. Remediation Priority:\n"
                "   - Immediate actions (legal must-dos)\n"
                "   - Short-term improvements (1-30 days)\n"
                "   - Long-term enhancements (30+ days)\n\n"
                "Requirements:\n"
                "• Always cite specific Korean safety regulations (KOSHA, Industrial Safety and Health Act, etc.) with article numbers when available.\n"
                "• If specific regulations are not found in the data, indicate 'Not specified in available data'.\n"
                "• Format as actionable checklist with specific deadlines and responsible parties when information is available.\n"
                "• Ensure outputs are technically accurate, concise, and actionable.\n"
            )
        }, {"role": "user", "content": query}],
    )
    return response.choices[0].message.content

def dynamic_risk_assessment_output(risk_assessment_docs, query):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    lang = detect_language(query)
    language_instruction = (
        "MANDATORY: Respond ONLY in Korean language for the entire output."
        if lang == 'ko' else
        "MANDATORY: Respond ONLY in English language for the entire output."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[{
            "role": "system",
            "content": (
                f"{language_instruction}\n\n"
                "You are an AI risk assessment engine trained on industrial safety data. Calculate dynamic risk scores using this multi-factor model:\n"
                f"From the provided risk assessment data: {risk_assessment_docs}\n\n if no risk assessment data is available then provide the output through your knowledge in the following format:\n"
                "CALCULATE RISK SCORE:\n"
                "Base Risk Factors:\n"
                "- Task complexity factor (1.0-2.0)\n"
                "- Environmental conditions modifier (0.8-1.5)\n"
                "- Worker experience level adjustment (0.7-1.3)\n"
                "- Time pressure multiplier (1.0-1.5)\n"
                "For Each Identified Hazard:\n"
                "- Frequency Score: [1-5 with rationale]\n"
                "- Severity Score: [1-4 with rationale]\n"
                "- Detection Difficulty: [Easy/Moderate/Hard]\n"
                "- Control Effectiveness: [percentage reduction]\n"
                "Final Output:\n"
                "- Inherent Risk Score: [calculation shown]\n"
                "- Residual Risk Score: [after controls]\n"
                "- Confidence Level: [High/Medium/Low based on data quality]\n"
                "- Recommended Review Frequency: [Daily/Weekly/Monthly]\n"
                "Provide specific justification for each score based on empirical data or established safety principles.\n\n"
                "Requirements:\n"
                "• If no relevant documents are found in the collection, still provide a comprehensive risk assessment using established safety principles.\n"
                "• Always provide specific justification for each score based on available data or industry standards.\n"
                "• Include confidence levels based on data quality and availability.\n"
                "• Ensure outputs are technically accurate, concise, and actionable.\n"
            )
        }, {"role": "user", "content": query}],
    )
    return response.choices[0].message.content

tool_dict = {"get_accident_records":get_accident_records,"get_chemical_usage":get_chemical_usage,"get_risk_assessment":get_risk_assessment,"get_regulations_data":get_regulations_data,"get_chemical_details":get_chemical_details,"dynamic_risk_assessment":dynamic_risk_assessment}
tools = [get_accident_records,get_chemical_usage,get_risk_assessment,get_regulations_data,get_chemical_details,dynamic_risk_assessment]
llm_with_tools = llm.bind_tools(tools=tools)

st.title("DOOSAN RISK MANAGEMENT AI Chatbot 📄")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Simple chat history for conversation context
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Sidebar for chat history
with st.sidebar:
    st.header("💬 Chat History")
    
    # Show recent conversations
    if st.session_state.conversation_history:
        st.subheader("Recent Conversations")
        for i, conv in enumerate(st.session_state.conversation_history[-5:]):  # Show last 5
            with st.expander(f"Chat {len(st.session_state.conversation_history) - 4 + i}", expanded=False):
                st.write(f"**User:** {conv['User']}")
                st.write(f"**AI:** {conv['AI'][:100]}...")  # Truncated for sidebar
    
    # Clear chat history button
    if st.button("🗑️ Clear Chat History", type="secondary"):
        st.session_state.conversation_history = []
        st.session_state.chat_history = []
        st.rerun()
    
    # Show total conversations
    st.metric("Total Conversations", len(st.session_state.conversation_history))

# Main chat area with background conversation history
col1, col2 = st.columns([3, 1])

with col1:
    # Display current conversation in background
    if st.session_state.conversation_history:
        with st.expander("📚 Previous Conversations (Background Context)", expanded=False):
            for i, conv in enumerate(st.session_state.conversation_history):
                st.write(f"**Conversation {i+1}:**")
                st.write(f"👤 **User:** {conv['User']}")
                st.write(f"🤖 **AI:** {conv['AI'][:300]}{'...' if len(conv['AI']) > 300 else ''}")
                st.write("---")

# for message in st.session_state.chat_history:
#     if message:
#         print("Here is messages tools--------",message)
#         with st.expander(label=message.tool_call_id,icon='📖'):
#             st.write(message.content)
#         continue
#     with st.chat_message(message.type):
#         st.markdown(message.content)

if user_query := st.chat_input("Ask a question about chemical usage, accidents, or risks / 화학 물질 사용, 사고 또는 위험에 대해 질문하세요"):
    
    with st.chat_message("human"):
        st.markdown(user_query)

    with st.chat_message("ai"):
        user_lang = detect_language(user_query)
        with st.spinner("Thinking..." if user_lang == 'en' else "생각 중..."):
            # Build context with recent chat history
            language_instruction = (
                "IMPORTANT: Respond ONLY in English language for ALL parts of your response."
                if user_lang == 'en' else
                "IMPORTANT: Respond ONLY in Korean language for ALL parts of your response."
            )
            context_query = user_query
            if st.session_state.conversation_history:
                # Get the last 3 conversations for context
                recent_history = st.session_state.conversation_history[-3:] if len(st.session_state.conversation_history) >= 3 else st.session_state.conversation_history
                history_context = "\n".join([f"Previous: User: {h['User']} | AI: {h['AI'][:200]}..." for h in recent_history])
                context_query = f"Previous conversation context:\n{history_context}\n\nCurrent query: {context_query}"
            
            accident_keywords = [
                'safety improvement', 'accident', 'injury', 'incident',
                'what happened', 'how many accidents', 'safety analysis',
                '안전 개선', '사고', '부상', '재해', '위험 분석'
            ]

            # Enhanced system instruction with accident context if relevant
            accident_context = ""
            if any(keyword in user_query.lower() for keyword in accident_keywords):
                try:
                    accident_docs_ctx = get_accident_records(user_query)
                    if accident_docs_ctx and len(accident_docs_ctx) > 100:
                        accident_context = f"\n\nRELEVANT ACCIDENT DATA CONTEXT:\n{accident_docs_ctx[:2000]}..."
                    else:
                        accident_context = "\n\nNote: Limited accident data available in system."
                except Exception as e:
                    print(f"Error retrieving accident context: {e}")
                    accident_context = "\n\nNote: Unable to retrieve accident data context."

            system_instruction = (
                f"{language_instruction}\n\n"
                "IMPORTANT RULES:\n"
                "1. When asked about safety improvements or accidents, use ACTUAL data\n"
                "2. Reference specific incidents, never generic advice\n"
                "3. Count real accidents and identify actual patterns\n"
                "4. Mention specific dates, names, and locations from the data"
                f"{accident_context}"
            )
            messages = [SystemMessage(content=system_instruction), HumanMessage(context_query)]
            ai_msg = llm_with_tools.invoke(messages)
            print(ai_msg)
            messages.append(ai_msg)
            # Initialize output variable
            output = "No response generated"
            
            # Use while loop for continuous tool calls
            while ai_msg.tool_calls:
                for tool_call in ai_msg.tool_calls:
                    selected_tool = tool_dict[tool_call["name"].lower()]
                    print("Here is the selected tool-------",selected_tool)
                    tool_name= selected_tool.name
                    print("Here is the selected tool name -------",tool_name)
                    
                    if tool_name == "get_chemical_details":
                        table_data=get_chemical_details(user_query)
                        output=chemical_output(table_data=table_data,query=user_query)
                    elif tool_name == "get_chemical_usage":
                        chemical_usage_docs=get_chemical_usage(user_query)
                        # Validate chemical data quality
                        is_valid, message = validate_chemical_data_quality(chemical_usage_docs)
                        if not is_valid:
                            output = f"Chemical data quality issue: {message}"
                        else:
                            output=chemical_output(table_data=chemical_usage_docs,query=user_query)
                    elif tool_name == "get_regulations_data":
                        regulations_docs=get_regulations_data(user_query)
                        output=regulations_output(regulations_docs=regulations_docs,query=user_query)
                    elif tool_name == "get_accident_records":
                        accident_docs=get_accident_records(user_query)
                        # Validate data quality
                        is_valid, message = validate_data_quality(accident_docs)
                        if not is_valid:
                            output = f"Data quality issue: {message}"
                        else:
                            output=accident_output(accident_docs=accident_docs,query=user_query)
                    elif tool_name == "get_risk_assessment":
                        risk_assessment_docs=get_risk_assessment(user_query)
                        if "table" in user_query.lower() or "테이블" in user_query.lower():
                            output=risk_assessment_docs
                            json_data=risk_assessment_table_output(risk_assessment_docs=risk_assessment_docs,query=user_query)
                            if json_data:
                                print("Here is the json data-------",json_data)
                                # Ensure json_data is a list
                                if isinstance(json_data, dict):
                                    json_data = [json_data]
                                elif not isinstance(json_data, list):
                                    json_data = [json_data]
                                df = pd.DataFrame(json_data)
                                st.subheader("📊 Risk Assessment Analysis")
                                st.dataframe(df, use_container_width=True, hide_index=True)
                                output = f"Retrieved {len(json_data)} risk assessment records"
                            else:
                                st.error("No valid risk assessment data to display")
                                output = "Error processing risk assessment data"
                        else:
                            output=risk_assessment_output(risk_assessment_docs=risk_assessment_docs,query=user_query)
                    elif tool_name == "dynamic_risk_assessment":
                        risk_assessment_docs=dynamic_risk_assessment(user_query)
                        output=dynamic_risk_assessment_output(risk_assessment_docs=risk_assessment_docs,query=user_query)
                    else:
                        # Handle unknown tool
                        output = f"Unknown tool: {tool_name}"
                    
                    # Add tool result to messages
                    from langchain_core.messages import ToolMessage
                    tool_message = ToolMessage(content=output, tool_call_id=tool_call["id"])
                    messages.append(tool_message)
                
                # Get next AI response
                ai_msg = llm_with_tools.invoke(messages)
                messages.append(ai_msg)

            # Display final answer
            answer = ai_msg.content if ai_msg.content else output
            st.markdown(answer)
            st.session_state.chat_history.extend(messages)
            
            # Add to conversation history
            history = {"User": user_query, "AI": answer}
            st.session_state.conversation_history.append(history)
