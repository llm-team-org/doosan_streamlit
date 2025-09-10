import os
import streamlit as st
from qdrant_client import models
import qdrant_client
import xml.etree.ElementTree as ET
from tabulate import tabulate
import requests
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import json
import pandas as pd
from typing import Dict, List, Optional, Any

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

if "client" not in st.session_state:
    st.session_state.client = qdrant_client.QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        port=None
    )

# ============== XML TO TABLE FUNCTION ==============
def xml_to_table(xml_data):
    """Parses XML data and returns a list of dictionaries"""
    root = ET.fromstring(xml_data)
    items = root.findall(".//item")
    table_data = []
    
    if items:
        headers = [child.tag for child in items[0]]
        table_data.append(headers)
        
        for item in items:
            row = []
            for header in headers:
                element = item.find(header)
                row.append(element.text if element is not None else "")
            table_data.append(row)
    
    return table_data

# ============== TOOL DEFINITIONS ==============
@tool
def get_accident_records(query: str) -> str:
    """
    Retrieves information from company documents.
    This tool should be used to answer questions about accident records.
    The search query must be in Korean.
    Example query: '화학 물질 사용', '사고 기록', '위험 평가'.
    """
    retrieved_docs = st.session_state.client.query_points(
        collection_name="doosan_accident_records",
        query=models.Document(text=query, model="Qdrant/minicoil-v1"),
        using="minicoil",
        limit=5
    )
    docs = [doc.payload['text'] for doc in retrieved_docs.points]
    return "\n\n".join(docs)

def accident_output(accident_docs, query):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"You are a helpful assistant. From this data give the user information {accident_docs}. From the collected docs table carefully extract the accident records relevant to the query entered by the user"},
            {"role": "user", "content": query}
        ]
    )
    return response.choices[0].message.content

@tool
def get_chemical_usage(query: str) -> str:
    """
    Retrieves information from company documents.
    This tool should be used to answer questions about chemical usage.
    The search query must be in Korean.
    Example query: '화학 물질 사용', '사고 기록', '위험 평가'.
    """
    retrieved_docs = st.session_state.client.query_points(
        collection_name="doosan_chemical_usage",
        query=models.Document(text=query, model="Qdrant/minicoil-v1"),
        using="minicoil",
        limit=5
    )
    docs = [doc.payload['text'] for doc in retrieved_docs.points]
    return "\n\n".join(docs)

@tool
def get_risk_assessment(query: str) -> str:
    """
    Retrieves information from company documents.
    This tool should be used to answer questions about risk assessment.
    The search query must be in Korean.
    Example query: '화학 물질 사용', '사고 기록', '위험 평가'.
    """
    retrieved_docs = st.session_state.client.query_points(
        collection_name="doosan_risk_assessment",
        query=models.Document(text=query, model="Qdrant/minicoil-v1"),
        using="minicoil",
        limit=5
    )
    docs = [doc.payload['text'] for doc in retrieved_docs.points]
    return "\n\n".join(docs)

def risk_assessment_output(risk_assessment_docs, query):
    """
    Complete risk assessment with all calculations using LLM
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Comprehensive prompt for complete risk calculation
    calculation_prompt = f"""
You are a Risk Assessment Expert. Analyze the provided risk assessment documents and calculate ALL risk values.

RISK CALCULATION FORMULAS AND RULES:

1. FREQUENCY LEVELS:
   - Low: Events occurring rarely (less than once per year)
   - Medium: Events occurring occasionally (monthly to yearly)
   - High: Events occurring frequently (weekly to daily)

2. SEVERITY LEVELS:
   - Low: Minor injury, no lost time, minimal damage
   - Medium: Medical treatment required, moderate damage
   - High: Serious injury, lost time, major damage

3. RISK MATRIX (Frequency x Severity):
   - Low x Low = LOW Risk
   - Low x Medium = LOW Risk
   - Low x High = MEDIUM Risk
   - Medium x Low = LOW Risk
   - Medium x Medium = MEDIUM Risk
   - Medium x High = HIGH Risk
   - High x Low = MEDIUM Risk
   - High x Medium = HIGH Risk
   - High x High = HIGH Risk

4. RISK REDUCTION CALCULATION:
   For reduction measures, calculate improved values based on:
   - Engineering controls: Reduce frequency/severity by 1-2 levels
   - Administrative controls: Reduce by 1 level
   - PPE: Minimal reduction, maybe reduce severity by 1 level
   - Combined measures: Cumulative effect

DOCUMENTS TO ANALYZE:
{risk_assessment_docs}

USER QUERY:
{query}

INSTRUCTIONS:
1. Extract ALL hazard information from the documents
2. For each hazard, CALCULATE:
   - Current Frequency (Low/Medium/High)
   - Current Severity (Low/Medium/High)
   - Current Risk (use matrix above)
   - Improved Frequency after reduction measures
   - Improved Severity after reduction measures
   - Improved Risk (recalculate using matrix)

3. Return ONLY a JSON array with this EXACT structure for each hazard:
[
  {{
    "Hazard Factor": "Specific description of the hazard",
    "Current Frequency": "Low/Medium/High",
    "Current Severity": "Low/Medium/High",
    "Current Risk": "Low/Medium/High",
    "Current Measures": "List of existing safety measures or N/A",
    "Reduction Measures": "List of proposed reduction measures or N/A",
    "Improved Frequency": "Low/Medium/High",
    "Improved Severity": "Low/Medium/High",
    "Improved Risk": "Low/Medium/High"
  }}
]

IMPORTANT:
- Calculate ALL fields, don't use N/A for calculated fields
- If no data exists, still provide reasonable estimates based on hazard type
- Ensure Current Risk and Improved Risk are calculated using the matrix
- Show improvement from reduction measures (Improved values should be better or same as Current)
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": calculation_prompt},
                {"role": "user", "content": f"Analyze and calculate risks for: {query}"}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )

        # Parse the response
        result = response.choices[0].message.content
        parsed_json = json.loads(result)
        
        # Ensure it's a list
        if isinstance(parsed_json, dict):
            # Check if it has a 'data' or 'results' key
            if 'data' in parsed_json:
                parsed_json = parsed_json['data']
            elif 'results' in parsed_json:
                parsed_json = parsed_json['results']
            elif 'hazards' in parsed_json:
                parsed_json = parsed_json['hazards']
            else:
                # If it's a single hazard dict, wrap it in a list
                parsed_json = [parsed_json]
        
        # Validate and ensure all fields are present
        validated_results = []
        for item in parsed_json:
            validated_item = {
                "Hazard Factor": item.get("Hazard Factor", "Unknown Hazard"),
                "Current Frequency": item.get("Current Frequency", "Medium"),
                "Current Severity": item.get("Current Severity", "Medium"),
                "Current Risk": item.get("Current Risk", "Medium"),
                "Current Measures": item.get("Current Measures", "N/A"),
                "Reduction Measures": item.get("Reduction Measures", "N/A"),
                "Improved Frequency": item.get("Improved Frequency", item.get("Current Frequency", "Low")),
                "Improved Severity": item.get("Improved Severity", item.get("Current Severity", "Low")),
                "Improved Risk": item.get("Improved Risk", "Low")
            }
            
            # Ensure improved values show improvement or stay the same
            risk_levels = {"Low": 1, "Medium": 2, "High": 3}
            current_risk_val = risk_levels.get(validated_item["Current Risk"], 2)
            improved_risk_val = risk_levels.get(validated_item["Improved Risk"], 1)
            
            # If improved risk is worse than current, set it to current
            if improved_risk_val > current_risk_val:
                validated_item["Improved Risk"] = validated_item["Current Risk"]
            
            validated_results.append(validated_item)
        
        return validated_results
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Response content: {response.choices[0].message.content if response else 'No response'}")
        # Return default structure on error
        return [{
            "Hazard Factor": "Error parsing risk data",
            "Current Frequency": "N/A",
            "Current Severity": "N/A",
            "Current Risk": "N/A",
            "Current Measures": "N/A",
            "Reduction Measures": "N/A",
            "Improved Frequency": "N/A",
            "Improved Severity": "N/A",
            "Improved Risk": "N/A"
        }]
    except Exception as e:
        print(f"Error in risk_assessment_output: {e}")
        return [{
            "Hazard Factor": f"Error: {str(e)}",
            "Current Frequency": "N/A",
            "Current Severity": "N/A",
            "Current Risk": "N/A",
            "Current Measures": "N/A",
            "Reduction Measures": "N/A",
            "Improved Frequency": "N/A",
            "Improved Severity": "N/A",
            "Improved Risk": "N/A"
        }]

@tool
def get_chemical_details(query: str):
    """
    This tool should be used when user wants to extract any information about chemicals
    Example query: 'What is cas no of oxygen', 'what is chemical ID of nitrogen'.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful chemical identifier assistant. Your job is to identify the chemical name only and return me only chemical name single word, Response: Only one word chemical name, if no chemical name is available then return 'none'"},
            {"role": "user", "content": query}
        ]
    )
    chemical_name = response.choices[0].message.content.strip()
    
    if chemical_name.lower() == "none":
        return None
    else:
        url = "https://msds.kosha.or.kr/openapi/service/msdschem/chemlist"
        params = {
            'serviceKey': os.getenv("KOSHA_API_KEY"),
            'searchWrd': chemical_name,
            'searchCnd': '0',
            'numOfRows': '10'
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return xml_to_table(response.text)
        return None

def chemical_output(table_data, query):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"You are a helpful assistant. From this data give the user information {table_data}. From this table carefully extract casNo, chemId, chemNameKor, enNo, KeNO, lastDate, unNo and is it Kosha Confirmed or not for the given Chemical name entered by the user"},
            {"role": "user", "content": query}
        ]
    )
    return response.choices[0].message.content

@tool
def get_regulations_data(query: str) -> str:
    """
    Retrieves information from company documents.
    This tool should be used to answer questions about regulations and compliance.
    The search query must be in Korean.
    Example query: '규정', '법규', '준수 사항', '컴플라이언스'.
    """
    retrieved_docs = st.session_state.client.query_points(
        collection_name="doosan_regulations_data",
        query=models.Document(text=query, model="Qdrant/minicoil-v1"),
        using="minicoil",
        limit=5
    )
    docs = [doc.payload['text'] for doc in retrieved_docs.points]
    return "\n\n".join(docs)

def regulations_output(regulations_docs, query):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"You are a helpful assistant. From this data give the user information {regulations_docs}. From the collected docs table carefully extract the relevant rules relevant to the query entered by the user"},
            {"role": "user", "content": query}
        ]
    )
    return response.choices[0].message.content

# ============== MAIN APPLICATION ==============

tool_dict = {
    "get_accident_records": get_accident_records,
    "get_chemical_usage": get_chemical_usage,
    "get_risk_assessment": get_risk_assessment,
    "get_regulations_data": get_regulations_data,
    "get_chemical_details": get_chemical_details
}

tools = [get_accident_records, get_chemical_usage, get_risk_assessment, 
         get_regulations_data, get_chemical_details]

llm_with_tools = llm.bind_tools(tools=tools)

# ============== STREAMLIT UI ==============

st.title("🛡️ DOOSAN RISK MANAGEMENT AI Chatbot")
st.caption("AI-Powered Risk Assessment with Automated Calculations")

# Sidebar with information
with st.sidebar:
    st.header("📊 Risk Assessment Matrix")
    st.markdown("""
    **Risk Calculation Formula:**
    ```
    Risk = Frequency × Severity
    ```
    
    **Risk Matrix:**
    | F\\S | Low | Medium | High |
    |-----|-----|--------|------|
    | **Low** | Low | Low | Medium |
    | **Medium** | Low | Medium | High |
    | **High** | Medium | High | High |
    
    **Improvement Factors:**
    - 🔧 Engineering Controls: High reduction
    - 📋 Administrative: Medium reduction
    - 🦺 PPE: Low reduction
    """)
    
    st.divider()
    
    st.info("""
    **How it works:**
    1. System retrieves hazard data
    2. AI calculates current risk levels
    3. AI evaluates reduction measures
    4. AI calculates improved risk
    5. All values displayed in table
    """)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    if hasattr(message, 'type'):
        with st.chat_message(message.type):
            if hasattr(message, 'content'):
                st.markdown(message.content)

# Chat input
if user_query := st.chat_input("Ask a question about chemical usage, accidents, or risks..."):
    
    # Display user message
    with st.chat_message("human"):
        st.markdown(user_query)
    
    # Process with AI
    with st.chat_message("ai"):
        with st.spinner("Analyzing and calculating risks..."):
            messages = [HumanMessage(user_query)]
            ai_msg = llm_with_tools.invoke(messages)
            messages.append(ai_msg)
            
            # Process tool calls
            if ai_msg.tool_calls:
                for tool_call in ai_msg.tool_calls:
                    selected_tool = tool_dict[tool_call["name"].lower()]
                    tool_name = selected_tool.name
                    
                    if tool_name == "get_chemical_details":
                        table_data = get_chemical_details(user_query)
                        output = chemical_output(table_data=table_data, query=user_query)
                        st.markdown(output)
                        
                    elif tool_name == "get_regulations_data":
                        regulations_docs = get_regulations_data(user_query)
                        output = regulations_output(regulations_docs=regulations_docs, query=user_query)
                        st.markdown(output)
                        
                    elif tool_name == "get_accident_records":
                        accident_docs = get_accident_records(user_query)
                        output = accident_output(accident_docs=accident_docs, query=user_query)
                        st.markdown(output)
                        
                    elif tool_name == "get_risk_assessment":
                        risk_assessment_docs = get_risk_assessment(user_query)
                        json_data = risk_assessment_output(
                            risk_assessment_docs=risk_assessment_docs,
                            query=user_query
                        )
                        
                        if json_data and len(json_data) > 0:
                            st.subheader("📊 Risk Assessment Analysis")
                            
                            # Create DataFrame
                            df = pd.DataFrame(json_data)
                            
                            # Display summary metrics
                            col1, col2, col3, col4 = st.columns(4)
                            
                            # Count risk levels
                            current_high = len(df[df['Current Risk'] == 'High'])
                            current_medium = len(df[df['Current Risk'] == 'Medium'])
                            improved_high = len(df[df['Improved Risk'] == 'High'])
                            improved_low = len(df[df['Improved Risk'] == 'Low'])
                            
                            with col1:
                                st.metric("Total Hazards", len(df))
                            with col2:
                                st.metric("High Risks", current_high, delta=f"{improved_high - current_high}")
                            with col3:
                                st.metric("Medium Risks", current_medium)
                            with col4:
                                st.metric("Low Risks (After)", improved_low)
                            
                            # Display the complete table with all calculated values
                            st.dataframe(
                                df,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "Current Risk": st.column_config.TextColumn(
                                        "Current Risk",
                                        help="Calculated from Frequency × Severity"
                                    ),
                                    "Improved Risk": st.column_config.TextColumn(
                                        "Improved Risk",
                                        help="Risk after reduction measures"
                                    )
                                }
                            )
                            
                            # Show risk improvement summary
                            st.success(f"✅ Risk Assessment Complete: Analyzed {len(json_data)} hazard(s)")
                            
                            # Display improvement analysis
                            improvements = []
                            for idx, row in df.iterrows():
                                if row['Current Risk'] != row['Improved Risk']:
                                    improvements.append(
                                        f"• **{row['Hazard Factor']}**: {row['Current Risk']} → {row['Improved Risk']}"
                                    )
                            
                            if improvements:
                                with st.expander("📈 Risk Improvements"):
                                    st.markdown("\n".join(improvements))
                            
                            output = f"Retrieved and calculated {len(json_data)} risk assessment(s) with all values computed"
                        else:
                            st.error("No risk assessment data found")
                            output = "No risk assessment data to display"
                    
                    elif tool_name == "get_chemical_usage":
                        chemical_docs = get_chemical_usage(user_query)
                        output = f"Chemical usage information:\n{chemical_docs[:500]}..."
                        st.markdown(output)
                    
                    messages.append(output)
            else:
                # No tool calls, just display AI response
                output = ai_msg.content if hasattr(ai_msg, 'content') else "I can help you with risk assessments, chemical usage, accident records, and regulations. Please ask a specific question."
                st.markdown(output)
                messages.append(output)
            
            # Update chat history
            st.session_state.chat_history.extend(messages)