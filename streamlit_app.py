import os
import cohere
import streamlit as st
from qdrant_client import models
import qdrant_client
import xml.etree.ElementTree as ET
from tabulate import tabulate
import requests
from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_qdrant import QdrantVectorStore
from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from dotenv import load_dotenv
import json
import pandas as pd
import re
from typing import List, Dict, Any


load_dotenv()

# Initialize Cohere client
COHERE_RERANKER_MODEL="rerank-v3.5"
compressor = CohereRerank(cohere_api_key=os.getenv("COHERE_API_KEY"), model=COHERE_RERANKER_MODEL, top_n=10)

#co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

llm = ChatOpenAI(model="gpt-4.1", temperature=0,api_key=os.getenv("OPENAI_API_KEY"))
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-large")


if "client" not in st.session_state:
    st.session_state.client = qdrant_client.QdrantClient(url=os.getenv("QDRANT_URL"),api_key=os.getenv("QDRANT_API_KEY"),port=None)

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

@tool
def get_accident_records(query: str) -> str:
    """
    Retrieves accident records and safety incident information from company documents.
    Use this tool for questions about workplace accidents, safety incidents, injuries, or safety violations.
    Supports both English and Korean queries.
    Example queries: '사고 기록', 'accident records', '안전 사고', 'workplace injuries', '사고 보고서', 'safety incidents', 'accident analysis', ''.
    """
    # retrieved_docs = st.session_state.client.query_points(
    #     collection_name="doosan_accident_records",
    #     query=models.Document(
    #         text=query,
    #         model="Qdrant/minicoil-v1"
    #     ),
    #     using="minicoil",
    #     limit=5
    # )
    # docs = [doc.payload['text'] for doc in retrieved_docs.points]
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
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever)
    docs=compression_retriever.invoke(query)
    # print("retrieved_docs",retrieved_docs)

    #     # Apply Cohere reranking
    # if retrieved_docs:
    #     # Convert retrieved docs to format expected by Cohere
    #     docs = [doc.page_content for doc in retrieved_docs]
        
    #     # Rerank using Cohere
    #     response = co.rerank(
    #         model="rerank-v3.5",
    #         query=query,
    #         documents=docs,
    #         top_n=5,
    #     )
    #     print("Cohere rerank response:", response)
        
    #     # Return reranked documents
    #     reranked_docs = []
    #     for result in response.results:
    #         reranked_docs.append(retrieved_docs[result.index])
    #     print("reranked_docs",reranked_docs)
        
    # return reranked_docs

    # docs = [doc.payload['text'] for doc in retrieved_docs.points]
    # print("docs",docs)
    return (docs)


def accident_output(accident_docs,query):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.responses.create(
        model="gpt-4.1",
        temperature=0.2,
        # instructions=(
        #     "You are an expert industrial safety analyst specializing in accident prevention and risk assessment. "
        #     f"Analyze the provided accident data: {accident_docs}\n"
        #     "Extract only information relevant to the user query and generate a comprehensive incident risk profile.\n\n"
            
        #     "INCIDENT RISK PROFILE\n\n"
            
        #     "HISTORICAL INCIDENT ANALYSIS\n"
        #     "- Similar Incidents: [identify patterns from accident database]\n"
        #     "- Frequency Trends: [how often similar incidents occur]\n"
        #     "- Common Factors: [shared characteristics across incidents]\n\n"
            
        #     "PROBABILITY ASSESSMENT\n"
        #     "- Base Rate: [% from historical data with confidence level]\n"
        #     "- Current Conditions: [1-5 scale adjustment with justification]\n"
        #     "- Overall Risk Level: [Low/Medium/High/Critical]\n\n"
            
        #     "SEVERITY OUTCOMES\n"
        #     "- Most Likely Scenario: [severity 1-4 with description]\n"
        #     "- Worst Case Scenario: [severity 1-4 with description]\n"
        #     "- Impact Assessment: [physical, financial, operational impacts]\n\n"
            
        #     "CRITICAL CONTROL POINTS\n"
        #     "- High-Risk Moments: [specific times/actions when incidents occur]\n"
        #     "- Vulnerable Processes: [processes most susceptible to accidents]\n"
        #     "- Human Factors: [behavioral or procedural risk factors]\n\n"
            
        #     "MONITORING AND PREVENTION\n"
        #     "- Leading Indicators: [measurable precursors to watch for]\n"
        #     "- Safety Barriers: [prevention and mitigation layers]\n"
        #     "- Emergency Response: [immediate actions if incident occurs]\n\n"
            
        #     "Requirements:\n"
        #     "- Provide confidence levels (High/Medium/Low) for each assessment\n"
        #     "- If no relevant data found, state 'No relevant historical data available'\n"
        #     "- Focus on actionable insights and specific preventive measures\n"
        #     "- Include accident patterns, root causes, and lessons learned\n"
        #     "- Ensure technical accuracy and practical applicability\n"
        #     "- If user query is in Korean, provide output in Korean; otherwise use English"
        # ),
        instructions=(
            f"""
            You are an expert industrial safety analyst with specialized expertise in accident prevention, incident investigation, and risk assessment. Provide comprehensive incident risk profiles with historical analysis and actionable prevention strategies.
            
            ## Context
            Available Accident Documentation:
            '''{accident_docs}'''
            
            ## User Current Query:
            '''{query}'''
            - Carefully understand the language of User Current Query and remember this as LANG. You will respond in LANG only.
            
            ## Task
            1) Analyze the provided accident documentation against the user's specific query
            2) Extract only information directly relevant to incident risk profiling
            3) Generate comprehensive incident risk profile using structured framework
            4) Provide probability assessments with confidence levels and historical context
            5) Develop actionable prevention and monitoring strategies
            6) When historical data is unavailable, clearly indicate limitations
            
            ## STRICT Language Policy (Hard Constraint)
            - Let LANG := the language of the User Current Query and previous conversation context: {query} (English or Korean)
            - Respond **must be** in LANG. Do **not** mix languages.
            - Do not include translations, bilingual headers, or explanations in any other language.
            
            ## Output Format
            - Response should always be in **MARKDOWN** format
            - Focus on actionable insights and specific preventive measures
            - Provide confidence levels for each assessment
            - No prefaces unless they are written in LANG
            - No mention of internal processing steps
            
            ## Required Output Structure
            
            ### INCIDENT RISK PROFILE
            
            **HISTORICAL INCIDENT ANALYSIS**
            - **Similar Incidents:** [identify patterns and trends from accident database]
            - **Frequency Trends:** [quantitative analysis of how often similar incidents occur]
            - **Common Factors:** [shared characteristics, root causes, and contributing elements across incidents]
            - **Data Quality:** [High/Medium/Low confidence based on available historical data]
            
            **PROBABILITY ASSESSMENT**
            - **Base Rate:** [percentage from historical data with confidence level and sample size]
            - **Current Conditions:** [1-5 scale adjustment with detailed justification]
              - 1: Much Lower | 2: Lower | 3: Similar | 4: Higher | 5: Much Higher
            - **Adjusted Probability:** [base rate × current conditions adjustment]
            - **Overall Risk Level:** [Low/Medium/High/Critical with probability ranges]
            - **Assessment Confidence:** [High/Medium/Low based on data quality and relevance]
            
            **SEVERITY OUTCOMES**
            - **Most Likely Scenario:** [severity 1-4 with detailed description and probability]
              - 1: Minor | 2: Moderate | 3: Major | 4: Catastrophic
            - **Worst Case Scenario:** [severity 1-4 with detailed description and probability]
            - **Impact Assessment:**
              - Physical: [injury types, severity, affected personnel]
              - Financial: [direct costs, indirect costs, business interruption]
              - Operational: [downtime, productivity loss, reputation impact]
            
            **CRITICAL CONTROL POINTS**
            - **High-Risk Moments:** [specific times, activities, or conditions when incidents are most likely]
            - **Vulnerable Processes:** [processes, equipment, or systems most susceptible to accidents]
            - **Human Factors:** [behavioral, procedural, or organizational risk factors]
            - **Environmental Triggers:** [conditions that increase incident probability]
            
            **MONITORING AND PREVENTION**
            - **Leading Indicators:** [measurable precursors and early warning signs to monitor]
            - **Safety Barriers:** 
              - Prevention Layers: [measures to prevent incident initiation]
              - Mitigation Layers: [measures to reduce consequences if incident occurs]
            - **Emergency Response:** [immediate actions and procedures if incident occurs]
            - **Continuous Improvement:** [lessons learned and preventive measure updates]
            
            ## Content Requirements
            - Extract accident patterns, root causes, and lessons learned from historical data
            - Provide quantitative analysis when possible, qualitative when necessary
            - Include confidence levels (High/Medium/Low) for each major assessment
            - Focus on practical applicability and actionable recommendations
            - Ensure technical accuracy and evidence-based conclusions
            - Distinguish between statistical trends and individual case studies
            
            ## Data Analysis Framework
            - Use historical accident frequency to establish base rates
            - Apply current condition modifiers to adjust probabilities
            - Consider both individual incident severity and cumulative risk
            - Analyze patterns across time, location, equipment, and personnel
            - Include near-miss and precursor event analysis when available
            
            ## Fallback Protocol
            When accident documentation is insufficient or unavailable:
            - Clearly state: "No relevant historical data available" for specific sections
            - Provide general industry knowledge and best practices where appropriate
            - Lower confidence levels and indicate data limitations
            - Recommend specific data collection for future risk profiling
            - Focus on preventive measures based on similar industry incidents
            
            ## Confidence Level Guidelines
            - **High Confidence:** Robust historical data (20+ incidents), clear patterns, recent data
            - **Medium Confidence:** Moderate data (5-19 incidents), some uncertainty, older data
            - **Low Confidence:** Limited data (<5 incidents), high uncertainty, extrapolated estimates
            
            Always ensure technical accuracy, evidence-based analysis, and actionable prevention recommendations while maintaining strict adherence to the language policy.
            """
        ),
        input=query,
    )
    return response.output_text

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
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever)
    docs=compression_retriever.invoke(query)
    # print("retrieved_docs",retrieved_docs)

        # Apply Cohere reranking
    # if retrieved_docs:
    #     # Convert retrieved docs to format expected by Cohere
    #     docs = [doc.page_content for doc in retrieved_docs]
        
    #     # Rerank using Cohere
    #     response = co.rerank(
    #         model="rerank-v3.5",
    #         query=query,
    #         documents=docs,
    #         top_n=5,
    #     )
    #     print("Cohere rerank response:", response)
        
    #     # Return reranked documents
    #     reranked_docs = []
    #     for result in response.results:
    #         reranked_docs.append(retrieved_docs[result.index])
    #     print("reranked_docs",reranked_docs)
        
    # return reranked_docs

    # docs = [doc.payload['text'] for doc in retrieved_docs.points]
    # print("docs",docs)
    return (docs)
    # return retrieved_docs


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
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever)
    docs=compression_retriever.invoke(query)
    # retriever = qdrant.as_retriever(search_kwargs={"k": 30})
    # retrieved_docs = retriever.invoke(query)
    # # print("retrieved_docs",retrieved_docs)
    #     # Apply Cohere reranking
    # if retrieved_docs:
    #     # Convert retrieved docs to format expected by Cohere
    #     docs = [doc.page_content for doc in retrieved_docs]
        
    #     # Rerank using Cohere
    #     response = co.rerank(
    #         model="rerank-v3.5",
    #         query=query,
    #         documents=docs,
    #         top_n=5,
    #     )
    #     print("Cohere rerank response:", response)
        
    #     # Return reranked documents
    #     reranked_docs = []
    #     for result in response.results:
    #         reranked_docs.append(retrieved_docs[result.index])
    #     print("reranked_docs",reranked_docs)
        
    # return reranked_docs
    # docs = [doc.payload['text'] for doc in retrieved_docs.points]
    return (docs)

def risk_assessment_table_output(risk_assessment_docs, query):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    json_instructions = f"""
You are a highly specialized safety engineering expert with comprehensive expertise in industrial risk assessments and structured data analysis. Generate complete risk assessment tables in strict JSON format.

## Context
Available Risk Assessment Documentation:
'''{risk_assessment_docs}'''

## User Current Query:
'''{query}'''
- Carefully understand the language of User Current Query and remember this as LANG. All table content will be in LANG only.

## Task
1) Analyze the provided risk assessment documents against the user's specific query
2) Extract comprehensive risk assessment data for ALL relevant tasks/processes
3) Generate complete risk assessment table covering the full scope of available data
4) Provide quantitative risk scores with clear methodology
5) Include actionable control measures and residual risk calculations
6) Return output in strict JSON format with predefined field structure

## STRICT Language Policy (Hard Constraint)
- Let LANG := the language of the User Current Query and previous conversation context: {query} (English or Korean)
- ALL table content **must be** in LANG. Do **not** mix languages.
- Field names remain in English, but all field VALUES must be in LANG
- Do not include translations or explanations in any other language.

## JSON Output Requirements
- ALWAYS return output in strict JSON format only
- Use the predefined field names exactly as specified below
- Include ALL available risk assessment entries, not just selected ones
- If specific data cannot be found for any field, set value to "N/A"
- If no structured data is available at all, return JSON array with single object having all fields set to "N/A"
- Do not include explanations, extra text, or markdown formatting
- Ensure valid JSON syntax with proper escaping

## Risk Scoring Methodology
- Frequency Scale: 1=Very Rare, 2=Rare, 3=Possible, 4=Likely, 5=Almost Certain
- Severity Scale: 1=Minor, 2=Moderate, 3=Major, 4=Catastrophic  
- Risk Score: Frequency × Severity (1-20 scale)
- Risk Level: Low(1-6), Medium(7-12), High(13-16), Critical(17-20)

## Required JSON Output Structure:
[
  {{
    "Task/Process Name": "[Specific work being performed in LANG or N/A]",
    "Hazard Factor": "[Primary hazard description in LANG or N/A]", 
    "Hazard Identification": "[List of 3-5 specific hazards in LANG or N/A]",
    "Current Frequency": "[1-5 numerical score or Low/Medium/High in LANG or N/A]",
    "Current Severity": "[1-4 numerical score or Low/Medium/High in LANG or N/A]", 
    "Current Risk": "[Risk score (1-20) or Low/Medium/High/Critical in LANG or N/A]",
    "Current Risk Level": "[Frequency (1-5) x Severity (1-4) = Risk Score calculation in LANG or N/A]",
    "Root Causes": "[Underlying reasons for hazards in LANG or N/A]",
    "Current Measures": "[Existing safety measures and controls in LANG or N/A]",
    "Control Measures": "[Specific preventive/protective actions in LANG or N/A]",
    "Reduction Measures": "[Proposed risk reduction measures in LANG or N/A]",
    "Improved Frequency": "[1-5 numerical score after controls in LANG or N/A]",
    "Improved Severity": "[1-4 numerical score after controls in LANG or N/A]",
    "Improved Risk": "[Risk score after controls (1-20) in LANG or N/A]",
    "Residual Risk After Controls": "[New Frequency x New Severity = New Risk Score calculation in LANG or N/A]"
  }}
]

## Content Requirements
- Extract information for ALL relevant tasks/processes found in the documentation
- Provide comprehensive risk assessment table covering complete scope
- Include specific, actionable recommendations for each risk assessment entry
- Use quantitative scoring when possible, qualitative descriptors when necessary
- Ensure technical accuracy and regulatory awareness
- Show risk score calculations transparently
- Include both current state and post-control state assessments

## Data Extraction Protocol
- Prioritize quantitative data over qualitative descriptions
- Extract complete risk assessment workflows when available
- Include all safety measures and control recommendations
- Capture both immediate and underlying root causes
- Document existing controls and proposed improvements

## Fallback Protocol
When risk assessment documentation is insufficient:
- Set unavailable fields to "N/A" 
- Extract any available partial information
- Do not fabricate data or make unsupported assumptions
- Include general safety principles only when specifically relevant
- Maintain JSON structure integrity regardless of data availability

Critical: Return ONLY the JSON array. No additional text, explanations, or formatting outside the JSON structure.
"""
    # Create structured prompt for JSON output
#     json_instructions = f"""
# You are a safety engineering expert specializing in industrial risk assessments. 
# Analyze the provided risk assessment documents: {risk_assessment_docs}

# Instructions:
# - Always return the output in strict JSON format
# - Use the predefined field names exactly as listed below
# - If data cannot be found, set the field value to "N/A"
# - If no structured data is available at all, return a JSON object with all fields set to "N/A"
# - Do not include explanations, extra text, or markdown—only JSON
# - Provide whole table 
# - If the user query is in korean language then provide the output table in korean language
# - Extract only the information relevant to the user query
# - Generate a comprehensive risk assessment with the specified format
# - Include specific, actionable recommendations for each risk assessment
# - If the user query is in korean language then provide the output in korean language
# - Include specific, actionable recommendations for each risk assessment

# Output Format:
# [
#   {{
#     "Task/Process Name": "Specific work being performed or N/A",
#     "Hazard Factor": "Description of the hazard",
#     "Hazard Identification": "List of 3-5 specific hazards or N/A",
#     "Current Frequency": "Low/Medium/High or 1-5 scale or N/A",
#     "Current Severity": "Low/Medium/High or 1-4 scale or N/A",
#     "Current Risk": "Low/Medium/High or Risk Score or N/A",
#     "Current Risk Level": "Frequency (1-5) x Severity (1-4) = Risk Score or N/A",
#     "Root Causes": "Underlying reasons for each hazard or N/A",
#     "Current Measures": "Existing safety measures or N/A",
#     "Control Measures": "Specific preventive/protective actions or N/A",
#     "Reduction Measures": "Proposed reduction measures or N/A",
#     "Improved Frequency": "Low/Medium/High or 1-5 scale or N/A",
#     "Improved Severity": "Low/Medium/High or 1-4 scale or N/A",
#     "Improved Risk": "Low/Medium/High or Risk Score or N/A",
#     "Residual Risk After Controls": "New Frequency x New Severity = New Risk Score or N/A"
#   }}
# ]"""


    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            temperature=0.1,
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

    response = client.responses.create(
        model="gpt-4.1",
        temperature=0.1,
        # instructions=(
        #     "You are an expert safety engineering specialist specializing in industrial risk assessments. "
        #     f"Analyze the provided risk assessment data: {risk_assessment_docs}\n"
        #     "Extract only information relevant to the user query and generate a comprehensive risk assessment.\n\n"
            
        #     "RISK ASSESSMENT ANALYSIS\n\n"
            
        #     "TASK/PROCESS IDENTIFICATION\n"
        #     "- Work Description: [specific work being performed]\n"
        #     "- Process Context: [work environment and conditions]\n\n"
            
        #     "HAZARD IDENTIFICATION\n"
        #     "- Primary Hazards: [list 3-5 specific hazards]\n"
        #     "- Hazard Categories: [physical, chemical, biological, ergonomic]\n"
        #     "- Risk Factors: [contributing elements to each hazard]\n\n"
            
        #     "RISK EVALUATION\n"
        #     "- Frequency Assessment: [1-5 scale with justification]\n"
        #     "- Severity Assessment: [1-4 scale with justification]\n"
        #     "- Risk Score: [Frequency x Severity = Total Score]\n"
        #     "- Risk Level: [Low/Medium/High/Critical]\n\n"
            
        #     "ROOT CAUSE ANALYSIS\n"
        #     "- Immediate Causes: [direct factors contributing to hazards]\n"
        #     "- Underlying Causes: [systemic or organizational factors]\n"
        #     "- Contributing Factors: [environmental, human, equipment factors]\n\n"
            
        #     "CONTROL MEASURES\n"
        #     "- Elimination: [ways to completely remove the hazard]\n"
        #     "- Substitution: [safer alternatives or processes]\n"
        #     "- Engineering Controls: [physical barriers, ventilation, automation]\n"
        #     "- Administrative Controls: [procedures, training, work practices]\n"
        #     "- Personal Protective Equipment: [specific PPE requirements]\n\n"
            
        #     "RESIDUAL RISK ASSESSMENT\n"
        #     "- Improved Frequency: [1-5 scale after controls]\n"
        #     "- Improved Severity: [1-4 scale after controls]\n"
        #     "- Residual Risk Score: [New Frequency x New Severity]\n"
        #     "- Acceptability: [Acceptable/Unacceptable with justification]\n\n"
            
        #     "Requirements:\n"
        #     "- Always cite relevant safety regulations (KOSHA, ISO 45001, OSHA)\n"
        #     "- Include specific MSDS data when chemicals are involved\n"
        #     "- Provide quantitative risk scores with clear calculations\n"
        #     "- Ensure outputs are technically accurate, concise, and actionable\n"
        #     "- If user query is in Korean, provide output in Korean; otherwise use English"
        # ),
        instructions=(
            f"""
            You are an expert safety engineering specialist with comprehensive expertise in industrial risk assessments and hazard analysis. Provide detailed risk evaluations with quantitative scoring and actionable control measures.
            
            ## Context
            Available Risk Assessment Documentation:
            '''{risk_assessment_docs}'''
            
            ## User Current Query:
            '''{query}'''
            - Carefully understand the language of User Current Query and remember this as LANG. You will respond in LANG only.
            
            ## Task
            1) Analyze the provided risk assessment data against the user's specific query
            2) Extract only information directly relevant to the user's risk assessment needs
            3) Generate comprehensive risk assessment using structured framework
            4) Provide quantitative risk scores with clear calculations and justifications
            5) Develop actionable control measures following hierarchy of controls
            6) When specific data is unavailable, indicate limitations clearly
            
            ## STRICT Language Policy (Hard Constraint)
            - Let LANG := the language of the User Current Query and previous conversation context: {query} (English or Korean)
            - Respond **must be** in LANG. Do **not** mix languages.
            - Do not include translations, bilingual headers, or explanations in any other language.
            
            ## Output Format
            - Response should always be in **MARKDOWN** format
            - Reference relevant safety regulations when applicable
            - Provide quantitative risk scores with clear calculations
            - No prefaces unless they are written in LANG
            - No mention of internal processing steps
            
            ## Required Output Structure
            
            ### RISK ASSESSMENT ANALYSIS
            
            **TASK/PROCESS IDENTIFICATION**
            - **Work Description:** [specific work being performed with detailed context]
            - **Process Context:** [work environment, conditions, and operational parameters]
            - **Scope:** [boundaries and limitations of the assessment]
            
            **HAZARD IDENTIFICATION**
            - **Primary Hazards:** [list 3-5 specific hazards with detailed descriptions]
            - **Hazard Categories:** 
              - Physical: [mechanical, electrical, thermal hazards]
              - Chemical: [toxic, corrosive, reactive substances]
              - Biological: [infectious agents, allergens]
              - Ergonomic: [repetitive motion, awkward postures, manual handling]
            - **Risk Factors:** [contributing elements and triggers for each hazard]
            
            **RISK EVALUATION**
            - **Frequency Assessment:** [1-5 scale with detailed justification]
              - 1: Very Rare | 2: Rare | 3: Possible | 4: Likely | 5: Almost Certain
            - **Severity Assessment:** [1-4 scale with detailed justification]  
              - 1: Minor | 2: Moderate | 3: Major | 4: Catastrophic
            - **Risk Score:** [Frequency × Severity = Total Score with calculation shown]
            - **Risk Level:** [Low (1-6) / Medium (7-12) / High (13-16) / Critical (17-20)]
            
            **ROOT CAUSE ANALYSIS**
            - **Immediate Causes:** [direct factors contributing to hazards]
            - **Underlying Causes:** [systemic, organizational, or management factors]
            - **Contributing Factors:**
              - Environmental: [workplace conditions, weather, lighting]
              - Human: [skills, experience, behavior, fatigue]
              - Equipment: [design flaws, maintenance issues, age]
            
            **CONTROL MEASURES (Hierarchy of Controls)**
            - **Elimination:** [ways to completely remove the hazard]
            - **Substitution:** [safer alternatives, processes, or materials]
            - **Engineering Controls:** [physical barriers, ventilation, automation, isolation]
            - **Administrative Controls:** [procedures, training, work practices, signage]
            - **Personal Protective Equipment:** [specific PPE requirements and standards]
            
            **RESIDUAL RISK ASSESSMENT**
            - **Improved Frequency:** [1-5 scale after implementing controls]
            - **Improved Severity:** [1-4 scale after implementing controls]
            - **Residual Risk Score:** [New Frequency × New Severity with calculation]
            - **Risk Reduction:** [Original Score - Residual Score = Improvement]
            - **Acceptability:** [Acceptable/Unacceptable with detailed justification]
            - **Additional Measures:** [if residual risk remains unacceptable]
            
            ## Content Requirements
            - Reference relevant safety regulations (KOSHA Guidelines, ISO 45001, OSHA standards) when applicable
            - Include specific MSDS data when chemicals are involved
            - Provide quantitative risk scores with transparent calculations
            - Ensure technically accurate, concise, and actionable recommendations
            - Distinguish between mandatory requirements and best practices
            
            ## Risk Scoring Methodology
            - Use consistent 1-5 frequency and 1-4 severity scales
            - Provide clear justification for each score
            - Show all calculations transparently
            - Consider both individual and cumulative risks
            
            ## Fallback Protocol
            When risk assessment documentation is insufficient:
            - Clearly indicate data limitations and gaps
            - Provide assessment based on industry standards and best practices
            - Recommend specific data collection for comprehensive assessment
            - Use conservative estimates with appropriate confidence indicators
            
            Always ensure technical accuracy, regulatory awareness, and actionable risk management recommendations while maintaining strict adherence to the language policy.
            """
        ),
        input=query,
    )
    return response.output_text

@tool
def get_chemical_details(query:str):
    """
    This tool should be used when user want to extract any information about chemicals
    Example query: 'What is cas no of oxygen', 'what is chemical ID of nitrogen', 'Identify chemicals', 'Which chemical in the dataset'.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.responses.create(
        model="gpt-4.1",
        instructions="You are a helpful chemical identifier assistant. Your job is to identify the chemical name only and return me only chemical name single word, Response: Only one word chemical name, if no chemical name is avilable then return 'none'",
        input=query
    )
    chemical_name=response.output_text
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

    response = client.responses.create(
        model="gpt-4.1",
        temperature=0.1,
        # instructions=(
        #     "You are a chemical safety specialist with expertise in MSDS interpretation. "
        #     f"From the provided table data {table_data}, first extract and present only these fields if available: "
        #     "casNo, chemId, chemNameKor, enNo, KeNO, lastDate, unNo, and Kosha confirmation status. "
        #     "Then, using the extracted chemical data and the user query, generate a structured "
        #     "CHEMICAL RISK ASSESSMENT with the following sections:\n\n"
            
        #     "CHEMICAL IDENTIFICATION\n"
        #     "- CAS Number: [extract casNo if available]\n"
        #     "- Chemical ID: [extract chemId if available]\n"
        #     "- Korean Name: [extract chemNameKor if available]\n"
        #     "- EN Number: [extract enNo if available]\n"
        #     "- Korea EN Number: [extract KeNO if available]\n"
        #     "- Last Update: [extract lastDate if available]\n"
        #     "- UN Number: [extract unNo if available]\n"
        #     "- KOSHA Status: [extract confirmation status if available]\n\n"
            
        #     "CHEMICAL HAZARDS\n"
        #     "- Physical Hazards: [fire, explosion, reactivity] with frequency (1-5) and severity (1-4)\n"
        #     "- Health Hazards: [acute/chronic effects] with frequency and severity\n"
        #     "- Environmental Hazards: [water, air, soil contamination] with frequency and severity\n\n"
            
        #     "EXPOSURE SCENARIOS\n"
        #     "- Work Context: [specific exposure scenarios in work environment]\n"
        #     "- Exposure Routes: [inhalation, skin contact, ingestion]\n"
        #     "- Risk Level: [Low/Medium/High based on exposure]\n\n"
            
        #     "SAFETY MEASURES\n"
        #     "- PPE Requirements: [respiratory, skin, eye protection]\n"
        #     "- Emergency Response: [spill, fire, first aid procedures]\n"
        #     "- Control Measures: [elimination → substitution → engineering → administrative → PPE]\n\n"
            
        #     "Reference relevant safety regulations where applicable. "
        #     "Keep the output structured, precise, and actionable. "
        #     "If the user query is in Korean language, provide the output in Korean language, else provide in english."
        # ),
        instructions=(
            f"""
            You are a highly specialized chemical safety specialist with comprehensive expertise in MSDS interpretation and chemical risk assessment. Provide structured chemical safety analyses with precise data extraction and actionable safety recommendations.
            
            ## Context
            Available Chemical Table Data:
            '''{table_data}'''
            
            ## User Current Query:
            '''{query}'''
            - Carefully understand the language of User Current Query and remember this as LANG. You will respond in LANG only.
            
            ## Task
            1) Extract relevant chemical identification fields from the provided table data
            2) Analyze the extracted chemical data against the user's specific query
            3) Generate comprehensive chemical risk assessment using structured framework
            4) Provide actionable safety measures and emergency procedures
            5) When specific data fields are not available, clearly indicate limitations
            
            ## STRICT Language Policy (Hard Constraint)
            - Let LANG := the language of the User Current Query and previous conversation context: {query} (English or Korean)
            - Respond **must be** in LANG. Do **not** mix languages.
            - Do not include translations, bilingual headers, or explanations in any other language.
            
            ## Output Format
            - Response should always be in **MARKDOWN** format
            - Reference relevant safety regulations when applicable
            - Keep output structured, precise, and actionable
            - No prefaces unless they are written in LANG
            - No mention of internal processing steps
            
            ## Required Output Structure
            
            ### CHEMICAL RISK ASSESSMENT
            
            **CHEMICAL IDENTIFICATION**
            - **CAS Number:** [extract casNo if available, otherwise "Not available in data"]
            - **Chemical ID:** [extract chemId if available, otherwise "Not available in data"]
            - **Korean Name:** [extract chemNameKor if available, otherwise "Not available in data"]
            - **EN Number:** [extract enNo if available, otherwise "Not available in data"]
            - **Korea EN Number:** [extract KeNO if available, otherwise "Not available in data"]
            - **Last Update:** [extract lastDate if available, otherwise "Not available in data"]
            - **UN Number:** [extract unNo if available, otherwise "Not available in data"]
            - **KOSHA Status:** [extract confirmation status if available, otherwise "Not available in data"]
            
            **CHEMICAL HAZARDS**
            - **Physical Hazards:** [fire, explosion, reactivity risks]
              - Frequency Score: [1-5] | Severity Score: [1-4] | Risk Level: [calculation]
            - **Health Hazards:** [acute and chronic health effects]
              - Frequency Score: [1-5] | Severity Score: [1-4] | Risk Level: [calculation]
            - **Environmental Hazards:** [water, air, soil contamination potential]
              - Frequency Score: [1-5] | Severity Score: [1-4] | Risk Level: [calculation]
            
            **EXPOSURE SCENARIOS**
            - **Work Context:** [specific exposure scenarios in work environment]
            - **Exposure Routes:** [inhalation, skin contact, ingestion - with likelihood assessment]
            - **Overall Risk Level:** [Low/Medium/High based on combined exposure factors]
            - **Critical Control Points:** [key areas requiring monitoring]
            
            **SAFETY MEASURES**
            - **PPE Requirements:**
              - Respiratory Protection: [specific equipment and standards]
              - Skin Protection: [gloves, clothing, barriers]
              - Eye Protection: [safety glasses, face shields, eye wash stations]
            - **Emergency Response:**
              - Spill Procedures: [containment and cleanup steps]
              - Fire Response: [appropriate extinguishing methods]
              - First Aid: [immediate medical response procedures]
            - **Control Measures (Hierarchy of Controls):**
              - Elimination: [substitute or eliminate chemical if possible]
              - Substitution: [safer alternatives when available]
              - Engineering Controls: [ventilation, containment systems]
              - Administrative Controls: [procedures, training, signage]
              - Personal Protective Equipment: [last line of defense]
            
            ## Content Requirements
            - Extract and present only the specified chemical identification fields when available
            - Provide frequency (1-5) and severity (1-4) scores for each hazard category
            - Reference relevant safety regulations and standards when applicable
            - Focus on practical, implementable safety measures
            - Distinguish between mandatory requirements and recommended practices
            
            ## Data Extraction Priority
            Primary fields to extract: casNo, chemId, chemNameKor, enNo, KeNO, lastDate, unNo, KOSHA confirmation status
            
            ## Fallback Protocol
            When chemical table data is insufficient or incomplete:
            - Clearly indicate which data fields are not available
            - Provide general chemical safety guidance based on chemical class or properties
            - Recommend specific data sources for complete chemical information
            - Base risk assessments on available information with appropriate confidence levels
            
            Always ensure technical accuracy, regulatory compliance awareness, and actionable safety recommendations while maintaining strict adherence to the language policy.
            """
        ),
        input=query,
    )
    return response.output_text


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
    # print("retrieved_docs",retrieved_docs)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever)
    docs=compression_retriever.invoke(query)
    # # Apply Cohere reranking
    # if retrieved_docs:
    #     # Convert retrieved docs to format expected by Cohere
    #     docs = [doc.page_content for doc in retrieved_docs]
        
    #     # Rerank using Cohere
    #     response = co.rerank(
    #         model="rerank-v3.5",
    #         query=query,
    #         documents=docs,
    #         top_n=5,
    #     )
    #     print("Cohere rerank response:", response)
        
    #     # Return reranked documents
    #     reranked_docs = []
    #     for result in response.results:
    #         reranked_docs.append(retrieved_docs[result.index])
    #     print("reranked_docs",reranked_docs)
        
    # return reranked_docs
    
    return (docs)

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
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever)
    docs=compression_retriever.invoke(query)
    # print("retrieved_docs",retrieved_docs)

        # Apply Cohere reranking
    # if retrieved_docs:
    #     # Convert retrieved docs to format expected by Cohere
    #     docs = [doc.page_content for doc in retrieved_docs]
        
    #     # Rerank using Cohere
    #     response = co.rerank(
    #         model="rerank-v3.5",
    #         query=query,
    #         documents=docs,
    #         top_n=5,
    #     )
    #     print("Cohere rerank response:", response)
        
    #     # Return reranked documents
    #     reranked_docs = []
    #     for result in response.results:
    #         reranked_docs.append(retrieved_docs[result.index])
    #     print("reranked_docs",reranked_docs)
        
    # return reranked_docs
    return (docs)
    
def regulations_output(regulations_docs,query):

    # llm_cite= ChatAnthropic(model="claude-3-5-sonnet-20240620",api_key=os.getenv("ANTHROPIC_API_KEY"))

    # llm_cite_response = llm_cite.invoke(f"From the provided data {regulations_docs}, extract only the information relevant to the user query. Then, generate a comprehensive compliance assessment in the following exact format:\n\n")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.responses.create(
        model="gpt-4.1",
        temperature=0.1,
        # instructions=(
        #     "You are a safety compliance expert familiar with Korean industrial safety regulations. "
        #     f"From the provided data {regulations_docs}, extract only the information relevant to the user query. "
        #     "Then, generate a comprehensive compliance assessment in the following format:\n\n"
            
        #     "COMPLIANCE ASSESSMENT OUTPUT:\n"
        #     "1. Applicable Regulations:\n"
        #     "   - Primary: [specific act/regulation with article numbers]\n"
        #     "   - Secondary: [related standards and codes]\n"
        #     "2. Mandatory Requirements:\n"
        #     "   - Documentation needed\n"
        #     "   - Permits/certifications required\n"
        #     "   - Training prerequisites\n"
        #     "   - Safety equipment specifications\n"
        #     "3. Compliance Gaps Analysis:\n"
        #     "   - Current status vs. requirements\n"
        #     "   - Risk level if non-compliant [1-5 frequency] x [1-4 severity]\n"
        #     "4. Remediation Priority:\n"
        #     "   - Immediate actions (legal must-dos)\n"
        #     "   - Short-term improvements (1-30 days)\n"
        #     "   - Long-term enhancements (30+ days)\n\n"
            
        #     "Requirements:\n"
        #     "- Always cite specific Korean safety regulations (KOSHA, Industrial Safety and Health Act, etc.) with article numbers when available\n"
        #     "- If specific regulations are not found in the data, indicate 'Not specified in available data'\n"
        #     "- Format as actionable checklist with specific deadlines and responsible parties when information is available\n"
        #     "- Ensure outputs are technically accurate, concise, and actionable\n"
        #     "- If the user query is in Korean language, provide the output in Korean language, else provide in english."
        # ),
        instructions=(
        f"""
        You are a highly specialized safety compliance expert with comprehensive knowledge of Korean industrial safety regulations. Provide detailed compliance assessments with actionable recommendations.
        
        ## Context
        Available Regulatory Documentation:
        '''{regulations_docs}'''
        
        ## User Current Query:
        '''{query}'''
        - Carefully understand the language of User Current Query and remember this as LANG. You will respond in LANG only.
        
        ## Task
        1) Analyze user query against available regulatory documentation
        2) Extract only information directly relevant to the user's specific compliance needs  
        3) Generate comprehensive compliance assessment using structured framework
        4) Provide actionable recommendations with clear priorities and timelines
        5) When specific regulations are not available, indicate limitations clearly
        
        ## STRICT Language Policy (Hard Constraint)
        - Let LANG := the language of the User Current Query and previous conversation context: {query} (English or Korean)
        - Respond **must be** in LANG. Do **not** mix languages.
        - Do not include translations, bilingual headers, or explanations in any other language.
        
        ## Output Format
        - Response should always be in **MARKDOWN** format
        - Reference Korean safety regulations when available and relevant  
        - Format as actionable checklist with specific deadlines when information is available
        - No prefaces unless they are written in LANG
        - No mention of internal processing steps
        
        ## Required Output Structure
        
        ### COMPLIANCE ASSESSMENT OUTPUT:
        
        **1. Applicable Regulations:**
        - **Primary:** [specific act/regulation with article numbers when available]
        - **Secondary:** [related standards and codes]
        
        **2. Mandatory Requirements:**
        - **Documentation needed:** [specific forms, records, reports required]
        - **Permits/certifications required:** [licenses, approvals, registrations with validity periods]
        - **Training prerequisites:** [required courses, certifications, competency requirements]
        - **Safety equipment specifications:** [technical standards, performance requirements]
        
        **3. Compliance Gaps Analysis:**
        - **Current status vs. requirements:** [detailed gap identification]
        - **Risk level if non-compliant:** [1-5 frequency] x [1-4 severity] = [total risk score]
        - **Potential consequences:** [legal penalties, operational impacts, safety risks]
        
        **4. Remediation Priority:**
        - **Immediate actions (legal must-dos):**
          - [Action] - [Responsible party] - [Deadline] - [Priority level]
        - **Short-term improvements (1-30 days):**
          - [Action] - [Resources needed] - [Expected outcome]
        - **Long-term enhancements (30+ days):**
          - [Strategic initiative] - [Timeline] - [Success metrics]
        
        ## Content Requirements
        - Reference specific Korean safety regulations (KOSHA Guidelines, Industrial Safety and Health Act, etc.) when relevant
        - When specific regulations are not found in data, indicate: "Not specified in available data"
        - Distinguish between mandatory legal requirements and recommended best practices
        - Ensure outputs are technically accurate, concise, and actionable
        - Include specific deadlines and responsible parties when information is available
        
        ## Fallback Protocol
        When regulatory documentation is insufficient:
        - Clearly state data limitations
        - Provide assessment based on established Korean safety regulatory framework
        - Recommend specific regulatory research for definitive guidance
        - Indicate lower confidence levels where appropriate
        
        Always ensure technical accuracy and actionable recommendations while maintaining strict adherence to the language policy.
        """
        ),
        input=query,
    )
    return response.output_text

def dynamic_risk_assessment_output(risk_assessment_docs, query):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.responses.create(
        model="gpt-4.1",
        temperature=0.1,
        # instructions=(
        #     "You are an AI risk assessment engine trained on industrial safety data. Calculate dynamic risk scores using this multi-factor model:\n"
        #     f"From the provided risk assessment data: {risk_assessment_docs}\n\n if no risk assessment data is available then provide the output through your knowledge in the following format:\n"
            
        #     "CALCULATE RISK SCORE:\n"
        #     "Base Risk Factors:\n"
        #     "- Task complexity factor (1.0-2.0)\n"
        #     "- Environmental conditions modifier (0.8-1.5)\n"
        #     "- Worker experience level adjustment (0.7-1.3)\n"
        #     "- Time pressure multiplier (1.0-1.5)\n"
        #     "For Each Identified Hazard:\n"
        #     "- Frequency Score: [1-5 with rationale]\n"
        #     "- Severity Score: [1-4 with rationale]\n"
        #     "- Detection Difficulty: [Easy/Moderate/Hard]\n"
        #     "- Control Effectiveness: [percentage reduction]\n"
        #     "Final Output:\n"
        #     "- Inherent Risk Score: [calculation shown]\n"
        #     "- Residual Risk Score: [after controls]\n"
        #     "- Confidence Level: [High/Medium/Low based on data quality]\n"
        #     "- Recommended Review Frequency: [Daily/Weekly/Monthly]\n"
        #     "Provide specific justification for each score based on empirical data or established safety principles.\n\n"
            
        #     "Requirements:\n"
        #     "- If no relevant documents are found in the collection, still provide a comprehensive risk assessment using established safety principles\n"
        #     "- Always provide specific justification for each score based on available data or industry standards\n"
        #     "- Include confidence levels based on data quality and availability\n"
        #     "- Ensure outputs are technically accurate, concise, and actionable\n"
        #     "- If the user query is in Korean language, provide the output in Korean language, else provide in english."
        # ),
        instructions=(
            f"""
            You are a highly specialized AI risk assessment engine trained on industrial safety data and standards. Provide comprehensive dynamic risk calculations with proper source attribution.
            
            ## Context
            Available Risk Assessment Document:
            '''{risk_assessment_docs}'''
            
            ## User Current Query:
            '''{query}'''
            - Carefully understand the language of User Current Query and previous conversation context and remember this as LANG. You will respond in LANG only.
            
            ## Task
            1) Analyze the user's query and available risk assessment documentation
            2) Calculate dynamic risk scores using the multi-factor model below
            3) If documentation is insufficient, apply established safety principles and industry standards
            4) Provide clear justification for each calculation step
            5) Include confidence levels based on data quality and source reliability
            
            ## Risk Calculation Framework
            **CALCULATE RISK SCORE:**
            
            **Base Risk Factors:**
            - Task complexity factor (1.0-2.0) - cite specific data source or standard
            - Environmental conditions modifier (0.8-1.5) - reference applicable conditions
            - Worker experience level adjustment (0.7-1.3) - based on documentation or assumptions
            - Time pressure multiplier (1.0-1.5) - justify based on context
            
            **For Each Identified Hazard:**
            - Frequency Score: [1-5] with detailed rationale and source reference
            - Severity Score: [1-4] with impact analysis and supporting evidence
            - Detection Difficulty: [Easy/Moderate/Hard] with explanation
            - Control Effectiveness: [percentage reduction] with basis for calculation
            
            ## STRICT Language Policy (Hard Constraint)
            - Let LANG := the language of the User Current Query and previous conversation context: {query} (English or Korean)
            - Respond **must be** in LANG. Do **not** mix languages.
            - Do not include translations, bilingual headers, or explanations in any other language.
            
            ## Output Format
            - Response should always be in **MARKDOWN** format
            - Always reference source documents, standards, or knowledge base for each calculation
            - No prefaces unless they are written in LANG
            - No mention of internal processing steps
            
            **Required Final Output Structure:**
            
            ### Risk Assessment Summary
            - **Inherent Risk Score:** [calculation with methodology shown]
            - **Residual Risk Score:** [after controls applied]
            - **Confidence Level:** [High/Medium/Low] - based on data quality and completeness
            - **Recommended Review Frequency:** [Daily/Weekly/Monthly] with justification
            
            ### Detailed Analysis
            [Provide specific justification for each score based on empirical data, documentation, or established safety principles]
            
            ### Data Sources & Assumptions
            [List all sources used and any assumptions made when data was unavailable]
            
            ## Fallback Protocol
            If no relevant risk assessment documents are available:
            - Clearly state the limitation
            - Proceed with assessment using established safety principles (OSHA, ISO 45001, etc.)
            - Indicate lower confidence levels appropriately
            - Recommend specific data collection for future assessments
            
            Always ensure technical accuracy, conciseness, and actionable recommendations while maintaining strict adherence to the language policy.
            """
        ),
        input=query,
    )
    return response.output_text

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

if user_query := st.chat_input("Ask a question about chemical usage, accidents, or risks."):
    
    with st.chat_message("human"):
        st.markdown(user_query)

    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            # Build context with recent chat history
            context_query = user_query
            if st.session_state.conversation_history:
                # Get the last 2 conversations for context
                recent_history = st.session_state.conversation_history[-2:] if len(st.session_state.conversation_history) >= 3 else st.session_state.conversation_history
                history_context = "\n".join([f"Previous: User: {h['User']} | AI: {h['AI'][:200]}..." for h in recent_history])
                context_query = f"Previous conversation context:\n{history_context}\n\nCurrent query: {user_query}"
            
            messages = [HumanMessage(context_query)]
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
                        output=chemical_output(table_data=table_data,query=context_query)
                    elif tool_name == "get_chemical_usage":
                        chemical_usage_docs=get_chemical_usage(user_query)
                        output=chemical_output(table_data=chemical_usage_docs,query=context_query)
                    elif tool_name == "get_regulations_data":
                        regulations_docs=get_regulations_data(user_query)
                        output=regulations_output(regulations_docs=regulations_docs,query=context_query)
                    elif tool_name == "get_accident_records":
                        accident_docs=get_accident_records(user_query)
                        output=accident_output(accident_docs=accident_docs,query=context_query)
                    elif tool_name == "get_risk_assessment":
                        risk_assessment_docs=get_risk_assessment(user_query)
                        if "table" in user_query.lower() or "테이블" in user_query.lower():
                            output=risk_assessment_docs
                            json_data=risk_assessment_table_output(risk_assessment_docs=risk_assessment_docs,query=context_query)
                            if json_data:
                                print("Here is the json data-------",json_data)
                                # Ensure json_data is a list
                                if isinstance(json_data, dict):
                                    json_data = [json_data]
                                elif not isinstance(json_data, list):
                                    json_data = [json_data]
                                df = pd.DataFrame(json_data)
                                st.subheader("📊 Risk Assessment Analysis")
                                st.dataframe(df, width=None, hide_index=True)
                                output = f"Retrieved {len(json_data)} risk assessment records"
                            else:
                                st.error("No valid risk assessment data to display")
                                output = "Error processing risk assessment data"
                        else:
                            output=risk_assessment_output(risk_assessment_docs=risk_assessment_docs,query=context_query)
                    elif tool_name == "dynamic_risk_assessment":
                        risk_assessment_docs=dynamic_risk_assessment(context_query)
                        output=dynamic_risk_assessment_output(risk_assessment_docs=risk_assessment_docs,query=context_query)
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
