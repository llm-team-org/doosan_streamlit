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
load_dotenv()

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0,api_key=os.getenv("OPENAI_API_KEY"))


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
    Retrieves information from company documents.
    This tool should be used to answer questions about accident records.
    The search query must be in Korean.
    Example query: '화학 물질 사용', '사고 기록', '위험 평가'.
    """
    retrieved_docs = st.session_state.client.query_points(
        collection_name="doosan_accident_records",
        query=models.Document(
            text=query,
            model="Qdrant/minicoil-v1"
        ),
        using="minicoil",
        limit=5
    )
    docs = [doc.payload['text'] for doc in retrieved_docs.points]
    return "\n\n".join(docs)

def accident_output(accident_docs,query):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.responses.create(
            model="gpt-4.1-nano",
            instructions=f"You are a helpful assistant.From this data give the user information {accident_docs}. From the collected docs table carefully extract the accident records relevant to the query entered by the user",
            input=query,
        )
    return response.output_text

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
        query=models.Document(
            text=query,
            model="Qdrant/minicoil-v1"
        ),
        using="minicoil",
        limit=5
    )
    docs = [doc.payload['text'] for doc in retrieved_docs.points]
    return "\n\n".join(docs)

@tool
def get_risk_assessment(query: str) -> str:
    """
    Retrieves information from company documents.
    This tool should be used to answer questions about chemical usage.
    The search query must be in Korean.
    Example query: '화학 물질 사용', '사고 기록', '위험 평가'.
    """
    retrieved_docs = st.session_state.client.query_points(
        collection_name="doosan_risk_assessment",
        query=models.Document(
            text=query,
            model="Qdrant/minicoil-v1"
        ),
        using="minicoil",
        limit=5
    )
    docs = [doc.payload['text'] for doc in retrieved_docs.points]
    return "\n\n".join(docs)

def risk_assessment_output(risk_assessment_docs, query):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Create structured prompt for JSON output
    json_instructions = f"""
You are a Risk Assessment Assistant. Your task is to analyze the provided risk assessment documents: {risk_assessment_docs}

Instructions:
- Always return the output in strict JSON format.
- Use the predefined field names exactly as listed below.
- If data cannot be found, set the field value to "N/A".
- If no structured data is available at all, return a JSON object with all fields set to "N/A".
- Do not include explanations, extra text, or markdown—only JSON.

Output Format:
[
  {{
    "Hazard Factor": "Description of the hazard",
    "Current Frequency": "Low/Medium/High or N/A",
    "Current Severity": "Low/Medium/High or N/A",
    "Current Risk": "Low/Medium/High or N/A",
    "Current Measures": "Existing safety measures or N/A",
    "Reduction Measures": "Proposed reduction measures or N/A",
    "Improved Frequency": "Low/Medium/High or N/A",
    "Improved Severity": "Low/Medium/High or N/A",
    "Improved Risk": "Low/Medium/High or N/A"
  }}
]"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
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
            "Hazard Factor": "Error parsing data",
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
def get_chemical_details(query:str):
    """
    This tool should be used when user want to extract any information about chemicals
    Example query: 'What is cas no of oxygen', 'what is chemical ID of nitrogen'.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.responses.create(
        model="gpt-4.1-nano",
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

def chemical_output(table_data,query):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.responses.create(
            model="gpt-4.1-nano",
            instructions=f"You are a helpful assistant.From this data give the user information {table_data}. From this table carefully extract casNo,chemId,chemNameKor,enNo, KeNO,lastDate,unNo and is it Kosha Confirmed or not for the given Chemimal name entered by the user",
            input=query,
        )
    return response.output_text

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
        query=models.Document(
            text=query,
            model="Qdrant/minicoil-v1"
        ),
        using="minicoil",
        limit=5
    )
    docs = [doc.payload['text'] for doc in retrieved_docs.points]
    return "\n\n".join(docs)

def regulations_output(regulations_docs,query):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.responses.create(
            model="gpt-4.1-nano",
            instructions=f"You are a helpful assistant.From this data give the user information {regulations_docs}. From the collected docs table carefully extract the relevants rules relevant to the query entered by the user",
            input=query,
        )
    return response.output_text

tool_dict = {"get_accident_records":get_accident_records,"get_chemical_usage":get_chemical_usage,"get_risk_assessment":get_risk_assessment,"get_regulations_data":get_regulations_data,"get_chemical_details":get_chemical_details}
tools = [get_accident_records,get_chemical_usage,get_risk_assessment,get_regulations_data,get_chemical_details]
llm_with_tools = llm.bind_tools(tools=tools)

st.title("DOOSAN RISK MANAGEMENT AI Chatbot 📄")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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
            messages = [HumanMessage(user_query)]
            ai_msg = llm_with_tools.invoke(messages)
            print(ai_msg)
            messages.append(ai_msg)
            #while (ai_msg.tool_calls):
            for tool_call in ai_msg.tool_calls:
                selected_tool = tool_dict[tool_call["name"].lower()]
                print("Here is the selected tool-------",selected_tool)
                tool_name= selected_tool.name
                print("Here is the selected tool name -------",tool_name)
                if tool_name == "get_chemical_details":
                    table_data=get_chemical_details(user_query)
                    output=chemical_output(table_data=table_data,query=user_query)
                elif tool_name == "get_regulations_data":
                    regulations_docs=get_regulations_data(user_query)
                    output=regulations_output(regulations_docs=regulations_docs,query=user_query)
                elif tool_name == "get_accident_records":
                    accident_docs=get_accident_records(user_query)
                    output=accident_output(accident_docs=accident_docs,query=user_query)
                elif tool_name == "get_risk_assessment":
                    risk_assessment_docs=get_risk_assessment(user_query)
                    json_data=risk_assessment_output(risk_assessment_docs=risk_assessment_docs,query=user_query)
                    
                    # Construct and display table
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
                

                    #tool_msg = selected_tool.invoke(tool_call)
                #     messages.append(tool_msg)
                #     with st.expander(tool_msg.tool_call_id,icon='📖'):
                #         st.write(tool_msg.content)
                # ai_msg = llm_with_tools.invoke(messages)
                messages.append(output)

            answer = output
            st.markdown(answer)
            st.session_state.chat_history.extend(messages)
