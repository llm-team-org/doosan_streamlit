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
load_dotenv(dotenv_path=".env")

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

                    #tool_msg = selected_tool.invoke(tool_call)
                #     messages.append(tool_msg)
                #     with st.expander(tool_msg.tool_call_id,icon='📖'):
                #         st.write(tool_msg.content)
                # ai_msg = llm_with_tools.invoke(messages)
                messages.append(output)

            answer = output
            st.markdown(answer)
            st.session_state.chat_history.extend(messages)
