import os
import streamlit as st
from qdrant_client import models
import qdrant_client
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0,api_key=os.getenv("OPENAI_API_KEY"))


if "client" not in st.session_state:
    st.session_state.client = qdrant_client.QdrantClient(url=os.getenv("QDRANT_URL"),api_key=os.getenv("QDRANT_API_KEY"),port=None)

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

tool_dict = {"get_accident_records":get_accident_records,"get_chemical_usage":get_chemical_usage,"get_risk_assessment":get_risk_assessment}
tools = [get_accident_records,get_chemical_usage,get_risk_assessment]
llm_with_tools = llm.bind_tools(tools=tools)

st.title("RAG Chatbot for Company Documents 📄")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    if message.type == 'tool':
        with st.expander(label=message.tool_call_id,icon='📖'):
            st.write(message.content)
        continue
    with st.chat_message(message.type):
        st.markdown(message.content)

if user_query := st.chat_input("Ask a question about chemical usage, accidents, or risks."):
    
    with st.chat_message("human"):
        st.markdown(user_query)

    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            messages = [HumanMessage(user_query)]
            ai_msg = llm_with_tools.invoke(messages)
            messages.append(ai_msg)
            while (ai_msg.tool_calls):
                for tool_call in ai_msg.tool_calls:
                    selected_tool = tool_dict[tool_call["name"].lower()]
                    tool_msg = selected_tool.invoke(tool_call)
                    messages.append(tool_msg)
                    with st.expander(tool_msg.tool_call_id,icon='📖'):
                        st.write(tool_msg.content)
                ai_msg = llm_with_tools.invoke(messages)
                messages.append(ai_msg)

            answer = messages[-1].content
            st.markdown(answer)
            st.session_state.chat_history.extend(messages)