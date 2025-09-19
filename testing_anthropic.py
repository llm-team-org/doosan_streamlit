from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_qdrant import QdrantVectorStore
import os
import cohere
import streamlit as st
from qdrant_client import models
import qdrant_client
import xml.etree.ElementTree as ET
from tabulate import tabulate
import requests
from openai import OpenAI
import os
import cohere
from dotenv import load_dotenv
load_dotenv()

co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

llm = ChatAnthropic(model="claude-3-7-sonnet-latest", api_key=os.getenv("ANTHROPIC_API_KEY"))
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-large")



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
    # print("retrieved_docs",retrieved_docs)

        # Apply Cohere reranking
    if retrieved_docs:
        # Convert retrieved docs to format expected by Cohere
        docs = [doc.page_content for doc in retrieved_docs]
        
        # Rerank using Cohere
        response = co.rerank(
            model="rerank-v3.5",
            query=query,
            documents=docs,
            top_n=5,
        )
        print("Cohere rerank response:", response)
        
        # Return reranked documents
        reranked_docs = []
        for result in response.results:
            reranked_docs.append(retrieved_docs[result.index])
        #print("reranked_docs",reranked_docs)
        
    return reranked_docs

query="Give me the accident records for year 2023"
docs=get_accident_records(query)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "document",
                "source": {
                    "type": "text",
                    "media_type": "text/plain",
                    "data": str(docs),
                },
                "title": "My Document",
                "context": "This is a trustworthy document.",
                "citations": {"enabled": True},
            },
            {"type": "text", "text": query},
        ],
    }
]
response = llm.invoke(messages)
#response.content
print("response is:",response.content)
print("citetations are:",response.response_metadata.get('citations'))