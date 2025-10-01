from qdrant_client import models
import qdrant_client

QDRANT_URL="https://doosan-qdrant-dev.doaz.ai"
QDRANT_API_KEY="9A0G+KixvcB2Py9TkaXYGoNCX7jnxAY9ty/qsuE7wcA="

client=qdrant_client.QdrantClient(url=QDRANT_URL,api_key=QDRANT_API_KEY,port=None)
qdrant = client.query_points(
    collection_name="doosan_regulations_data",
    query=models.Document(
        text="What is risk management",
        model="Qdrant/minicoil-v1"
    ),
    using="minicoil",
    limit=5
)
# docs = [doc.payload['text'] for doc in qdrant.points]
# # retriever = qdrant.as_retriever(search_kwargs={"k": 15})
# # ans=retriever.invoke("What is risk management")
# print(docs)

for doc in qdrant.points:
    print(doc.payload['text'])