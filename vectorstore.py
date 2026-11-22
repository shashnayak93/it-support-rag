
import os
from getpass import getpass
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

if not os.getenv("CLAUDE_API_KEY"):
    os.environ["CLAUDE_API_KEY"] = getpass("Enter CLAUDE_API_KEY: ")

if not os.getenv("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = getpass("Enter TAVILY_API_KEY: ")

if not os.getenv("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = getpass("Enter PINECONE_API_KEY: ")

# print("CLAUDE_API_KEY configured: ", bool(os.getenv("CLAUDE_API_KEY")))
# print("TAVILY_API_KEY configured: ", bool(os.getenv("TAVILY_API_KEY")))
# print("PINECONE_API_KEY configured: ", bool(os.getenv("PINECONE_API_KEY")))

SOURCE_URL = "https://docs.langchain.com/oss/python/langgraph/agentic-rag"

loader = WebBaseLoader(
    web_paths=(SOURCE_URL,),
    requests_kwargs={
        'headers': {
            "User-Agent": "Mozilla/5.0 Agentic-RAG-Industry-Demo"
        }
    }
)

raw_docs = loader.load()

# print("Loaded documents: ", len(raw_docs))
# print("Source: ",raw_docs[0].metadata.get("source"))
# print("\nPreview:\n")
# print(raw_docs[0].page_content[:1500])

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 150,
    add_start_index = True
)

chunks = splitter.split_documents(raw_docs)

# print("Total chunks: ", len(chunks))
# print("\nFirst Chunk Preview\n")
# print(chunks[0].page_content[:900])

embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs = {"normalize_embeddings": True}
)

sample_vector = embeddings.embed_query("What is Agentic RAG?")
# print("Embedding dimensions: ", len(sample_vector))


INDEX_NAME = "industry-agentic-rag-kb"
NAMESPACE = "langgraph-agentic-rag"

# # Connect to Pinecone
# pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# # Create the index only if it does not already exist.
# existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

# if INDEX_NAME not in existing_indexes:
#     pc.create_index(
#         name=INDEX_NAME,
#         dimension=384,       # all-MiniLM-L6-v2 embedding dimension
#         metric="cosine",
#         spec=ServerlessSpec(
#             cloud="aws",
#             region="us-east-1",
#         ),
#     )

#     while not pc.describe_index(INDEX_NAME).status["ready"]:
#         time.sleep(1)

# # print("Pinecone index ready: ", INDEX_NAME)

# vectorstore = PineconeVectorStore.from_documents(
#     documents = chunks,
#     embedding = embeddings,
#     index_name = INDEX_NAME,
#     namespace = NAMESPACE,
# )

# retriever = vectorstore.as_retriever(
#     search_kwargs = {
#         "k": 4,
#         "namespace": NAMESPACE,
#     }
# )

# print("Pinecone vector database and retriever are ready.")

# Connect to Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Load existing Pinecone index
index = pc.Index(INDEX_NAME)

# Connect existing index with LangChain
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    namespace=NAMESPACE,
)

# Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 4,
        "namespace": NAMESPACE,
    }
)

print("Existing Pinecone index loaded:", INDEX_NAME)