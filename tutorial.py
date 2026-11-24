
import os
from getpass import getpass
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from typing import List, Literal, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.documents import Document

import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

if not os.getenv("ANTHROPIC_API_KEY"):
    os.environ["ANTHROPIC_API_KEY"] = getpass("Enter ANTHROPIC_API_KEY: ")

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

# print("Existing Pinecone index loaded:", INDEX_NAME)

test_question = "What happens if retrieved documents are not relevant in Agentic RAG?"

kb_docs = retriever.invoke(test_question)

# for i, doc in enumerate(kb_docs,1):
#     print(f'\n---KB Result {i}---')
#     print("Source: ", doc.metadata.get("source"))
#     print(doc.page_content[:700])

llm = ChatAnthropic(
    model="claude-sonnet-5-5",
    max_tokens = 4096
)

messages = [
    ("human", "Explain Agentic RAG in one sentence."),
]

response = llm.invoke(messages)

# print(response.content)

web_search = TavilySearch(
    max_search = 5,
    topic = 'general',
    include_answer = True,
    include_raw_content = False
)

class RouteDecision(BaseModel):
    route: Literal["kb","direct"] = Field(
         description="Use kb for questions needing Agentic RAG docs; direct for greetings/simple chat."
    )

class EvidenceGrade(BaseModel):
    grade: Literal["good", "weak"] = Field(
        description="good means evidence can answer the question; weak means not enough evidence."
    )

class AgentState(TypedDict):
    question: str
    current_query: str
    kb_docs: List[Document]
    web_results: str
    kb_grade: str
    answer: str
    source_used: str
    retry_count: str


#NODE 1 - Route the Question
router_llm = llm.with_structured_output(RouteDecision, method="json_schema")

def route_question(state: AgentState):
    question = state["question"]

    decision = router_llm.invoke(f'''
    You are a router for an Agentic RAG assistant.

    Route to "kb" if the user asks about:
    - Agentic RAG
    - LangGraph Agentic RAG workflow
    - retrieval grading
    - query rewriting
    - RAG architecture
    - retriever tools
    - web fallback in RAG

    Route to "direct" only for greetings, thanks, or very simple conversation.

    Question:
    {question}

    Return your response as valid JSON.
    Example:
    {{"route": "kb"}}
    ''')
    print("[Router]", decision.route)

    return {
        "current_query": question,
        "source_used": decision.route
    }

#NODE 2 - Retrieve the KB
def route_after_router(state: AgentState) -> Literal["retrieve_kb","direct_answer"]:
    if state["source_used"] == "kb":
        return "retrieve_kb"
    else:
        return "direct_answer"

def retrieve_kb(state: AgentState):
    query = state["current_query"]
    docs = retriever.invoke(query)
    print(f"[KB Retriever] Query: {query}")
    print(f"[KB Retriever] Retrieved: {len(docs)} chunks")

    return {"kb_docs": docs}

#NODE 3 - Grade Private KB Evidence

kb_grader_llm = llm.with_structured_output(EvidenceGrade,method="json_schema")

def grade_kb_evidence(state:AgentState):
    question = state["question"]

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source')}\n{doc.page_content}"
        for doc in state["kc_docs"])

    grade = kb_grader_llm.invoke(f'''
You are an evidence grader.

Question:
{question}

Private KB evidence:
{context}

Can this private KB evidence answer the question?
Return "good" if it can answer.
Return "weak" if it cannot answer or is incomplete.

Return your response as valid JSON.
Example:
{{"grade": "good"}}
''')

    print("[KB Grader]", grade.grade)
    return {"kb_grade": grade.grade}

def decide_after_kb_grade(state: AgentState) -> Literal["generate_from_kb","search_web"]:
    if state["kb_grade"] == "good":
        return "generate_from_kb"
    else:
        return "search_web"

#NODE 4 - Tavily Web Search Fallback
def search_web(state:AgentState):
    query = state["current_query"]

    print(f"Tavily Search Query: {query}")

    result = web_search.invoke({"query": query})

    # Tavily can return dict/list structures depending on package version.
    # Convert it into readable text for grading and generation.

    if isinstance(result,dict):
        answer = result.get("answer","")
        results = result.get("results","")
        lines = []
        if answer:
            lines.append(f"Tavily answer: {answer}")

        for item in results:
            title = item.get("title","")
            url = item.get("url","")
            content = item.get("content","")
            lines.append(f"Title: {title}\nURL: {url}\nContent: {content}")

        web_text = "\n\n".join(lines) if lines else str(result)
    else:
        web_text = str(result)

    print("[Tavily Search] Results characters:", len(web_text))

    return{
        "web_results":web_text,
        "source_used": "web"
    }

#NODE 5 - Grade the Web Evidence

web_grader_llm = llm.with_structured_output(EvidenceGrade, method="json_schema")

def grade_web_evidence(state: AgentState):
    question = state["question"]
    web_results = state["web_results"]

    grade = web_grader_llm.invoke(f'''
You are an evidence grader.

Question:
{question}

Web search evidence:
{web_results}

Can this web evidence answer the question?
Return "good" if it can answer.
Return "weak" if it cannot answer or is incomplete.

Return your response as valid JSON.
Example:
{{"grade": "good"}}
''')

    print("[Web Grader]", grade.grade)

    return {"web_grade": grade.grade}


MAX_RETRIES = 1

def decide_after_web_grade(state:AgentState) -> Literal["generate_from_web","rewrite_query", "answer_insufficient"]:
    if state["web_grade"] == "good":
        return "generate_from_web"

    if state["retry_count"] < MAX_RETRIES:
        return "rewrite_query"

    return "answer_insufficient"
