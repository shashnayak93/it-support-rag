from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunks():
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

    return chunks