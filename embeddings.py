from langchain_huggingface import HuggingFaceEmbeddings

def embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs = {"normalize_embeddings": True}
    )

    sample_vector = embeddings.embed_query("What is Agentic RAG?")
    print("Embedding dimensions: ", len(sample_vector))
    return embeddings