import os
from getpass import getpass

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

print("CLAUDE_API_KEY configured: ", bool(os.getenv("CLAUDE_API_KEY")))
print("TAVILY_API_KEY configured: ", bool(os.getenv("TAVILY_API_KEY")))
print("PINECONE_API_KEY configured: ", bool(os.getenv("PINECONE_API_KEY")))