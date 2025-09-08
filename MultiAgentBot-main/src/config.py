import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from crewai import LLM

load_dotenv()
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_API_BASE"] = "https://nandu899.openai.azure.com/"
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
os.environ["OPENAI_DEPLOYMENT_NAME"] = "gpt-35-turbo"

def get_llm():
    return LLM(
    model="azure/gpt-35-turbo",   # 👈 Notice the "azure/" prefix (very important)
    api_key=os.environ["OPENAI_API_KEY"],
    api_base=os.environ["OPENAI_API_BASE"],
    api_version=os.environ["OPENAI_API_VERSION"],
    temperature=0.3,
    max_tokens=1000
)
