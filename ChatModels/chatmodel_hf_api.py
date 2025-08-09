import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Load environment variables
load_dotenv()

# Get the Hugging Face API key
hf_api_key = os.getenv("HF_API_KEY")
if not hf_api_key:
    raise ValueError("Hugging Face API key not found. Please set HF_API_KEY in your .env file.")

# Create the underlying LLM endpoint
llm_endpoint = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",  # Change to your model
    task="text-generation",
    huggingfacehub_api_token=hf_api_key
)

# Wrap in ChatHuggingFace
model = ChatHuggingFace(llm=llm_endpoint)

# Run a test query
result = model.invoke("What is the capital of India?")
print(result)
