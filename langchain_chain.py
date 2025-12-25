"""
LangChain + Gemini Chain Example
Demonstrates a simple chain using LangChain's ChatPromptTemplate and ChatGoogleGenerativeAI
"""

from dotenv import load_dotenv
load_dotenv()

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Read API key from environment
api_key = os.getenv("GENAI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise SystemExit("GENAI_API_KEY or GOOGLE_API_KEY environment variable is required.")

# Initialize the LangChain Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=api_key,
    temperature=0.7
)

# Create a prompt template
prompt_template = ChatPromptTemplate.from_template(
    """You are a helpful AI assistant. Answer the following question concisely.

Question: {question}

Answer:"""
)

# Create the chain: prompt -> model -> output parser
chain = prompt_template | model | StrOutputParser()

# Example: Run the chain
if __name__ == "__main__":
    # Test 1: Simple question
    print("=" * 60)
    print("Test 1: Simple Question")
    print("=" * 60)
    question1 = "What is machine learning?"
    result1 = chain.invoke({"question": question1})
    print(f"Q: {question1}")
    print(f"A: {result1}\n")
    
    # Test 2: Another question
    print("=" * 60)
    print("Test 2: Another Question")
    print("=" * 60)
    question2 = "Explain neural networks in one sentence."
    result2 = chain.invoke({"question": question2})
    print(f"Q: {question2}")
    print(f"A: {result2}\n")
    
    # Test 3: Code-related question
    print("=" * 60)
    print("Test 3: Code Question")
    print("=" * 60)
    question3 = "What is a Python lambda function?"
    result3 = chain.invoke({"question": question3})
    print(f"Q: {question3}")
    print(f"A: {result3}\n")
