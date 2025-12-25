"""
LangChain Chain Utilities for Gemini API
Provides reusable chain patterns: Q&A, Summarization, Code Generation, etc.
"""

from dotenv import load_dotenv
load_dotenv()

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize the model
api_key = os.getenv("GENAI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise SystemExit("GENAI_API_KEY or GOOGLE_API_KEY environment variable is required.")

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=api_key,
    temperature=0.7
)


class GeminiChains:
    """Collection of reusable LangChain chains with Gemini API."""
    
    @staticmethod
    def qa_chain():
        """Simple Q&A chain."""
        prompt = ChatPromptTemplate.from_template(
            """You are a helpful AI assistant. Answer the following question concisely.

Question: {question}

Answer:"""
        )
        return prompt | model | StrOutputParser()
    
    @staticmethod
    def summarization_chain():
        """Chain to summarize text."""
        prompt = ChatPromptTemplate.from_template(
            """Summarize the following text in 2-3 sentences:

Text: {text}

Summary:"""
        )
        return prompt | model | StrOutputParser()
    
    @staticmethod
    def code_generation_chain():
        """Chain to generate Python code."""
        prompt = ChatPromptTemplate.from_template(
            """Generate Python code to {task}. Return only the code, no explanation.

Code:"""
        )
        return prompt | model | StrOutputParser()
    
    @staticmethod
    def translation_chain():
        """Chain to translate text."""
        prompt = ChatPromptTemplate.from_template(
            """Translate the following text to {target_language}:

Text: {text}

Translation:"""
        )
        return prompt | model | StrOutputParser()
    
    @staticmethod
    def sentiment_analysis_chain():
        """Chain to analyze sentiment of text."""
        prompt = ChatPromptTemplate.from_template(
            """Analyze the sentiment of the following text. Return: Positive, Negative, or Neutral with a brief explanation.

Text: {text}

Sentiment:"""
        )
        return prompt | model | StrOutputParser()
    
    @staticmethod
    def explanation_chain():
        """Chain to explain a concept."""
        prompt = ChatPromptTemplate.from_template(
            """Explain the following concept in simple terms suitable for a beginner:

Concept: {concept}

Explanation:"""
        )
        return prompt | model | StrOutputParser()


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("LangChain Gemini Chain Utilities - Demo")
    print("=" * 70)
    
    # Q&A Example
    print("\n1. Q&A Chain")
    print("-" * 70)
    qa = GeminiChains.qa_chain()
    qa_result = qa.invoke({"question": "What is the difference between lists and tuples in Python?"})
    print(f"Q: What is the difference between lists and tuples in Python?")
    print(f"A: {qa_result}\n")
    
    # Summarization Example
    print("\n2. Summarization Chain")
    print("-" * 70)
    summarize = GeminiChains.summarization_chain()
    text = "Artificial intelligence (AI) has revolutionized multiple industries. From healthcare to finance, AI systems are helping humans make better decisions, improve efficiency, and solve complex problems. Machine learning models can now process vast amounts of data and extract valuable insights."
    summary_result = summarize.invoke({"text": text})
    print(f"Original text: {text}")
    print(f"Summary: {summary_result}\n")
    
    # Code Generation Example
    print("\n3. Code Generation Chain")
    print("-" * 70)
    code_gen = GeminiChains.code_generation_chain()
    code_result = code_gen.invoke({"task": "sort a list of dictionaries by a specific key"})
    print(f"Task: Sort a list of dictionaries by a specific key")
    print(f"Generated code:\n{code_result}\n")
    
    # Translation Example
    print("\n4. Translation Chain")
    print("-" * 70)
    translate = GeminiChains.translation_chain()
    trans_result = translate.invoke({"text": "Hello, how are you?", "target_language": "Spanish"})
    print(f"Original: Hello, how are you?")
    print(f"Spanish: {trans_result}\n")
    
    # Sentiment Analysis Example
    print("\n5. Sentiment Analysis Chain")
    print("-" * 70)
    sentiment = GeminiChains.sentiment_analysis_chain()
    sentiment_result = sentiment.invoke({"text": "I absolutely love this product! It has made my life so much easier."})
    print(f"Text: I absolutely love this product! It has made my life so much easier.")
    print(f"Analysis: {sentiment_result}\n")
    
    # Explanation Example
    print("\n6. Explanation Chain")
    print("-" * 70)
    explain = GeminiChains.explanation_chain()
    explain_result = explain.invoke({"concept": "API (Application Programming Interface)"})
    print(f"Concept: API (Application Programming Interface)")
    print(f"Explanation: {explain_result}\n")
    
    print("=" * 70)
    print("Demo Complete!")
    print("=" * 70)
