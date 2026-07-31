"""
PDF Q&A Agent using LlamaIndex.

Loads a PDF, indexes it, and answers questions about its content.
Maintains conversation history for follow-up questions.

Usage:
    python agent.py --pdf path/to/document.pdf
    python agent.py --pdf report.pdf --question "What is the main finding?"
""" 


import argparse
import os

from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

load_dotenv()

Settings.llm = Groq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

def build_index(pdf_path: str) -> VectorStoreIndex:
    print(f"📄 Loading and indexing {pdf_path}... ")
    reader = SimpleDirectoryReader(input_files=[pdf_path])
    docs = reader.load_data()
    index = VectorStoreIndex.from_documents(docs)
    print("✅ Indexing complete.")
    return index


def interactive_qa(index: VectorStoreIndex, question: str = None) -> None:
    chat_memory = ChatMemoryBuffer.from_defaults()
    chat_engine = index.as_chat_engine(chat_mode="context", memory=chat_memory)

    if question:
        print(f"🤖 Question: {question}")
        response = chat_engine.chat(question)
        print(f"💬 Answer: {response.response}")
    else:
        print("🤖 Enter your questions (type 'exit' to quit):")
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                break
            response = chat_engine.chat(user_input)
            print(f"💬 Answer: {response.response}")
            
            
def single_question(index: VectorStoreIndex, question: str) -> None: 
    query_engine = index.as_query_engine() 
    response = query_engine.query(question) 
    print("\n" + "=" * 60) 
    print("Answer")  
    print("=" * 60)
    print(response.response) 
    if hasattr(response, "source_nodes") and response.source_nodes: 
        print(f"\n 📚 Sources: {len(response.source_nodes)} chunk(s) referenced")



def main(): 
    parser = argparse.ArgumentParser(description="PDF Q&A Agent") 
    parser.add_argument("--pdf", type=str, required=True, help="Path to the PDF file") 
    parser.add_argument("--question", help="Single question (omit for interactive mode)")
    args = parser.parse_args() 
    
    index = build_index(args.pdf) 
    
    if args.question: 
        single_question(index, args.question) 
    else: 
        interactive_qa(index) 
        
if __name__ == "__main__": 
    main()