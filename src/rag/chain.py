import os
import json
from typing import Dict, Any, List
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.rag.retriever import get_retriever

# Load environment variables
load_dotenv()

def get_router_llm() -> ChatGoogleGenerativeAI:
    """
    Initialize gemini-2.5-flash-lite for rapid routing, intent analysis, and query rewriting.
    """
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY must be set in environment variables.")
    gemini_key = gemini_key.strip()
        
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=gemini_key,
        temperature=0.0
    )

def get_generation_llm() -> ChatGoogleGenerativeAI:
    """
    Initialize gemini-2.5-flash with hybrid reasoning for document answering and hallucination check.
    """
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY must be set in environment variables.")
    gemini_key = gemini_key.strip()
        
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=gemini_key,
        temperature=0.2
    )

def format_docs(docs) -> str:
    """
    Format list of Documents into a single string context.
    """
    return "\n\n".join(
        f"Document Chunk:\n{doc.page_content}\nMetadata: {json.dumps(doc.metadata)}"
        for doc in docs
    )

def get_rag_chain():
    """
    Constructs and returns the full dual-model RAG pipeline.
    The chain returns a dict: {"answer": str, "source_documents": List[Document]}
    """
    router_llm = get_router_llm()
    generation_llm = get_generation_llm()
    retriever = get_retriever()
    
    # 1. Routing & Query Rewriting prompts
    router_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an intelligent query router. Analyze the user query and output a JSON object containing:\n"
            "1. 'is_conversational': true if the query is a simple greeting, thank you, or chit-chat. false if it requires fetching factual documents.\n"
            "2. 'optimized_query': A rewritten version of the query optimized for semantic search. Expand abbreviations, add synonyms, and remove filler words. If 'is_conversational' is true, set this to the original query.\n"
            "Your output MUST be a valid JSON block containing only those two keys. Do not include markdown code block formatting."
        )),
        ("human", "{question}")
    ])
    
    # 2. Answer generation prompt for Gemini 2.5 Flash
    generation_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a helpful, enterprise-grade clinical AI assistant.\n"
            "Your goal is to provide highly specific, detailed, and particular answers to the user's query.\n\n"
            "Instructions:\n"
            "1. Analyze the provided context carefully. If the context contains relevant medical/clinical information, answer the user's question with utmost precision, highlighting specific numbers, terms, symptoms, recommendations, and evidence from the context.\n"
            "2. If the context is empty, irrelevant, or does not contain the answer to the query, do NOT say 'I cannot find the answer'. Instead, provide a highly detailed, comprehensive, and particular answer based on your general medical and clinical knowledge. You MUST clearly prefix your response with the exact tag: '**[General Knowledge Response - Not Grounded in Uploaded Documents]**' so the user is aware.\n"
            "3. Format your response beautifully using bold text, structured paragraphs, and bullet points where appropriate to make it highly readable and premium.\n\n"
            "Context:\n{context}"
        )),
        ("human", "{optimized_query}")
    ])
    
    # 3. Hallucination and groundedness check prompt for Gemini 2.5 Flash
    hallucination_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert hallucination checker.\n"
            "Your task is to analyze the draft answer and compare it strictly with the provided context.\n"
            "If the draft answer is marked with '**[General Knowledge Response - Not Grounded in Uploaded Documents]**', then it is allowed to use general clinical knowledge; in this case, review the medical accuracy of the response, ensure it is extremely detailed, helpful, and particular, and output it with the prefix intact.\n"
            "If the draft answer is NOT marked with that prefix, it MUST be strictly grounded in the provided context. If it contains any facts, numbers, or assumptions not explicitly stated in the context, you must rewrite or prune them to maintain strict groundedness. Ensure the response remains highly specific, comprehensive, and detailed.\n\n"
            "Context:\n{context}\n\n"
            "Draft Answer:\n{draft_answer}"
        )),
        ("human", "Check the draft answer and output your final grounded answer.")
    ])

    def run_pipeline(question: str) -> Dict[str, Any]:
        # Run step 1: Intelligent routing and rewriting
        router_chain = router_prompt | router_llm | StrOutputParser()
        router_result_raw = router_chain.invoke({"question": question})
        
        # Parse JSON output from router
        try:
            # Strip any markdown wrappers if present
            clean_json = router_result_raw.strip().replace("```json", "").replace("```", "").strip()
            router_result = json.loads(clean_json)
        except Exception:
            # Fallback in case of parsing errors
            router_result = {"is_conversational": False, "optimized_query": question}
            
        is_conversational = router_result.get("is_conversational", False)
        optimized_query = router_result.get("optimized_query", question)
        
        # If it is a purely conversational chit-chat query, bypass retrieval entirely
        if is_conversational:
            direct_reply_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a friendly enterprise-grade AI assistant. Reply to the user's greeting or comment politely and naturally."),
                ("human", "{question}")
            ])
            direct_chain = direct_reply_prompt | router_llm | StrOutputParser()
            answer = direct_chain.invoke({"question": question})
            return {
                "answer": answer,
                "source_documents": []
            }
            
        # Run step 2: Retrieve relevant documents
        docs = retriever.invoke(optimized_query)
        context_str = format_docs(docs)
        
        # Run step 3: Draft answer generation
        gen_chain = generation_prompt | generation_llm | StrOutputParser()
        draft_answer = gen_chain.invoke({
            "context": context_str,
            "optimized_query": optimized_query
        })
        
        # Run step 4: Hallucination check & correction pass
        checker_chain = hallucination_prompt | generation_llm | StrOutputParser()
        final_answer = checker_chain.invoke({
            "context": context_str,
            "draft_answer": draft_answer
        })
        
        return {
            "answer": final_answer,
            "source_documents": docs
        }

    return run_pipeline
