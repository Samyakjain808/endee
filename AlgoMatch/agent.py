import ollama

def generate_recommendation(user_problem: str, retrieved_context: list) -> str:
    """Uses Ollama's Llama 3 to analyze the user problem against mapped vector context."""
    
    # Parse the metadata contexts returned by Endee
    context_str = ""
    for idx, item in enumerate(retrieved_context):
        # Depending on the Endee SDK return signature, we extract standard metadata
        meta = item.get("metadata", item) 
        context_str += f"\n--- Pattern {idx+1}: {meta.get('name', 'Unknown')} ---\n"
        context_str += f"Description: {meta.get('description', '')}\n"
        context_str += f"Use Cases: {meta.get('use_cases', '')}\n"
        context_str += f"Time Complexity: {meta.get('time_complexity', '')}\n"
        context_str += f"Space Complexity: {meta.get('space_complexity', '')}\n"
        
    prompt = f"""
A user has outlined the following real-world software problem:
"{user_problem}"

Based strictly on our semantic search from the Endee vector database, here are the most relevant DSA patterns available:
{context_str}
    """
    
    system_prompt = """You are an elite C++ Software Engineer. 
Analyze the user's software problem and the provided Data Structure/Algorithm context.
1. Explain concisely WHY this algorithm is the best fit.
2. Provide a clean, compilable C++ implementation. 
CRITICAL RULES FOR C++: 
- Do not use iterators on std::priority_queue or std::stack. 
- Use the 'Two Heaps' pattern (std::priority_queue and std::greater) for running median problems.
- Ensure no data is destroyed during median calculation."""

    response = ollama.chat(
        model='llama3',
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt}
        ]
    )
    
    return response['message']['content']
