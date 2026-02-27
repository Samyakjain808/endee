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
You are an elite AI Software Architect and C++ expert. 
A user has outlined the following real-world software problem:
"{user_problem}"

Based strictly on our semantic search from the Endee vector database, here are the most relevant DSA patterns available:
{context_str}

Please respond formatted in clear Markdown with the following requirements:
1. **Algorithm Analysis:** Pick the most optimal pattern from the provided context. Explain clearly *why* it's the perfect fit for their exact problem. Mention time/space tradeoffs.
2. **C++ Boilerplate:** Output a boilerplate C++ implementation template solving a generalized variant of the problem using the recommended data structure/algorithm.
    """
    
    response = ollama.chat(
        model='llama3',
        messages=[
            {'role': 'system', 'content': 'You are a master competitive programmer helping developers design optimal systems.'},
            {'role': 'user', 'content': prompt}
        ]
    )
    
    return response['message']['content']
