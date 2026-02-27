import streamlit as st
from vector_engine import VectorEngine
from agent import generate_recommendation

# Configure page UI
st.set_page_config(page_title="AlgoMatch: Agentic DSA Pattern Recommender", page_icon="🧠", layout="centered")

@st.cache_resource
def init_system():
    """Initializes the Endee DB and populates knowledge base on streamlt startup."""
    engine = VectorEngine(host="http://localhost:8080")
    engine.initialize_db()
    return engine

def main():
    st.title("🧠 AlgoMatch")
    st.subheader("Agentic DSA Pattern Recommender")
    st.markdown("Describe a software architecture or algorithmic problem you're trying to solve. AlgoMatch will query the local Endee vector database and leverage LLaMA 3 to recommend the best algorithm and generate C++ code.")
    
    # Initialize Vector DB on startup
    with st.spinner("Initializing Vector Engine and Local Endee DB..."):
        try:
            engine = init_system()
        except Exception as e:
            st.error(f"Error connecting to Endee DB at localhost:8080: {e}")
            return
        
    # User Input
    user_problem = st.text_area("Describe your technical problem:", height=120, 
                                placeholder="E.g., I need a way to schedule a batch of dependent background tasks ensuring they run in the right logical order...")
    
    if st.button("Find Best Algorithm"):
        if not user_problem.strip():
            st.warning("Please describe your problem.")
            return
            
        with st.spinner("Executing semantic search on Endee DB..."):
            retrieved_patterns = engine.search_pattern(user_problem, top_k=2)
            
        if not retrieved_patterns:
            st.error("No relevant patterns found in the standard database.")
            return
            
        with st.spinner("Generating recommendation and C++ architectural boilerplate with LLaMA 3..."):
            try:
                recommendation = generate_recommendation(user_problem, retrieved_patterns)
                st.success("Analysis Complete!")
                st.divider()
                st.markdown(recommendation)
            except Exception as e:
                st.error(f"An error occurred with Ollama execution: {e}")

if __name__ == "__main__":
    main()
