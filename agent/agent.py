from datetime import datetime
import sys
from pathlib import Path

from google.adk.agents.llm_agent import Agent

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.context import app_context
from app.services.chunk_store import ChunkStore 

if not app_context.chunk_store:
    app_context.chunk_store = ChunkStore()

def get_datetime() -> str:
    """Get the current date and time.
    
    Returns:
        str: Current date and time in ISO 8601 format (YYYY-MM-DD HH:MM:SS)
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def search_knowledge_base(query: str, n_results: int = 5) -> str:
    """Search the knowledge base (ChromaDB) for relevant information.
    
    This tool searches through stored text chunks in the vector database
    to find information relevant to the user's query.
    
    Args:
        query: The search query or question to find relevant information for
        n_results: Number of relevant chunks to retrieve (default: 5, max: 20)
    
    Returns:
        str: Formatted string containing the relevant text chunks found, 
             or a message if no results are found
    """

    n_results = max(1, min(n_results, 20))
    
    try:
        if not app_context.chunk_store:
            return "Error: ChunkStore not initialized."

        chunks = app_context.chunk_store.retrieve_chunks(query, n_results)
        
        if not chunks:
            return "No relevant information found in the knowledge base."
        
        result_parts = [f"Found {len(chunks)} relevant chunks:\n"]
        for i, chunk in enumerate(chunks, 1):
            distance = chunk.metadata.get('distance', 'N/A')
            result_parts.append(f"\n--- Chunk {i} (similarity: {distance}) ---")
            result_parts.append(chunk.text)
            
            other_metadata = {k: v for k, v in chunk.metadata.items() if k != 'distance'}
            if other_metadata:
                result_parts.append(f"Metadata: {other_metadata}")
        
        return "\n".join(result_parts)
        
    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"

root_agent = Agent(
    model='gemini-2.5-flash',
    name='root_agent',
    description='A helpful assistant with access to a knowledge base',
    instruction="""Answer user questions to the best of your ability. 
    You have access to a knowledge base through the search_knowledge_base tool.
    When users ask questions that might be answered by stored information, 
    use the search_knowledge_base tool to find relevant information before answering.""",
    tools=[get_datetime, search_knowledge_base],

)
