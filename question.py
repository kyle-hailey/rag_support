import psycopg2
import openai
import os
from llama_index.embeddings.openai import OpenAIEmbedding

# --- Setup ---
openai.api_key = os.getenv("OPENAI_API_KEY")

embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
client = openai.OpenAI()
connection_string = "postgresql://yugabyte:Password123#@10.9.109.47:5433/yugabyte"

def ask_question(question, top_k=7):
    # Connect to DB
    conn = psycopg2.connect(connection_string)
    cursor = conn.cursor()
    
    # Get embedding of question
    query_embedding = embed_model.get_query_embedding(question)
    embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
    
    # Vector search
    search_sql = """
        SELECT id, article_text, embedding <=> %s AS distance
        FROM vectors
        ORDER BY embedding <=> %s
        LIMIT %s;
    """
    cursor.execute(search_sql, (embedding_str, embedding_str, top_k))
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    
    # Get baseline answer (without RAG context)
    baseline_response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer the user's question based on your general knowledge."},
            {"role": "user", "content": question}
        ],
        temperature=0.3,
    )
    baseline_answer = baseline_response.choices[0].message.content
    
    if not results:
        return baseline_answer, "No matching documents found for RAG context."
    
    # Print the first 40 characters of each retrieved chunk
    print("\n🔍 Retrieved context snippets:")
    for _, text, distance in results:
        print(f"- {text[:40]!r} (distance: {distance:.4f})")
    
    # Build context
    context = "\n\n".join([f"{text}" for (_, text, _) in results])
    
    # Ask GPT with RAG context
    rag_response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions using the provided context."},
            {"role": "user", "content": f"Context:\n{context}"},
            {"role": "user", "content": f"Question: {question}"}
        ],
        temperature=0.3,
    )
    rag_answer = rag_response.choices[0].message.content
    
    return baseline_answer, rag_answer

# --- Interactive Loop ---
if __name__ == "__main__":
    try:
        print("Ask me a question (press Ctrl+C to quit):\n")
        while True:
            question = input("❓ Your question: ").strip()
            if not question:
                continue
            
            baseline_answer, rag_answer = ask_question(question)
            
            print("\n" + "="*80)
            print("🤖 BASELINE ANSWER (without RAG context):")
            print("="*80)
            print(baseline_answer)
            
            print("\n" + "="*80)
            print("🧠 RAG-ENHANCED ANSWER (with context):")
            print("="*80)
            print(rag_answer)
            print("="*80 + "\n")
            
    except KeyboardInterrupt:
        print("\nGoodbye! 👋")