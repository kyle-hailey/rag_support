from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.schema import Document
import openai
import psycopg2
import os

# Get OpenAI API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ OPENAI_API_KEY environment variable not set")
    exit(1)

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = api_key

connection_string = "postgresql://yugabyte:Password123#@10.9.109.47:5433/yugabyte"

# Try to connect and give feedback
try:
    conn = psycopg2.connect(connection_string)
    cursor = conn.cursor()
    print("✅ Successfully connected to the database.\n")
except Exception as e:
    print("❌ Failed to connect to the database.")
    print("Error:", e)
    exit(1)

# Load documents and create index
print("📄 Loading documents...")
try:
    documents = SimpleDirectoryReader("./data").load_data()
    print(f"📦 Loaded {len(documents)} documents.\n")
except Exception as e:
    print(f"❌ Failed to load documents: {e}")
    print("Creating a sample document for testing...")
    # Create a sample document if ./data directory doesn't exist
    documents = [Document(text="This is a sample document for testing vector embeddings.")]

print("🔍 Vectorizing documents...")
index = VectorStoreIndex.from_documents(documents)
print("✅ Vectorization complete.\n")

# Insert documents with clean feedback
for i, doc in enumerate(documents):
    # print(f"Debug - Document {i}: {type(doc)}")
    # print(f"Debug - Document attributes: {dir(doc)}")
    
    print("my debug ", i,doc)
    
    doc_id = f"doc_{i}"
    
    # Use get_content() for modern LlamaIndex
    doc_text = doc.get_content() if hasattr(doc, 'get_content') else getattr(doc, 'text', None)
    # print(f"Debug - Extracted text: {repr(doc_text)[:100]}")
    
    if not doc_text or len(str(doc_text).strip()) == 0:
        print(f"⚠️  Skipping document {doc_id}: empty text")
        continue

    # Truncate if too long for OpenAI embedding
    if len(doc_text) > 8000:
        doc_text = doc_text[:8000]
        print(f"⚠️  Truncated document {doc_id} to 8000 characters")

    try:
        embedding = index._embed_model.get_text_embedding(doc_text)
        embedding_str = "[" + ",".join(map(str, embedding)) + "]"
        insert_sql = """
            INSERT INTO vectors (id, article_text, embedding)
            VALUES (%s, %s, %s)
            ON CONFLICT (id) DO NOTHING;
        """
        cursor.execute(insert_sql, (doc_id, doc_text, embedding_str))
        conn.commit()
        text_snippet = doc_text[:40].replace("\n", " ").strip()
        print(f"📥 {len(doc_text):4d} chars | \"{text_snippet}\" | { [round(v, 4) for v in embedding[:5]] }")
    except Exception as e:
        print(f"❌ Failed to process document {doc_id}: {e}")
        print(f"   Text preview: {doc_text[:100]}...")

print("\n🎉 Done inserting all data.")
cursor.close()
conn.close()