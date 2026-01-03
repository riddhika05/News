# news_app.py
import pathway as pw
import os
import json
import requests
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up Pathway license
pw.set_license_key("C9602B-C84476-AB9915-36D2D6-92FC0E-V3")

# ====================
# 1. News Source Setup
# ====================

class GNewsSchema(pw.Schema):
    doc: str
    _metadata: dict

class GNewsConnector(pw.io.python.ConnectorSubject):
    def __init__(self, api_key: str, refresh_interval: int = 300):
        super().__init__()
        self.api_key = api_key
        self.refresh_interval = refresh_interval  # Refresh every 5 minutes
        
    def run(self):
        while True:
            try:
                url = f"https://gnews.io/api/v4/top-headlines?category=general&apikey={self.api_key}"
                print(f"📡 Fetching news from GNews API...")
                
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                data = response.json()
                articles = data.get("articles", [])
                
                print(f"✅ Fetched {len(articles)} articles")
                
                for art in articles:
                    text_content = f"Headline: {art.get('title', 'No title')}\nSummary: {art.get('description', 'No description')}"
                    
                    self.next(
                        doc=text_content,
                        _metadata={
                            "url": art.get("url", ""),
                            "source": "gnews_api",
                            "title": art.get('title', ''),
                            "published_at": art.get('publishedAt', '')
                        }
                    )
                
                print(f"⏳ Waiting {self.refresh_interval} seconds before next fetch...")
                time.sleep(self.refresh_interval)
                
            except Exception as e:
                print(f"❌ Error fetching news: {e}")
                time.sleep(60)  # Wait 1 minute before retrying on error

def get_news_source():
    """Create news source using Python connector"""
    api_key = os.environ.get("G_NEWS_API_KEY")
    
    if not api_key:
        raise ValueError("G_NEWS_API_KEY environment variable is not set!")
    
    connector = GNewsConnector(api_key=api_key, refresh_interval=300)
    
    return pw.io.python.read(
        connector,
        schema=GNewsSchema,
        autocommit_duration_ms=5000
    )

# ====================
# 2. News Pipeline Components
# ====================
print("📰 Setting up News RAG Pipeline...")

# Get news source
news_table = get_news_source()

# Filter out empty documents
news_table = news_table.filter(news_table.doc != "")

# Transform news table
news_table_fixed = news_table.select(data=news_table.doc, _metadata=news_table._metadata)

# Import required modules
from pathway.xpacks.llm.document_store import DocumentStore
from pathway.xpacks.llm.splitters import TokenCountSplitter
from pathway.xpacks.llm.llms import LiteLLMChat
from pathway.xpacks.llm.embedders import GeminiEmbedder
from pathway.stdlib.indexing import BruteForceKnnFactory, TantivyBM25Factory, HybridIndexFactory
from pathway.xpacks.llm.question_answering import BaseRAGQuestionAnswerer
from pathway.xpacks.llm.servers import QARestServer

print("✅ News source configured!")

# News splitter
news_splitter = TokenCountSplitter(min_tokens=100, max_tokens=500)

# News embedder
news_embedder = GeminiEmbedder(
    model="models/embedding-001",
    cache_strategy=pw.udfs.DefaultCache(),
    retry_strategy=pw.udfs.ExponentialBackoffRetryStrategy(max_retries=3)
)

# News retriever setup
news_knn_index = BruteForceKnnFactory(
    reserved_space=500, 
    embedder=news_embedder, 
    metric=pw.engine.BruteForceKnnMetricKind.COS
)
news_bm25_index = TantivyBM25Factory()
news_retriever_factory = HybridIndexFactory(retriever_factories=[news_knn_index, news_bm25_index])

# News DocumentStore
news_document_store = DocumentStore(
    docs=[news_table_fixed],
    parser=None,
    splitter=news_splitter,
    retriever_factory=news_retriever_factory
)

print("✅ News document store created!")

# News LLM
news_llm = LiteLLMChat(
    model="gemini/gemini-2.0-flash",
    retry_strategy=pw.udfs.ExponentialBackoffRetryStrategy(max_retries=2),
    cache_strategy=pw.udfs.DefaultCache(),
    temperature=0.1,
    verbose=False
)

# News-specific prompt template
news_prompt_template = """You are a helpful news assistant. Answer the question based ONLY on the recent news articles provided.
If the news doesn't contain relevant information, say "I don't have recent news about that."

Question: {query}

Recent News: {context}

Answer based on news:"""

# Create the news question answerer
news_qa = BaseRAGQuestionAnswerer(
    llm=news_llm,
    indexer=news_document_store,
    prompt_template=news_prompt_template,
    search_topk=3
)

print("✅ News QA pipeline created!")

# ====================
# 3. Create and Run REST Server
# ====================
print("\n🚀 Setting up REST server...")

# Create and run the REST server
server = QARestServer(
    host="0.0.0.0",
    port=8080,
    rag_question_answerer=news_qa
)

print("\n🚀 Starting News RAG server on http://0.0.0.0:8080")
print("   • POST /v1/answer - Get answers")
print("   • POST /v1/retrieve - Get retrieved context")
print("   • POST /v1/summarize - Get summaries")
print("\n📰 News will refresh every 5 minutes")
print("\nPress Ctrl+C to stop")

# Run the server
pw.run(monitoring_level=pw.MonitoringLevel.NONE)