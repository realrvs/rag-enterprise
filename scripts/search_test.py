"""
Test script for searching indexed documents.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval import QdrantClientWrapper


def test_search():
    """Test search functionality."""
    qdrant = QdrantClientWrapper()
    
    # Check if collection exists and has points
    count = qdrant.count_points()
    print("=" * 60)
    print(f"🔍 Search Test")
    print("=" * 60)
    print(f"📊 Points in collection: {count}")
    
    if count == 0:
        print("⚠️ No points found in collection!")
        print("📝 Run: python scripts/index_documents.py first")
        return
    
    queries = [
        "Что такое тариф Бизнес-Старт?",
        "Сколько стоит тариф Бизнес-Про?",
        "Какие тарифы предлагает МТС?",
    ]
    
    for query in queries:
        print(f"\n📝 Query: {query}")
        print("-" * 40)
        
        results = qdrant.search(
            query=query,
            top_k=3,
            use_hybrid=True,
        )
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"{i}. Score: {result['score']:.4f}")
                print(f"   Text: {result['text'][:200]}...")
                print(f"   Source: {result['metadata'].get('source', 'unknown')}")
                print()
        else:
            print("   No results found")


def test_api():
    """Test API via requests."""
    try:
        import requests
        
        print("\n" + "=" * 60)
        print("🔍 API Test")
        print("=" * 60)
        
        response = requests.get("http://localhost:8000/health")
        print(f"✅ Health: {response.status_code} - {response.json()}")
        
        test_question = "Какие тарифы предлагает МТС?"
        response = requests.post(
            "http://localhost:8000/query",
            json={"question": test_question},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        print(f"✅ Query: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        
    except ImportError:
        print("⚠️ Requests library not installed. Skipping API test.")
    except Exception as e:
        print(f"⚠️ API test failed: {e}")


if __name__ == "__main__":
    test_search()
    test_api()