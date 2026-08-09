"""
Тест создания RAG Pipeline.
"""

import sys
import logging
from pathlib import Path

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent))

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_pipeline_import():
    """Тест 1: Проверка импорта."""
    print("\n" + "=" * 60)
    print("🧪 ТЕСТ 1: Проверка импорта")
    print("=" * 60)
    
    try:
        from src.pipeline.rag_pipeline import RAGPipeline
        print("✅ RAGPipeline импортирован успешно")
        return True
    except Exception as e:
        print(f"❌ Ошибка импорта: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline_creation():
    """Тест 2: Создание экземпляра пайплайна."""
    print("\n" + "=" * 60)
    print("🧪 ТЕСТ 2: Создание RAGPipeline")
    print("=" * 60)
    
    try:
        from src.pipeline.rag_pipeline import RAGPipeline
        pipeline = RAGPipeline()
        print(f"✅ RAGPipeline создан: {pipeline}")
        print(f"   Тип: {type(pipeline)}")
        print(f"   Атрибуты: {dir(pipeline)}")
        return pipeline
    except Exception as e:
        print(f"❌ Ошибка создания: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_qdrant_connection():
    """Тест 3: Проверка подключения к Qdrant."""
    print("\n" + "=" * 60)
    print("🧪 ТЕСТ 3: Подключение к Qdrant")
    print("=" * 60)
    
    try:
        from src.retrieval.qdrant_client import QdrantClientWrapper
        qdrant = QdrantClientWrapper()
        print(f"✅ Qdrant клиент создан: {qdrant}")
        
        # Проверяем, что коллекция существует или создаётся
        try:
            info = qdrant.get_collection_info()
            print(f"   Коллекция: {info.get('name', 'N/A')}")
            print(f"   Количество точек: {info.get('points_count', 0)}")
        except Exception as e:
            print(f"   Информация о коллекции: {e}")
            print("   (это нормально, если коллекция ещё не создана)")
        
        return qdrant
    except Exception as e:
        print(f"❌ Ошибка подключения к Qdrant: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_pipeline_query():
    """Тест 4: Тест метода query."""
    print("\n" + "=" * 60)
    print("🧪 ТЕСТ 4: Тест pipeline.query()")
    print("=" * 60)
    
    try:
        from src.pipeline.rag_pipeline import RAGPipeline
        pipeline = RAGPipeline()
        
        test_query = "Что такое тариф Бизнес-Старт?"
        print(f"📝 Запрос: {test_query}")
        
        response = pipeline.query(question=test_query, top_k=3, use_hybrid=False)
        
        print("✅ Ответ получен:")
        print(f"   Статус: {response.get('status', 'unknown')}")
        print(f"   Контекстов: {len(response.get('contexts', []))}")
        print(f"   Время: {response.get('latency_ms', 0):.2f}ms")
        
        if response.get('contexts'):
            print("\n📄 Контексты:")
            for i, ctx in enumerate(response.get('contexts', [])[:3], 1):
                print(f"   {i}. {ctx[:100]}...")
        
        return response
    except Exception as e:
        print(f"❌ Ошибка запроса: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_all_tests():
    """Запуск всех тестов."""
    print("\n" + "=" * 70)
    print("🚀 ЗАПУСК ТЕСТОВ RAG PIPELINE")
    print("=" * 70)
    print(f"📁 Текущая директория: {Path.cwd()}")
    print(f"🐍 Python: {sys.version}")
    
    results = {}
    
    # Тест 1: Импорт
    results['import'] = test_pipeline_import()
    if not results['import']:
        print("\n❌ Тест импорта провален. Дальнейшие тесты не имеют смысла.")
        return
    
    # Тест 2: Создание
    results['creation'] = test_pipeline_creation()
    if results['creation'] is None:
        print("\n❌ Тест создания провален.")
        return
    
    # Тест 3: Qdrant
    results['qdrant'] = test_qdrant_connection()
    
    # Тест 4: Запрос
    results['query'] = test_pipeline_query()
    
    # Итоги
    print("\n" + "=" * 70)
    print("📊 ИТОГИ ТЕСТИРОВАНИЯ")
    print("=" * 70)
    
    for name, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {name}: {result}")
    
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()