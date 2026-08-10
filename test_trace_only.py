"""
Минимальный тест трейсинга.
"""

import sys
from pathlib import Path

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("🔍 ЗАПУСК ТЕСТА ТРЕЙСИНГА")
print("=" * 70)

# Шаг 1: Импорт
print("\n1️⃣ Импортируем setup_tracing...")
try:
    from src.observability import setup_tracing
    print("✅ Импорт успешен")
except Exception as e:
    print(f"❌ Ошибка импорта: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Шаг 2: Создание трейсера
print("\n2️⃣ Вызываем setup_tracing...")
try:
    tracer = setup_tracing(service_name="test-service")
    print(f"✅ Трейсер создан: {tracer}")
    print(f"   Тип: {type(tracer)}")
except Exception as e:
    print(f"❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Шаг 3: Создание span
print("\n3️⃣ Пробуем создать span...")
try:
    with tracer.start_span("test_span", {"test": "value", "user": "test_user"}):
        print("   ✅ Внутри span")
        import time
        time.sleep(0.2)
        
        # Вложенный span
        with tracer.start_span("inner_span", {"inner": "data"}):
            print("   ✅ Внутри вложенного span")
            time.sleep(0.1)
    
    print("✅ Span завершён успешно!")
except Exception as e:
    print(f"❌ Ошибка при создании span: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ ТЕСТ ЗАВЕРШЁН УСПЕШНО!")
print("=" * 70)
print("\n📝 Теперь проверьте вывод выше — должны быть видны трейсы:")
print("   - 🔍 Starting span: test_span")
print("   - test: value")
print("   - user: test_user")
print("   - ✅ Span completed: test_span")
print("   - 🔍 Starting span: inner_span")
print("   - inner: data")
print("   - ✅ Span completed: inner_span")