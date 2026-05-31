"""
Сброс памяти модели
"""
import os
import json
from models.trader_model import trader_model_instance

print("=" * 60)
print("🧹 СБРОС ПАМЯТИ МОДЕЛИ")
print("=" * 60)

model = trader_model_instance

# Текущее состояние
print(f"\nТекущая память: {len(model.memory)} опытов")
print(f"Приоритетный буфер: {model.prioritized_buffer.size} опытов")

# Подтверждение
print("\n⚠️ Это удалит ВСЕ накопленные опыты.")
print("Модель начнёт обучение с нуля.")
confirm = input("Продолжить? (yes/no): ").strip().lower()

if confirm != "yes":
    print("Отменено.")
    exit()

# Очистка памяти
model.memory.clear()

# Очистка приоритетного буфера
from models.trader_model import PrioritizedReplayBuffer
old_alpha = model.prioritized_buffer.alpha
old_beta = model.prioritized_buffer.beta
old_beta_inc = model.prioritized_buffer.beta_increment
model.prioritized_buffer = PrioritizedReplayBuffer(
    max_size=model.rl_config.get("memory_size", 5000),
    alpha=old_alpha,
    beta=old_beta,
    beta_increment=old_beta_inc
)

# Удаление файла памяти
memory_file = model.memory_config.get('memory_file', 'models/saved_trader/memory_buffer.pkl')
if os.path.exists(memory_file):
    os.remove(memory_file)
    print(f"Файл памяти удалён: {memory_file}")

# Сохранение пустой памяти
model.save_memory()

print(f"\n✅ Память очищена: {len(model.memory)} опытов")
print(f"✅ Приоритетный буфер: {model.prioritized_buffer.size} опытов")
print("\nМожно перезапускать main.py")