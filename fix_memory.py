import pickle
import gzip
import numpy as np

# Загружаем текущую память
with gzip.open('models/saved_trader/memory_buffer.pkl', 'rb') as f:
    memory = pickle.load(f)

print(f"Загружено {len(memory)} опытов")
print(f"Средний reward до: {np.mean([exp['reward'] for exp in memory]):.2f}")

# Пересчитываем reward (делим на 2000, чтобы из рублей получить проценты)
# Средний PnL -210 / средняя цена 1000 = -0.21 (проценты)
for exp in memory:
    exp['reward'] = exp['reward'] / 2000.0  # грубая нормализация

print(f"Средний reward после: {np.mean([exp['reward'] for exp in memory]):.4f}")

# Сохраняем обратно
with gzip.open('models/saved_trader/memory_buffer_fixed.pkl', 'wb') as f:
    pickle.dump(memory, f)

print("Память пересчитана и сохранена как memory_buffer_fixed.pkl")