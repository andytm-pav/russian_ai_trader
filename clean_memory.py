#!/usr/bin/env python3
"""
Очистка памяти модели от убыточных опытов
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def clean_memory():
    print("=" * 60)
    print("ОЧИСТКА ПАМЯТИ МОДЕЛИ")
    print("=" * 60)

    # Загружаем конфиг для параметров очистки
    try:
        with open("config/rl_config.json", "r", encoding="utf-8") as f:
            rl_config = json.load(f)
    except:
        rl_config = {}

    memory_clean_config = rl_config.get('memory_clean', {
        'keep_profitable_only': True,
        'keep_hold_ratio': 0.1,
        'max_memory_after_clean': 3000
    })

    from models.trader_model import trader_model_instance
    model = trader_model_instance

    original_size = len(model.memory)
    print(f"\nИсходный размер памяти: {original_size}")

    if original_size == 0:
        print("Память пуста, нечего очищать")
        return

    new_memory = []
    keep_profitable = memory_clean_config.get('keep_profitable_only', True)
    keep_hold_ratio = memory_clean_config.get('keep_hold_ratio', 0.1)
    max_memory = memory_clean_config.get('max_memory_after_clean', 3000)

    profitable_count = 0
    hold_count = 0

    for exp in model.memory:
        pnl = exp.get('pnl_rub', 0) if isinstance(exp, dict) else 0
        action = exp.get('action', 1) if isinstance(exp, dict) else 1

        if keep_profitable and pnl > 0:
            new_memory.append(exp)
            profitable_count += 1
        elif action == 1 and len(new_memory) % int(1 / keep_hold_ratio) == 0:
            # Оставляем часть HOLD опытов
            new_memory.append(exp)
            hold_count += 1

    # Ограничиваем размер
    if len(new_memory) > max_memory:
        new_memory = new_memory[:max_memory]

    model.memory.clear()
    model.memory.extend(new_memory)

    # Очищаем приоритетный буфер
    if hasattr(model, 'prioritized_buffer'):
        model.prioritized_buffer = type(model.prioritized_buffer)(
            max_size=model.memory.maxlen,
            alpha=model.prioritized_buffer_config.get('alpha', 0.6),
            beta=model.prioritized_buffer_config.get('beta', 0.4)
        )
        for exp in new_memory:
            model.prioritized_buffer.add(exp)

    model.save_memory()

    print(f"\nРезультаты очистки:")
    print(f"  Прибыльных опытов сохранено: {profitable_count}")
    print(f"  HOLD опытов сохранено: {hold_count}")
    print(f"  Итоговый размер: {len(model.memory)}")
    print(f"  Удалено: {original_size - len(model.memory)}")
    print("\n✅ Память очищена и сохранена")


if __name__ == "__main__":
    clean_memory()