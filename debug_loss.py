import json
import numpy as np

# Ваши реальные данные из лога
test_data = {
    'td_errors': [-112.58, -161.28, -339.12, -235.34, -1038.44],
    'pnl_values': [-45.42, -83.73, -193.45, -500.0, -397.41,
                   -107.0, -50.0, -92.4, -12.49, -4.58,
                   -63.72, -120.62, -147.82, -30.73, 10.54,
                   -500.0, -39.6, -16.24, -125.37, 1.19,
                   -50.0, -0.956, -9.91, -49.18, -91.94,
                   -481.53, -50.0, -81.31, -33.96, -81.42,
                   -78.21, -246.62],
    'loss': -1123.116943,
    'accuracy': 93.75,
    'reward_scaling': 0.02  # из вашего конфига
}

print("=" * 60)
print("ДИАГНОСТИКА ПРОБЛЕМЫ LOSS")
print("=" * 60)

# 1. Анализ PnL значений
pnl_array = np.array(test_data['pnl_values'])
print(f"\n1. АНАЛИЗ PnL В BATCH:")
print(f"   Всего значений: {len(pnl_array)}")
print(f"   Минимум: {pnl_array.min():.2f}")
print(f"   Максимум: {pnl_array.max():.2f}")
print(f"   Среднее: {pnl_array.mean():.2f}")
print(f"   Медиана: {np.median(pnl_array):.2f}")
print(f"   Стандартное отклонение: {pnl_array.std():.2f}")

# 2. Проверка масштаба
print(f"\n2. ПРОВЕРКА МАСШТАБА:")
print(f"   95% значений между: {np.percentile(pnl_array, 2.5):.2f} и {np.percentile(pnl_array, 97.5):.2f}")

# 3. Сравнение с td_errors
td_array = np.array(test_data['td_errors'])
print(f"\n3. СВЯЗЬ PnL И TD-ERRORS:")
print(f"   Средний |td_error|: {np.abs(td_array).mean():.2f}")
print(f"   Средний |pnl|: {np.abs(pnl_array).mean():.2f}")
print(f"   Соотношение |td_error|/|pnl|: {np.abs(td_array).mean() / np.abs(pnl_array).mean():.2f}")

# 4. Что должно быть при правильном масштабе
print(f"\n4. РАСЧЕТ ПРАВИЛЬНОГО МАСШТАБА:")
# Предполагаем, что pnl_array - это рубли
correct_pnl_percent = pnl_array / 1000  # допустим, средняя цена 1000₽
correct_reward = correct_pnl_percent * test_data['reward_scaling']
print(f"   Если pnl в рублях, то:")
print(f"   Средний pln%: {correct_pnl_percent.mean() * 100:.2f}%")
print(f"   Средний reward (правильный): {correct_reward.mean():.4f}")
print(f"   Средний reward (сейчас в логе): {pnl_array.mean():.2f}")

# 5. Имитация обучения
print(f"\n5. ИМИТАЦИЯ ОБУЧЕНИЯ:")
# Создаем синтетические current_values (маленькие, т.к. сеть только учится)
current_values = np.random.normal(0, 1, len(td_array)) * 10
target_values = current_values + td_array

print(f"   current_values mean: {current_values.mean():.2f}")
print(f"   target_values mean: {target_values.mean():.2f}")
print(f"   Разница target-current: {(target_values - current_values).mean():.2f}")

# 6. Диагностика
print(f"\n6. ВЫВОД:")
if np.abs(td_array).mean() > 100:
    print("   🔴 TD-ERRORS СЛИШКОМ БОЛЬШИЕ (>100)")
    print("      Причина: rewards (pnl * scaling) дают огромные значения")

if np.abs(pnl_array).mean() > 50:
    print("   🔴 PnL В БАТЧЕ СЛИШКОМ БОЛЬШИЕ (>50)")
    print("      Это точно РУБЛИ, а не ПРОЦЕНТЫ!")
    print("      Нужно делить на цену акции или использовать price_change_ratio")

if test_data['loss'] < 0:
    print("   🔴 LOSS ОТРИЦАТЕЛЬНЫЙ")
    print("      Entropy bonus перевешивает основные компоненты loss")

print("\n" + "=" * 60)
print("РЕКОМЕНДАЦИИ:")
print("=" * 60)
print("""
1. В remember_experience СОХРАНЯЙТЕ и reward (проценты) и pnl_abs (рубли)
2. В learn_from_experience ИСПОЛЬЗУЙТЕ price_change_ratio для классов цены
3. УБЕДИТЕСЬ что pnl в процентах (0.xx), а не в рублях (100.500)
4. Добавьте НОРМАЛИЗАЦИЮ target_values: target_values = target_values / 100
""")