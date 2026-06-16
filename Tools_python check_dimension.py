"""
Проверка реальной размерности модели (без конфигов)
"""
import json
import numpy as np
from models.trader_model import trader_model_instance

model = trader_model_instance

print("=" * 60)
print("🔍 ПРОВЕРКА РАЗМЕРНОСТИ МОДЕЛИ")
print("=" * 60)

# 1. Реальная размерность из загруженной модели
actual_state_dim = model.policy_net.state_dim
actual_action_dim = model.policy_net.action_dim

# 2. Ожидаемая размерность из конфига (для сравнения)
with open("config/rl_config.json", "r", encoding="utf-8") as f:
    rl_config = json.load(f)

state_params = rl_config.get("state_parameters", {})
config_state_dim = state_params.get("state_vector_size", 210)
config_total_dim = state_params.get("total_state_size", 216)

# 3. Размерность из кода (как её видит __init__)
code_base_dim = model.base_state_dim
code_total_dim = model.total_state_dim

print(f"\n{'Источник':<30} {'state_dim':<15} {'total_dim':<15}")
print("-" * 60)
print(f"{'policy_net (реальная)':<30} {actual_state_dim:<15} —")
print(f"{'model.base_state_dim':<30} {code_base_dim:<15} —")
print(f"{'model.total_state_dim':<30} —               {code_total_dim:<15}")
print(f"{'rl_config.json':<30} {config_state_dim:<15} {config_total_dim:<15}")

# 4. Проверка совпадения
print(f"\n{'='*60}")
print("ПРОВЕРКА СОВПАДЕНИЯ")
print(f"{'='*60}")

errors = []

if actual_state_dim != code_total_dim:
    errors.append(f"policy_net ожидает {actual_state_dim}, но total_state_size = {code_total_dim}")

if code_base_dim != config_state_dim:
    errors.append(f"base_state_dim ({code_base_dim}) != state_vector_size в конфиге ({config_state_dim})")

if code_total_dim != config_total_dim:
    errors.append(f"total_state_dim ({code_total_dim}) != total_state_size в конфиге ({config_total_dim})")

if errors:
    print("❌ НЕСОВПАДЕНИЯ:")
    for e in errors:
        print(f"   - {e}")
else:
    print("✅ Все размерности совпадают")

# 5. Размерность опыта в памяти
if len(model.memory) > 0:
    sample = list(model.memory)[-1]
    memory_state_dim = sample['state'].shape[0]
    print(f"\nРазмерность опыта в памяти: {memory_state_dim}")
    if memory_state_dim != code_total_dim:
        print(f"⚠️ Память ({memory_state_dim}) не совпадает с моделью ({code_total_dim}) — нужен сброс")
else:
    print(f"\nПамять пуста — не с чем сравнивать")

# 6. Веса первого слоя
first_layer = model.policy_net.state_net[0]
weight_shape = first_layer.weight.shape
print(f"\nВеса первого слоя policy_net: {weight_shape}")
print(f"  Ожидаемый вход: {weight_shape[1]} (должен быть = total_state_size)")

print(f"\nГотово.")