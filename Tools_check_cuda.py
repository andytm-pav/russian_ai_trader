"""
Проверка возможности перехода на CUDA (GPU)
"""
import sys
import torch

print("=" * 60)
print("🔍 ПРОВЕРКА CUDA")
print("=" * 60)

# 1. Доступность CUDA
cuda_available = torch.cuda.is_available()
print(f"\nCUDA доступна: {cuda_available}")

if cuda_available:
    print(f"Версия CUDA: {torch.version.cuda}")
    print(f"Количество GPU: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU #{i}: {props.name}")
        print(f"  Память: {props.total_mem / 1024 ** 3:.1f} ГБ")
        print(f"  Ядер: {props.multi_processor_count}")
        print(f"  Архитектура: sm_{props.major}{props.minor}")

    # Проверка текущего использования
    print(f"\nПамять занято: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} ГБ")
    print(f"Память зарезервировано: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} ГБ")
else:
    print("\nGPU не найден. Возможные причины:")
    print("  1. Драйверы NVIDIA не установлены")
    print("  2. PyTorch установлен без поддержки CUDA")
    print("  3. Видеокарта не поддерживает CUDA")

# 2. Версия PyTorch
print(f"\nPyTorch: {torch.__version__}")
print(f"Python:  {sys.version}")

# 3. Проверка, собран ли PyTorch с CUDA
if hasattr(torch, 'cuda') and hasattr(torch.cuda, 'is_built'):
    print(f"PyTorch собран с CUDA: {torch.cuda.is_built()}")

# 4. Рекомендация
print(f"\n{'=' * 60}")
print("РЕКОМЕНДАЦИЯ")
print(f"{'=' * 60}")

if cuda_available:
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024 ** 3
    print(f"GPU доступен ({gpu_mem:.1f} ГБ).")
    print(f"Для переключения измени в trader_model.py:")
    print(f"  self.device = torch.device('cuda')")
    print(f"Модель загрузится на GPU автоматически.")
else:
    print("GPU недоступен. Остаёмся на CPU.")
    print("Для установки поддержки CUDA:")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

print(f"\nГотово.")