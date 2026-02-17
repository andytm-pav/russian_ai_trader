import pickle

# Какая версия протокола используется по умолчанию?
print(f"Протокол по умолчанию: {pickle.DEFAULT_PROTOCOL}")

# Какая самая новая версия протокола доступна?
print(f"Высший доступный протокол: {pickle.HIGHEST_PROTOCOL}")

# Какая версия Python используется?
import sys
print(f"Версия Python: {sys.version}")