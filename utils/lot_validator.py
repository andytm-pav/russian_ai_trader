"""
Валидатор лотности и кратности акций
"""

from typing import Tuple, Optional
from utils.logger import get_logger

logger = get_logger("LOT_VALIDATOR")


class LotValidator:
    """Класс для проверки и корректировки лотности"""

    @staticmethod
    def validate_and_adjust_quantity(quantity: int, lot_size: int) -> Tuple[int, bool]:
        """
        Проверка и корректировка количества по лотности

        Возвращает: (скорректированное_количество, требуется_корректировка)
        """
        # 🆕 v16.1: Принудительное приведение к int — float qty вызывал дробные продажи
        # Даже если приходит 0.45 или 1.5 — становится 0 или 1, потом валидатор корректно работает
        try:
            quantity = int(quantity)
        except (ValueError, TypeError):
            return 0, True

        if lot_size <= 1:
            return quantity, False

        if quantity % lot_size == 0:
            return quantity, False

        # Округляем до ближайшего кратного
        adjusted = (quantity // lot_size) * lot_size

        # Если после округления получился 0
        if adjusted == 0:
            # Проверяем, близко ли к половине лота
            if quantity >= lot_size / 2:
                adjusted = lot_size  # Берем 1 лот
                logger.debug(f"Количество {quantity} округлено до 1 лота ({adjusted})")
            else:
                logger.debug(f"Количество {quantity} меньше половины лота {lot_size}")
                return 0, True
        else:
            logger.debug(f"Количество {quantity} округлено до {adjusted} (лот: {lot_size})")

        return adjusted, True

    @staticmethod
    def validate_and_adjust_price(price: float, min_step: float) -> Tuple[float, bool]:
        """
        Проверка и корректировка цены по минимальному шагу

        Возвращает: (скорректированная_цена, требуется_корректировка)
        """
        if min_step <= 0 or price <= 0:
            return price, False

        if min_step >= price:  # Некорректный min_step
            return price, False

        # Проверяем кратность
        remainder = price % min_step
        if abs(remainder) < 1e-10:  # Учитываем погрешность float
            return price, False

        # Округляем до ближайшего кратного
        adjusted = round(price / min_step) * min_step

        # Проверяем значимость изменения
        change_pct = abs(adjusted - price) / price
        if change_pct > 0.01:  # Более 1% изменения
            logger.warning(f"Цена скорректирована на {change_pct * 100:.2f}%: {price:.4f} → {adjusted:.4f}")

        return adjusted, True

    @staticmethod
    def calculate_min_lot_value(price: float, lot_size: int) -> float:
        """Расчет минимальной стоимости лота"""
        return price * lot_size

    @staticmethod
    def get_optimal_quantity(desired_quantity: int,
                             lot_size: int,
                             max_quantity: Optional[int] = None) -> int:
        """Получение оптимального количества с учетом лотности и ограничений"""
        if lot_size <= 1:
            return desired_quantity

        # Округляем до ближайшего кратного
        quantity = (desired_quantity // lot_size) * lot_size

        # Если получился 0, пробуем взять 1 лот если это в пределах max_quantity
        if quantity == 0 and desired_quantity >= lot_size / 2:
            quantity = lot_size

        # Применяем ограничение максимума
        if max_quantity is not None:
            # Округляем max_quantity тоже по лотности
            max_adjusted = (max_quantity // lot_size) * lot_size
            quantity = min(quantity, max_adjusted)

        return quantity