#!/usr/bin/env python3
"""
Тест: получение актуальных макро-данных через MoexFetcher
"""
import sys
sys.path.insert(0, '.')
from fetchers.moex_fetcher import MoexFetcher

moex = MoexFetcher()
macro = moex.get_macro_data()

print("=" * 60)
print("АКТУАЛЬНЫЕ МАКРО-ДАННЫЕ С MOEX")
print("=" * 60)
print(f"IMOEX:      {macro.get('imoex', 0):.2f}")
print(f"RTSI:       {macro.get('rtsi', 0):.2f}")
print(f"RVI:        {macro.get('rvi', 0):.2f}")
print(f"Brent:      {macro.get('brent', 0):.2f}")
print(f"USD/RUB:    {macro.get('usd_rub', 0):.2f}")
print(f"ЦБ ставка:  {macro.get('cbr_rate', 0):.2f}")
print(f"VIX:        {macro.get('vix', 0):.2f}")
print(f"MOEXOG:     {macro.get('moexog', 0):.2f}")
print(f"MOEXFN:     {macro.get('moexfn', 0):.2f}")