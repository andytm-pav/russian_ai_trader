#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ПОЛНЫЙ ПОИСК ВСЕХ ИНДЕКСОВ
"""

import requests


def find_all_indices():
    print("=" * 60)
    print("ПОИСК ВСЕХ ДОСТУПНЫХ ИНДЕКСОВ")
    print("=" * 60)

    # 1. Индексы из boards/SNDX (IMOEX, MOEXBMI, и др.)
    url1 = "https://iss.moex.com/iss/engines/stock/markets/index/boards/SNDX/securities.json"
    params1 = {
        'iss.meta': 'off',
        'iss.only': 'marketdata',
        'marketdata.columns': 'SECID,CURRENTVALUE,LASTCHANGE,LASTCHANGEPRC'
    }

    print(f"\n📡 SNDX индексы:")
    response = requests.get(url1, params=params1)
    data = response.json()

    sndx_indices = []
    for row in data['marketdata']['data']:
        ticker = row[0]
        if ticker in ['IMOEX', 'IMOEX2', 'MOEXBMI', 'MOEXFN', 'MOEXOG', 'MOEXMM', 'MOEXCN', 'MOEXTL']:
            value = row[1]
            change = row[2]
            change_pct = row[3]
            print(f"   ✅ {ticker}: {value} (изм: {change_pct}%)")
            sndx_indices.append(ticker)

    # 2. Отдельные индексы (RTSI, RVI)
    print(f"\n📡 Отдельные индексы:")

    # RTSI
    url2 = "https://iss.moex.com/iss/engines/stock/markets/index/securities/RTSI.json"
    params2 = {
        'iss.meta': 'off',
        'iss.only': 'marketdata',
        'marketdata.columns': 'SECID,CURRENTVALUE,LASTCHANGE,LASTCHANGEPRC'
    }
    response = requests.get(url2, params=params2)
    data = response.json()
    row = data['marketdata']['data'][0]
    print(f"   ✅ {row[0]}: {row[1]} (изм: {row[3]}%)")

    # RVI
    url3 = "https://iss.moex.com/iss/engines/stock/markets/index/securities/RVI.json"
    response = requests.get(url3, params=params2)
    data = response.json()
    row = data['marketdata']['data'][0]
    print(f"   ✅ {row[0]}: {row[1]} (изм: {row[3]}%)")

    print("\n" + "=" * 60)
    print(f"✅ ВСЕГО НАЙДЕНО: {len(sndx_indices) + 2} индексов")


if __name__ == "__main__":
    find_all_indices()