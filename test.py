#!/usr/bin/env python3
"""
Конвертер: ticker_sectors (2).json → config/ticker_sectors.json
Автоматически группирует сектора по общим словам.
Английские названия переводятся на русский через словарь.
Сами сектора берутся ТОЛЬКО из исходного файла.
"""

import json
import re
from collections import OrderedDict, defaultdict


# Единственный хардкод — перевод английских названий секторов
EN_TO_RU = {
    "Communication Services": "Телеком",
    "Consumer Cyclical": "Потребительские товары",
    "Consumer Discretionary": "Потребительские товары",
    "Consumer Staples": "Потребительские товары",
    "ETF": "ETF",
    "Energy": "Нефть и газ",
    "Financials": "Финансы",
    "HIGH TECH": "Технологии",
    "Healthcare": "Фармацевтика",
    "Industrials": "Промышленность",
    "Materials": "Металлургия",
    "Other": "Разное",
    "Real Estate": "Недвижимость",
    "Technology": "Технологии",
    "Telecom": "Телеком",
    "Utilities": "Энергетика",
}


def clean_name(name: str) -> set:
    """Очищает название и возвращает множество значимых слов"""
    name = name.lower()
    name = re.sub(r'[&;,./()"\'«»]', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    words = set(name.split())
    stop_words = {
        'ао', 'в', 'и', 'с', 'на', 'по', 'от', 'до', 'за', 'из', 'к', 'о', 'у',
        'не', 'то', 'а', 'it', 'of', 'the', 'and', 'in', 'for', 'inc', 'ltd',
        'ап', 'old', 'new', 'plc', 'se', 'ag', 'de', 'group', 'limited', 'corp',
        'corporation', 'company', 'co', 'holding', 'holdings', 'international',
        'эш', 'разное', 'компании', 'компания', 'добавить', 'черн', 'цвет'
    }
    return {w for w in words if len(w) >= 3 and w not in stop_words}


def is_russian(s: str) -> bool:
    return bool(re.search(r'[а-яё]', s.lower()))


def main():
    with open("ticker_sectors (2).json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Собираем тикеры и сырые сектора
    ticker_raw_sectors = {}
    raw_sectors_set = set()

    for entry in raw_data:
        ticker = entry.get("ticker", "").strip()
        raw_sector = entry.get("sector", "").strip()
        if not ticker or not raw_sector:
            continue
        if "/" in ticker or ticker in ["gr", "...."]:
            continue
        ticker_raw_sectors[ticker] = raw_sector
        raw_sectors_set.add(raw_sector)

    # Шаг 1: Автоматическая группировка по общим словам
    raw_sectors_list = sorted(raw_sectors_set)
    sector_words = {s: clean_name(s) for s in raw_sectors_list}

    graph = defaultdict(set)
    for i, s1 in enumerate(raw_sectors_list):
        for s2 in raw_sectors_list[i + 1:]:
            if sector_words[s1] & sector_words[s2]:
                graph[s1].add(s2)
                graph[s2].add(s1)

    visited = set()
    clusters = []
    for sector in raw_sectors_list:
        if sector in visited:
            continue
        stack = [sector]
        cluster = []
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            cluster.append(node)
            stack.extend(graph[node] - visited)
        if cluster:
            clusters.append(cluster)

    # Назначаем имена кластерам
    cluster_names = {}
    for cluster in clusters:
        russian = [s for s in cluster if is_russian(s)]
        if russian:
            representative = max(russian, key=lambda s: len(clean_name(s)))
        else:
            representative = max(cluster, key=lambda s: len(clean_name(s)))
        for sector in cluster:
            cluster_names[sector] = representative

    for sector in raw_sectors_list:
        if sector not in cluster_names:
            cluster_names[sector] = sector

    # Шаг 2: Перевод английских названий
    for raw_sector, grouped_name in cluster_names.items():
        if grouped_name in EN_TO_RU:
            cluster_names[raw_sector] = EN_TO_RU[grouped_name]

    # Шаг 3: Группировка русских синонимов после перевода
    # Собираем финальные группы
    final_groups = defaultdict(set)
    for raw_sector, name in cluster_names.items():
        final_groups[name].add(raw_sector)

    # Объединяем похожие русские названия
    merge_map = {}
    russian_names = sorted(final_groups.keys())

    for i, n1 in enumerate(russian_names):
        for n2 in russian_names[i + 1:]:
            if clean_name(n1) & clean_name(n2):
                # Находим корневую группу
                root1 = merge_map.get(n1, n1)
                root2 = merge_map.get(n2, n2)
                if root1 != root2:
                    # Приоритет: более длинное русское название
                    if is_russian(root1) and is_russian(root2):
                        winner = root1 if len(clean_name(root1)) >= len(clean_name(root2)) else root2
                    elif is_russian(root1):
                        winner = root1
                    else:
                        winner = root2
                    for name in list(merge_map.keys()):
                        if merge_map[name] in [root1, root2]:
                            merge_map[name] = winner
                    merge_map[root1] = winner
                    merge_map[root2] = winner

    for name in russian_names:
        if name not in merge_map:
            merge_map[name] = name

    # Строим финальный маппинг
    final_cluster_names = {}
    for raw_sector, name in cluster_names.items():
        final_cluster_names[raw_sector] = merge_map.get(name, name)

    # Итоговый словарь тикер → сектор
    sectors = {}
    for ticker, raw_sector in ticker_raw_sectors.items():
        sectors[ticker] = final_cluster_names[raw_sector]

    sectors = OrderedDict(sorted(sectors.items()))

    # Назначаем цвета
    unique_sectors = sorted(set(sectors.values()))
    color_palette = [
        "#00ff88", "#0088ff", "#ff4444", "#ffaa00", "#aa00ff",
        "#00cc66", "#00aaff", "#ff8800", "#ff00aa", "#88ff00",
        "#ff6600", "#00ffcc", "#cc00ff", "#ffff00", "#888888"
    ]
    sector_colors = {}
    for i, sector in enumerate(unique_sectors):
        sector_colors[sector] = color_palette[i % len(color_palette)]

    result = {
        "sectors": sectors,
        "sector_colors": sector_colors
    }

    with open("config/ticker_sectors.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✅ Сформирован config/ticker_sectors.json")
    print(f"   Тикеров: {len(sectors)}")
    print(f"   Секторов после группировки: {len(unique_sectors)}")
    for sector in sorted(unique_sectors):
        count = sum(1 for s in sectors.values() if s == sector)
        print(f"   {sector}: {count} тикеров")


if __name__ == "__main__":
    main()