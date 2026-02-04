"""
Исправленный скрипт для создания config/tickers.json
Все данные получаются с MOEX API
"""

import json
import requests
from datetime import datetime
import time

# ВАШ СПИСОК ТИКЕРОВ
TICKERS_TO_ADD = [
    "SBER",   # Сбербанк
    "GAZP",   # Газпром
    "LKOH",   # Лукойл
    "YNDX",   # Яндекс
    "VTBR",   # ВТБ
    "ROSN",   # Роснефть
    "GMKN",   # Норникель
    "NVTK",   # Новатэк
    "PLZL",   # Полюс
    "POLY",   # Polymetal
    "MTSS",   # МТС
]

def get_ticker_info_from_marketdata(ticker):
    """Получение информации о тикере через marketdata API MOEX"""
    try:
        # URL для получения текущих данных
        url = f"https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/{ticker}.json"
        params = {
            'iss.meta': 'off',
            'iss.only': 'securities,marketdata'
        }

        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            print(f"  HTTP {response.status_code} для {ticker}")
            return None

        data = response.json()

        # Получаем данные из раздела securities
        securities_data = {}
        if 'securities' in data and data['securities']['data']:
            columns = data['securities']['columns']
            row = data['securities']['data'][0]

            # Маппинг колонок
            col_index = {col: idx for idx, col in enumerate(columns)}

            securities_data = {
                'ticker': ticker,
                'shortname': row[col_index['SHORTNAME']] if 'SHORTNAME' in col_index else ticker,
                'secname': row[col_index['SECNAME']] if 'SECNAME' in col_index else ticker,
                'lot_size': int(row[col_index['LOTSIZE']]) if 'LOTSIZE' in col_index and row[col_index['LOTSIZE']] else 1,
                'minstep': float(row[col_index['MINSTEP']]) if 'MINSTEP' in col_index and row[col_index['MINSTEP']] else 0.01,
                'prevprice': float(row[col_index['PREVPRICE']]) if 'PREVPRICE' in col_index and row[col_index['PREVPRICE']] else 0,
            }

        # Получаем данные из раздела marketdata (текущие цены)
        marketdata = {}
        if 'marketdata' in data and data['marketdata']['data']:
            columns = data['marketdata']['columns']
            row = data['marketdata']['data'][0]
            col_index = {col: idx for idx, col in enumerate(columns)}

            marketdata = {
                'last': float(row[col_index['LAST']]) if 'LAST' in col_index and row[col_index['LAST']] else securities_data.get('prevprice', 0),
                'open': float(row[col_index['OPEN']]) if 'OPEN' in col_index and row[col_index['OPEN']] else 0,
                'low': float(row[col_index['LOW']]) if 'LOW' in col_index and row[col_index['LOW']] else 0,
                'high': float(row[col_index['HIGH']]) if 'HIGH' in col_index and row[col_index['HIGH']] else 0,
                'valtoday': float(row[col_index['VALTODAY']]) if 'VALTODAY' in col_index and row[col_index['VALTODAY']] else 0,
                'voltoday': int(row[col_index['VOLTODAY']]) if 'VOLTODAY' in col_index and row[col_index['VOLTODAY']] else 0,
            }

        # Определяем сектор по названию
        name_lower = securities_data.get('shortname', '').lower()
        sector = "other"
        priority = 5

        if any(word in name_lower for word in ['банк', 'финанс', 'инвест', 'капитал', 'сбер', 'втб']):
            sector = "финансы"
            priority = 9
        elif any(word in name_lower for word in ['нефть', 'газ', 'энерг', 'ресурс', 'лукойл', 'роснефть']):
            sector = "нефтегаз"
            priority = 8
        elif any(word in name_lower for word in ['металл', 'золот', 'никел', 'алюмин', 'полюс', 'норни']):
            sector = "металлы"
            priority = 7
        elif any(word in name_lower for word in ['техн', 'софт', 'интернет', 'it', 'яндекс', 'mail']):
            sector = "IT"
            priority = 8
        elif any(word in name_lower for word in ['связь', 'телеком', 'мтс']):
            sector = "телеком"
            priority = 6

        # Рассчитываем средний дневной объем (используем VOLTODAY как пример)
        avg_daily_volume = marketdata.get('voltoday', 0)
        if avg_daily_volume == 0:
            avg_daily_volume = 1000000  # Дефолт

        info = {
            'ticker': ticker,
            'name': securities_data.get('shortname', ticker),
            'full_name': securities_data.get('secname', ticker),
            'sector': sector,
            'lot_size': securities_data.get('lot_size', 1),
            'min_step': securities_data.get('minstep', 0.01),
            'avg_daily_volume': avg_daily_volume,
            'priority': priority,
            'current_price': marketdata.get('last', 0),
            'market_data': {
                'open': marketdata.get('open', 0),
                'high': marketdata.get('high', 0),
                'low': marketdata.get('low', 0),
                'volume_today': marketdata.get('voltoday', 0),
                'value_today': marketdata.get('valtoday', 0)
            }
        }

        return info

    except Exception as e:
        print(f"  Ошибка получения данных для {ticker}: {str(e)[:100]}")
        return None

def get_ticker_info_batch(tickers):
    """Получение информации о нескольких тикерах через batch запрос"""
    try:
        if not tickers:
            return []

        # Используем batch запрос к securities.json
        url = "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities.json"
        params = {
            'securities': ','.join(tickers),
            'iss.meta': 'off',
            'iss.only': 'securities',
            'securities.columns': 'SECID,SHORTNAME,SECNAME,LOTSIZE,MINSTEP,PREVPRICE',
            'limit': len(tickers)
        }

        response = requests.get(url, params=params, timeout=15)
        if response.status_code != 200:
            print(f"  Batch запрос HTTP {response.status_code}")
            return []

        data = response.json()

        if 'securities' not in data:
            return []

        columns = data['securities']['columns']
        col_index = {col: idx for idx, col in enumerate(columns)}
        results = []

        for row in data['securities']['data']:
            try:
                ticker = row[col_index['SECID']]
                shortname = row[col_index['SHORTNAME']] if col_index.get('SHORTNAME') is not None and row[col_index['SHORTNAME']] else ticker
                secname = row[col_index['SECNAME']] if col_index.get('SECNAME') is not None and row[col_index['SECNAME']] else shortname

                # Получаем lot_size и min_step
                lot_size = 1
                if col_index.get('LOTSIZE') is not None and row[col_index['LOTSIZE']]:
                    try:
                        lot_size = int(row[col_index['LOTSIZE']])
                    except:
                        lot_size = 1

                min_step = 0.01
                if col_index.get('MINSTEP') is not None and row[col_index['MINSTEP']]:
                    try:
                        min_step = float(row[col_index['MINSTEP']])
                    except:
                        min_step = 0.01

                # Определяем сектор
                sector = "other"
                priority = 5
                name_lower = shortname.lower()

                if any(word in name_lower for word in ['банк', 'финанс', 'инвест', 'капитал', 'сбер', 'втб']):
                    sector = "финансы"
                    priority = 9
                elif any(word in name_lower for word in ['нефть', 'газ', 'энерг', 'ресурс', 'лукойл', 'роснефть', 'газпром', 'новатэк']):
                    sector = "нефтегаз"
                    priority = 8
                elif any(word in name_lower for word in ['металл', 'золот', 'никел', 'алюмин', 'полюс', 'норни', 'полиметал']):
                    sector = "металлы"
                    priority = 7
                elif any(word in name_lower for word in ['техн', 'софт', 'интернет', 'it', 'яндекс', 'mail']):
                    sector = "IT"
                    priority = 8
                elif any(word in name_lower for word in ['связь', 'телеком', 'мтс']):
                    sector = "телеком"
                    priority = 6

                info = {
                    'ticker': ticker,
                    'name': shortname,
                    'full_name': secname,
                    'sector': sector,
                    'lot_size': lot_size,
                    'min_step': min_step,
                    'avg_daily_volume': 1000000,  # Дефолтное значение
                    'priority': priority
                }

                results.append(info)

            except Exception as e:
                print(f"  Ошибка обработки строки: {e}")
                continue

        return results

    except Exception as e:
        print(f"  Ошибка batch запроса: {e}")
        return []

def create_config():
    """Создание конфигурационного файла"""

    print("=" * 50)
    print("Создание конфигурации тикеров из MOEX API")
    print(f"Будет добавлено {len(TICKERS_TO_ADD)} тикеров")
    print("Тикеры:", ", ".join(TICKERS_TO_ADD))
    print("=" * 50)

    # Пробуем batch запрос
    print("Попытка batch запроса...")
    watchlist = get_ticker_info_batch(TICKERS_TO_ADD)

    if len(watchlist) < len(TICKERS_TO_ADD):
        print(f"\nBatch запрос получил только {len(watchlist)}/{len(TICKERS_TO_ADD)} тикеров")
        print("Пробуем индивидуальные запросы для отсутствующих...")

        # Находим какие тикеры не получены
        received_tickers = {item['ticker'] for item in watchlist}
        missing_tickers = [t for t in TICKERS_TO_ADD if t not in received_tickers]

        # Индивидуальные запросы для недостающих
        for ticker in missing_tickers:
            print(f"  Индивидуальный запрос для {ticker}...")
            info = get_ticker_info_from_marketdata(ticker)
            if info:
                watchlist.append(info)
                print(f"    ✓ Получен: {info['name']} (лот: {info['lot_size']})")
            else:
                print(f"    ✗ Не удалось получить {ticker}")
                # Добавляем с дефолтными значениями
                watchlist.append({
                    'ticker': ticker,
                    'name': ticker,
                    'full_name': ticker,
                    'sector': 'other',
                    'lot_size': 1,
                    'min_step': 0.01,
                    'avg_daily_volume': 1000000,
                    'priority': 5
                })

    # Сортируем по приоритету
    watchlist.sort(key=lambda x: x['priority'], reverse=True)

    # Создаем конфиг
    config = {
        'watchlist': watchlist,
        'sector_limits': {
            'финансы': 40,
            'нефтегаз': 30,
            'металлы': 20,
            'IT': 15,
            'телеком': 15,
            'other': 10
        },
        'liquidity_filter': {
            'min_avg_volume': 1000000,
            'min_price': 10,
            'max_spread_percent': 0.5
        },
        'trading_params': {
            'max_position_per_ticker_percent': 25,
            'min_hold_period_hours': 4,
            'cooldown_period_hours': 2
        },
        'metadata': {
            'created': datetime.now().isoformat(),
            'total_tickers': len(watchlist),
            'sectors': list(set([t['sector'] for t in watchlist])),
            'source': 'MOEX ISS API',
            'note': f'Создано автоматически. Тикеры: {", ".join(TICKERS_TO_ADD)}'
        }
    }

    # Сохраняем в файл
    with open('config/tickers.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print(f"УСПЕХ: Создан файл config/tickers.json")
    print(f"Добавлено тикеров: {len(watchlist)}/{len(TICKERS_TO_ADD)}")

    # Статистика
    sectors = {}
    for ticker in watchlist:
        sector = ticker['sector']
        sectors[sector] = sectors.get(sector, 0) + 1

    print("\nСтатистика по секторам:")
    for sector, count in sorted(sectors.items(), key=lambda x: x[1], reverse=True):
        print(f"  {sector}: {count} тикеров")

    print("\nИнформация о тикерах:")
    for ticker in watchlist:
        print(f"  {ticker['ticker']}: {ticker['name']} | "
              f"Сектор: {ticker['sector']} | "
              f"Лот: {ticker['lot_size']} | "
              f"Шаг: {ticker['min_step']} | "
              f"Приоритет: {ticker['priority']}")

    print("\nФайл готов к использованию!")
    print("=" * 50)

if __name__ == "__main__":
    create_config()
