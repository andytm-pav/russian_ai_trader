import apimoex
import requests
import pandas as pd

with requests.Session() as session:
    # Получение исторических данных по акции 'SBER' на основной сессии 'TQBR'
    data = apimoex.get_board_history(session, 'SBER')
    df = pd.DataFrame(data)
    if not df.empty:
        df.set_index('TRADEDATE', inplace=True)
        print(df.head())