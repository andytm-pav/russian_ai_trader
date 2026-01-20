"""
Универсальный монитор настроения российского рынка акций.
Анализирует новости для портфеля и наиболее упоминаемых тикеров.
"""

import json
import time
import threading
import logging
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional
import dash
from dash import Dash, dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px

# Импорт модулей проекта
try:
    from utils.portfolio_manager import PortfolioManager
    from core.core_news_trader import NewsTraderCore
    from fetchers.rss_fetcher import RSSFetcher
except ImportError as e:
    print(f"Ошибка импорта модулей проекта: {e}. Убедитесь, что структура проекта корректна.")
    # Заглушки для отладки
    class PortfolioManager:
        def __init__(self): self.positions = {'SBER': {}, 'GAZP': {}, 'VTBR': {}}
    class NewsTraderCore:
        def get_current_sentiment(self, ticker): return 0.0
        def get_market_sentiment(self): return {'sentiment': 0.0, 'trend': 'neutral'}
    class RSSFetcher:
        def fetch_all_news(self): return []

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RussianMarketMonitor")

class RussianMarketMonitor:
    """
    Монитор российского рынка акций, ориентированный на ключевые факторы влияния:
    - Геополитика и санкции [citation:1][citation:4][citation:5]
    - Ключевая ставка ЦБ и монетарная политика [citation:1]
    - Динамика сырьевых рынков (нефть, металлы) [citation:1][citation:5]
    - Корпоративные новости и дивиденды [citation:1][citation:5]
    """

    # Ключевые сектора российского рынка и их эталонные тикеры [citation:1][citation:3]
    MARKET_SECTORS = {
        'finance': ['SBER', 'VTBR', 'TCSG'],
        'oil_gas': ['GAZP', 'LKOH', 'ROSN', 'NVTK', 'SNGS'],
        'metals': ['GMKN', 'NLMK', 'PLZL', 'RUAL', 'CHMF', 'MAGN'],
        'electricity': ('FEES', 'HYDR', 'TGKA'),
        'retail': ('MGNT', 'FIXP', 'DSKY'),
        'it_telecom': ('YNDX', 'MTSS', 'IRAO')
    }

    def __init__(self):
        """Инициализация монитора с загрузкой данных проекта."""
        self.portfolio = PortfolioManager()
        self.news_core = NewsTraderCore()
        self.rss_fetcher = RSSFetcher()

        # Основные данные
        self.market_sentiment_history = []
        self.news_flow = []
        self.sector_sentiment = {sector: [] for sector in self.MARKET_SECTORS.keys()}

        # Загрузка эталонных тикеров из проекта
        self.reference_tickers = self._load_reference_tickers()
        logger.info(f"Монитор инициализирован. Отслеживается {len(self.reference_tickers)} тикеров.")

        # Старт фонового обновления
        self._start_background_updater()

    def _load_reference_tickers(self) -> List[str]:
        """
        Формирует список тикеров для анализа.
        Приоритет: 1) портфель, 2) секторальные лидеры, 3) часто упоминаемые в новостях.
        """
        tickers_set = set()

        # 1. Тикеры из портфеля
        portfolio_tickers = list(self.portfolio.positions.keys())
        tickers_set.update(portfolio_tickers)

        # 2. Ключевые тикеры по секторам [citation:1][citation:3]
        for sector_tickers in self.MARKET_SECTORS.values():
            tickers_set.update(sector_tickers)

        # 3. Можно добавить загрузку из config/tickers.json
        try:
            with open('config/tickers.json', 'r', encoding='utf-8') as f:
                config_tickers = json.load(f).get('watchlist', [])
                tickers_set.update([item['ticker'] for item in config_tickers])
        except FileNotFoundError:
            logger.warning("Файл config/tickers.json не найден. Используются тикеры по умолчанию.")

        # 4. Тикеры из MOEX Broad Market Index (широкий охват) [citation:6]
        moexbm_tickers = ['SBER', 'GAZP', 'LKOH', 'NVTK', 'GMKN', 'ROSN', 'VTBR']
        tickers_set.update(moexbm_tickers)

        return list(tickers_set)

    def _start_background_updater(self):
        """Запускает фоновый поток для обновления данных."""
        def update_loop():
            while True:
                try:
                    self._update_market_data()
                except Exception as e:
                    logger.error(f"Ошибка в фоновом обновлении: {e}")
                time.sleep(300)  # Обновление каждые 5 минут

        thread = threading.Thread(target=update_loop, daemon=True)
        thread.start()
        logger.info("Фоновое обновление данных запущено.")

    def _update_market_data(self):
        """Основной метод сбора и анализа данных."""
        logger.info("Начало цикла обновления данных...")

        # 1. Сбор новостей
        all_news = self.rss_fetcher.fetch_all_news()
        self.news_flow = all_news[-50:]  # Последние 50 новостей

        # 2. Анализ упоминаний и сентимента
        current_time = datetime.now().isoformat()
        mention_counter = Counter()
        ticker_sentiments = {}

        for news_item in self.news_flow:
            text = (news_item.get('title', '') + ' ' + news_item.get('summary', '')).lower()

            # Поиск упоминаний тикеров
            for ticker in self.reference_tickers:
                if ticker.lower() in text:
                    mention_counter[ticker] += 1

        # 3. Расчет сентимента для наиболее упоминаемых тикеров
        top_mentioned = mention_counter.most_common(15)
        for ticker, count in top_mentioned:
            try:
                sentiment = self.news_core.get_current_sentiment(ticker)
                ticker_sentiments[ticker] = {
                    'sentiment': sentiment,
                    'mentions': count,
                    'last_updated': current_time
                }
            except Exception as e:
                logger.debug(f"Не удалось получить сентимент для {ticker}: {e}")

        # 4. Обновление истории рыночного сентимента
        market_data = self.news_core.get_market_sentiment()
        self.market_sentiment_history.append({
            'timestamp': current_time,
            'sentiment': market_data.get('sentiment', 0.0),
            'trend': market_data.get('trend', 'neutral')
        })

        # Сохранение последних 100 точек
        self.market_sentiment_history = self.market_sentiment_history[-100:]

        logger.info(f"Данные обновлены. Новостей: {len(self.news_flow)}, "
                    f"упомянуто тикеров: {len(mention_counter)}")

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Формирует структурированные данные для дашборда."""
        # Топ тикеров по упоминаниям
        mention_counter = Counter()
        for news in self.news_flow:
            text = (news.get('title', '') + ' ' + news.get('summary', '')).lower()
            for ticker in self.reference_tickers:
                if ticker.lower() in text:
                    mention_counter[ticker] += 1

        top_mentioned = mention_counter.most_common(10)

        # Данные для графиков
        chart_data = []
        for ticker, count in top_mentioned:
            sentiment_data = self.news_core.get_current_sentiment(ticker)
            chart_data.append({
                'ticker': ticker,
                'mentions': count,
                'sentiment': sentiment_data,
                'in_portfolio': ticker in self.portfolio.positions
            })

        return {
            'market_overview': {
                'last_update': datetime.now().isoformat(),
                'total_news': len(self.news_flow),
                'tracked_tickers': len(self.reference_tickers),
                'portfolio_tickers': len(self.portfolio.positions),
                'market_sentiment': self.news_core.get_market_sentiment()
            },
            'top_tickers': chart_data,
            'sector_overview': self._get_sector_overview(),
            'recent_news': self.news_flow[-5:] if self.news_flow else [],
            'sentiment_history': self.market_sentiment_history[-20:]  # Последние 20 точек
        }

    def _get_sector_overview(self) -> List[Dict]:
        """Анализ сентимента по секторам."""
        sector_data = []
        for sector, tickers in self.MARKET_SECTORS.items():
            sector_tickers = [t for t in tickers if t in self.reference_tickers]
            if not sector_tickers:
                continue

            sentiments = []
            for ticker in sector_tickers[:5]:  # Берем до 5 тикеров на сектор
                try:
                    sentiment = self.news_core.get_current_sentiment(ticker)
                    sentiments.append(sentiment)
                except:
                    continue

            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0

            # Определение тональности
            if avg_sentiment > 0.3:
                tone = "positive"
            elif avg_sentiment < -0.3:
                tone = "negative"
            else:
                tone = "neutral"

            sector_data.append({
                'sector': sector,
                'avg_sentiment': avg_sentiment,
                'tone': tone,
                'key_tickers': sector_tickers[:3]
            })

        return sector_data

# ================== ИНИЦИАЛИЗАЦИЯ И ЗАПУСК DASH-ПРИЛОЖЕНИЯ ==================

monitor = RussianMarketMonitor()
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Монитор настроения российского рынка"

# Макет приложения
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("📈 Анализ настроения российского рынка акций",
                       className="text-center mb-4"), width=12)
    ]),

    # Статусная панель
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Обзор рынка"),
            dbc.CardBody([
                html.P(id="market-status"),
                html.P(id="news-count"),
                html.P(id="tracking-info")
            ])
        ]), width=4),
        dbc.Col(dbc.Card([
            dbc.CardHeader("Рыночный сентимент (общий)"),
            dbc.CardBody(dcc.Graph(id="market-sentiment-chart"))
        ]), width=8)
    ], className="mb-4"),

    # Основные графики
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Топ тикеров по упоминаниям в новостях"),
            dbc.CardBody(dcc.Graph(id="ticker-mentions-chart"))
        ]), width=6),
        dbc.Col(dbc.Card([
            dbc.CardHeader("Настроение по секторам"),
            dbc.CardBody(dcc.Graph(id="sector-sentiment-chart"))
        ]), width=6)
    ], className="mb-4"),

    # История сентимента
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Динамика рыночного настроения"),
            dbc.CardBody(dcc.Graph(id="sentiment-history-chart"))
        ]), width=12)
    ], className="mb-4"),

    # Последние новости
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Последние новости"),
            dbc.CardBody(id="recent-news-list")
        ]), width=12)
    ]),

    # Скрытый элемент для хранения данных и интервал обновления
    dcc.Store(id='dashboard-data'),
    dcc.Interval(id='update-interval', interval=60000)  # Обновление каждую минуту
], fluid=True)

# ================== CALLBACKS ДЛЯ ОБНОВЛЕНИЯ ДАННЫХ ==================

@app.callback(
    Output('dashboard-data', 'data'),
    Input('update-interval', 'n_intervals')
)
def update_data(n):
    """Основной callback для обновления данных."""
    return monitor.get_dashboard_data()

@app.callback(
    [Output('market-status', 'children'),
     Output('news-count', 'children'),
     Output('tracking-info', 'children')],
    Input('dashboard-data', 'data')
)
def update_overview(data):
    """Обновление текстового обзора."""
    if not data:
        return "Загрузка...", "Загрузка...", "Загрузка..."

    overview = data['market_overview']
    market_sentiment = overview['market_sentiment']

    status_text = f"Настроение: {market_sentiment.get('trend', 'Н/Д')} " \
                  f"({market_sentiment.get('sentiment', 0):.2f})"
    news_text = f"Новостей обработано: {overview['total_news']}"
    tracking_text = f"Тикеров отслеживается: {overview['tracked_tickers']} " \
                    f"(в портфеле: {overview['portfolio_tickers']})"

    return status_text, news_text, tracking_text

@app.callback(
    Output('market-sentiment-chart', 'figure'),
    Input('dashboard-data', 'data')
)
def update_market_sentiment_chart(data):
    """Индикатор общего рыночного сентимента."""
    if not data:
        return go.Figure()

    sentiment = data['market_overview']['market_sentiment'].get('sentiment', 0)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sentiment * 100,
        title={'text': "Общий рыночный сентимент"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [-100, 100]},
            'bar': {'color': "green" if sentiment > 0 else "red"},
            'steps': [
                {'range': [-100, -50], 'color': "lightcoral"},
                {'range': [-50, 0], 'color': "lightpink"},
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 100], 'color': "limegreen"}
            ]
        }
    ))

    fig.update_layout(height=300)
    return fig

@app.callback(
    Output('ticker-mentions-chart', 'figure'),
    Input('dashboard-data', 'data')
)
def update_ticker_mentions_chart(data):
    """График топ тикеров по упоминаниям."""
    if not data or not data['top_tickers']:
        return go.Figure()

    tickers = [item['ticker'] for item in data['top_tickers']]
    mentions = [item['mentions'] for item in data['top_tickers']]
    sentiments = [item['sentiment'] for item in data['top_tickers']]
    in_portfolio = [item['in_portfolio'] for item in data['top_tickers']]

    colors = ['#2E86AB' if in_port else '#A9A9A9' for in_port in in_portfolio]

    fig = go.Figure(data=[
        go.Bar(
            x=tickers,
            y=mentions,
            marker_color=colors,
            text=[f"{s:.2f}" for s in sentiments],
            textposition='auto',
            hovertemplate=(
                "<b>%{x}</b><br>" +
                "Упоминаний: %{y}<br>" +
                "Сентимент: %{text}<br>" +
                "В портфеле: %{customdata}<extra></extra>"
            ),
            customdata=["Да" if ip else "Нет" for ip in in_portfolio]
        )
    ])

    fig.update_layout(
        title="Топ тикеров по упоминаниям в новостях",
        yaxis_title="Количество упоминаний",
        xaxis_title="Тикер",
        height=400
    )

    return fig

@app.callback(
    Output('sector-sentiment-chart', 'figure'),
    Input('dashboard-data', 'data')
)
def update_sector_sentiment_chart(data):
    """График сентимента по секторам."""
    if not data or not data['sector_overview']:
        return go.Figure()

    sectors = [item['sector'] for item in data['sector_overview']]
    sentiments = [item['avg_sentiment'] for item in data['sector_overview']]

    # Маппинг русских названий секторов
    sector_names = {
        'finance': 'Финансы',
        'oil_gas': 'Нефть и газ',
        'metals': 'Металлургия',
        'electricity': 'Электроэнергетика',
        'retail': 'Ритейл',
        'it_telecom': 'IT и телеком'
    }

    display_sectors = [sector_names.get(s, s) for s in sectors]

    fig = go.Figure(data=[
        go.Bar(
            x=display_sectors,
            y=sentiments,
            marker_color=['green' if s > 0 else 'red' if s < 0 else 'gray' for s in sentiments],
            text=[f"{s:.2f}" for s in sentiments],
            textposition='auto'
        )
    ])

    fig.update_layout(
        title="Средний сентимент по секторам",
        yaxis_title="Сентимент",
        xaxis_title="Сектор",
        yaxis_range=[-1, 1],
        height=400
    )

    return fig

@app.callback(
    Output('sentiment-history-chart', 'figure'),
    Input('dashboard-data', 'data')
)
def update_sentiment_history_chart(data):
    """График истории рыночного сентимента."""
    if not data or not data['sentiment_history']:
        return go.Figure()

    history = data['sentiment_history']
    timestamps = [datetime.fromisoformat(h['timestamp']).strftime('%H:%M') for h in history]
    sentiments = [h['sentiment'] for h in history]

    fig = go.Figure(data=[
        go.Scatter(
            x=timestamps,
            y=sentiments,
            mode='lines+markers',
            line=dict(color='#2E86AB', width=2),
            marker=dict(size=6),
            fill='tozeroy',
            fillcolor='rgba(46, 134, 171, 0.2)'
        )
    ])

    fig.update_layout(
        title="Динамика рыночного настроения",
        yaxis_title="Сентимент",
        xaxis_title="Время",
        yaxis_range=[-1, 1],
        height=350
    )

    return fig

@app.callback(
    Output('recent-news-list', 'children'),
    Input('dashboard-data', 'data')
)
def update_recent_news(data):
    """Список последних новостей."""
    if not data or not data['recent_news']:
        return html.P("Нет новых новостей.")

    news_items = []
    for news in data['recent_news']:
        title = news.get('title', 'Без заголовка')
        source = news.get('source', 'Неизвестно')
        time_str = news.get('published', '')

        # Поиск упоминаний тикеров
        mentioned_tickers = []
        for ticker in monitor.reference_tickers:
            if ticker.lower() in title.lower():
                mentioned_tickers.append(ticker)

        # Создаем бейджи для тикеров
        ticker_badges = []
        if mentioned_tickers:
            for ticker in mentioned_tickers[:3]:  # Показываем до 3 тикеров
                badge_color = "success" if ticker in monitor.portfolio.positions else "secondary"
                ticker_badges.append(
                    dbc.Badge(ticker, color=badge_color, className="ms-1")
                )

        # Создаем карточку новости
        news_card = dbc.Card([
            dbc.CardBody([
                html.H6(title, className="card-title"),
                html.P([
                    html.Small(f"{source} • {time_str[:16]}", className="text-muted"),
                    html.Br(),
                    html.Span("Тикеры: ", className="text-muted"),
                    *(ticker_badges if ticker_badges else [html.Span("—", className="text-muted")])
                ])
            ])
        ], className="mb-2")

        news_items.append(news_card)

    return news_items

if __name__ == '__main__':
    logger.info("Запуск монитора настроения российского рынка...")
    logger.info("Откройте http://localhost:8000 в браузере")


    @app.callback(
        Output('page-content', 'children'),
        Input('url', 'pathname')
    )
    def display_page(pathname):
        return app.layout  # Возвращаем ваш текущий макет


    # И добавьте dcc.Location в макет
    # ИСПРАВЛЕННЫЙ МАКЕТ - убраны все ...
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("📈 Анализ настроения российского рынка акций",
                            className="text-center mb-4"), width=12)
        ]),

        # Статусная панель
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader("Обзор рынка"),
                dbc.CardBody([
                    html.P(id="market-status"),
                    html.P(id="news-count"),
                    html.P(id="tracking-info")
                ])
            ]), width=4),
            dbc.Col(dbc.Card([
                dbc.CardHeader("Рыночный сентимент (общий)"),
                dbc.CardBody(dcc.Graph(id="market-sentiment-chart"))
            ]), width=8)
        ], className="mb-4"),

        # Основные графики
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader("Топ тикеров по упоминаниям в новостях"),
                dbc.CardBody(dcc.Graph(id="ticker-mentions-chart"))
            ]), width=6),
            dbc.Col(dbc.Card([
                dbc.CardHeader("Настроение по секторам"),
                dbc.CardBody(dcc.Graph(id="sector-sentiment-chart"))
            ]), width=6)
        ], className="mb-4"),

        # История сентимента
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader("Динамика рыночного настроения"),
                dbc.CardBody(dcc.Graph(id="sentiment-history-chart"))
            ]), width=12)
        ], className="mb-4"),

        # Последние новости
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader("Последние новости"),
                dbc.CardBody(id="recent-news-list")
            ]), width=12)
        ]),

        # Скрытый элемент для хранения данных и интервал обновления
        dcc.Store(id='dashboard-data'),
        dcc.Interval(id='update-interval', interval=60000)
    ], fluid=True)
    app.run(debug=False, port=8000)