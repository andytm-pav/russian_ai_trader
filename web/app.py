"""
Веб-сервер на Dash для торгового интерфейса
"""

import dash
from dash import dcc, html, Input, Output, State, callback, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import json
import threading
import time
from typing import Dict, List, Optional, Any

from utils.logger import setup_logger
from models.smart_broker import SmartPortfolioBroker
from models.trainer import model_trainer_instance

logger = setup_logger("WEB_APP")

# Инициализация Dash приложения
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    suppress_callback_exceptions=True
)

app.title = "AI Trader - Российский рынок"

# Глобальные переменные для состояния
broker_instance = None
update_interval = 5000  # 5 секунд
web_stop_event = threading.Event()


def run_web_server(broker: SmartPortfolioBroker, stop_event: threading.Event):
    """Запуск веб-сервера"""
    global broker_instance, web_stop_event
    broker_instance = broker
    web_stop_event = stop_event

    # Загрузка конфигурации
    try:
        with open('config/settings.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
            port = config.get('web_port', 8050)
    except:
        port = 8050

    logger.info(f"Запуск веб-сервера на порту {port}")

    try:
        # ИСПРАВЛЕНИЕ: используем app.run() вместо app.run_server()
        app.run(
            host='0.0.0.0',
            port=port,
            debug=False
        )
    except Exception as e:
        logger.error(f"Ошибка запуска веб-сервера: {e}")


# Макет приложения
app.layout = dbc.Container([
    # Заголовок
    dbc.Row([
        dbc.Col([
            html.H1("🤖 AI Trader - Российский рынок",
                    className="text-center mb-4",
                    style={'color': '#00ff88'}),
            html.P("Профессиональная система алгоритмической торговли с AI",
                   className="text-center text-muted mb-4")
        ])
    ]),

    # Навигация
    dbc.Row([
        dbc.Col([
            dbc.Nav([
                dbc.NavLink("📊 Дашборд", href="/", active="exact", id="nav-dashboard"),
                dbc.NavLink("💼 Портфель", href="/portfolio", active="exact", id="nav-portfolio"),
                dbc.NavLink("📈 Графики", href="/charts", active="exact", id="nav-charts"),
                dbc.NavLink("⚙️ Настройки", href="/settings", active="exact", id="nav-settings"),
                dbc.NavLink("📋 Логи", href="/logs", active="exact", id="nav-logs")
            ], pills=True, vertical=False, className="mb-4")
        ])
    ]),

    # Контент
    dbc.Row([
        dbc.Col([
            html.Div(id="page-content")
        ])
    ]),

    # Интервал для обновления
    dcc.Interval(
        id='interval-component',
        interval=update_interval,
        n_intervals=0
    ),

    # Скрытые хранилища
    dcc.Store(id='portfolio-store'),
    dcc.Store(id='signals-store'),
    dcc.Store(id='session-store')
], fluid=True, className="p-4")

# Страница дашборда
dashboard_layout = dbc.Container([
    # Статус системы
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🟢 Статус системы", className="bg-success text-white"),
                dbc.CardBody([
                    html.H4("АКТИВНА", id="system-status", className="text-success"),
                    html.P("Торговая система работает", className="text-muted"),
                    html.Div([
                        dbc.Badge("Рынок открыт", color="success", id="market-status"),
                        dbc.Badge("AI онлайн", color="info", className="ms-2"),
                        dbc.Badge("Новости активны", color="warning", className="ms-2")
                    ], className="mt-2")
                ])
            ])
        ], width=4),

        # Капитал
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("💰 Капитал", className="bg-primary text-white"),
                dbc.CardBody([
                    html.H2("10,000 ₽", id="total-capital"),
                    html.P("Общая стоимость", className="text-muted"),
                    html.Div([
                        html.Span("Кэш: ", className="text-muted"),
                        html.Span("10,000 ₽", id="cash-amount", className="fw-bold"),
                        html.Span(" | Позиции: ", className="text-muted ms-2"),
                        html.Span("0 ₽", id="positions-value", className="fw-bold")
                    ])
                ])
            ])
        ], width=4),

        # PnL
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📈 PnL", className="bg-info text-white"),
                dbc.CardBody([
                    html.H2("+0.00%", id="pnl-percent", className="text-success"),
                    html.P("Доходность", className="text-muted"),
                    html.Div([
                        html.Span("Сегодня: ", className="text-muted"),
                        html.Span("+0 ₽", id="daily-pnl", className="fw-bold"),
                        html.Span(" | Всего: ", className="text-muted ms-2"),
                        html.Span("+0 ₽", id="total-pnl", className="fw-bold")
                    ])
                ])
            ])
        ], width=4)
    ], className="mb-4"),

    # Позиции и сигналы
    dbc.Row([
        # Активные позиции
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📊 Активные позиции"),
                dbc.CardBody([
                    html.Div(id="positions-table",
                             className="table-responsive",
                             style={'maxHeight': '300px', 'overflowY': 'auto'})
                ])
            ])
        ], width=6),

        # Торговые сигналы
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🚦 Последние сигналы"),
                dbc.CardBody([
                    html.Div(id="signals-list",
                             className="table-responsive",
                             style={'maxHeight': '300px', 'overflowY': 'auto'})
                ])
            ])
        ], width=6)
    ], className="mb-4"),

    # Контроль
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🎮 Управление системой"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("▶️ Старт торговли",
                                       id="start-trading-btn",
                                       color="success",
                                       className="w-100 mb-2",
                                       disabled=False),
                            dbc.Button("⏸️ Пауза",
                                       id="pause-trading-btn",
                                       color="warning",
                                       className="w-100 mb-2",
                                       disabled=True),
                            dbc.Button("⏹️ Стоп",
                                       id="stop-trading-btn",
                                       color="danger",
                                       className="w-100",
                                       disabled=False)
                        ], width=4),

                        dbc.Col([
                            dbc.Button("📊 Обновить данные",
                                       id="refresh-data-btn",
                                       color="info",
                                       className="w-100 mb-2"),
                            dbc.Button("💾 Сохранить состояние",
                                       id="save-state-btn",
                                       color="primary",
                                       className="w-100 mb-2"),
                            dbc.Button("🔄 Ребалансировка",
                                       id="rebalance-btn",
                                       color="secondary",
                                       className="w-100")
                        ], width=4),

                        dbc.Col([
                            html.Div([
                                html.Label("Скорость обновления:", className="form-label"),
                                dcc.Slider(
                                    id="update-speed-slider",
                                    min=1,
                                    max=10,
                                    step=1,
                                    value=5,
                                    marks={i: f"{i}s" for i in range(1, 11)}
                                )
                            ]),
                            html.Div([
                                html.Label("Уровень риска:", className="form-label mt-3"),
                                dcc.Slider(
                                    id="risk-level-slider",
                                    min=1,
                                    max=10,
                                    step=1,
                                    value=5,
                                    marks={1: 'Консерв.', 5: 'Умерен.', 10: 'Агресс.'}
                                )
                            ])
                        ], width=4)
                    ])
                ])
            ])
        ])
    ])
])

# Страница портфеля
portfolio_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("💼 Детали портфеля", className="mb-4")
        ])
    ]),

    # Детали позиций
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📋 Все позиции"),
                dbc.CardBody([
                    html.Div(id="detailed-positions-table")
                ])
            ])
        ])
    ], className="mb-4"),

    # Распределение
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📊 Распределение по секторам"),
                dbc.CardBody([
                    dcc.Graph(id="sector-pie-chart")
                ])
            ])
        ], width=6),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📈 Динамика стоимости"),
                dbc.CardBody([
                    dcc.Graph(id="portfolio-value-chart")
                ])
            ])
        ], width=6)
    ], className="mb-4"),

    # История сделок
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📝 История сделок"),
                dbc.CardBody([
                    html.Div(id="trade-history-table")
                ])
            ])
        ])
    ])
])

# Страница графиков
charts_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("📈 Аналитические графики", className="mb-4")
        ])
    ]),

    # Выбор тикера
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🔍 Выбор инструмента"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Тикер:", className="form-label"),
                            dcc.Dropdown(
                                id="ticker-selector",
                                options=[
                                    {'label': 'Сбербанк (SBER)', 'value': 'SBER'},
                                    {'label': 'Газпром (GAZP)', 'value': 'GAZP'},
                                    {'label': 'Лукойл (LKOH)', 'value': 'LKOH'},
                                    {'label': 'Яндекс (YNDX)', 'value': 'YNDX'},
                                    {'label': 'ВТБ (VTBR)', 'value': 'VTBR'},
                                    {'label': 'Роснефть (ROSN)', 'value': 'ROSN'},
                                    {'label': 'Норникель (GMKN)', 'value': 'GMKN'},
                                    {'label': 'Новатэк (NVTK)', 'value': 'NVTK'}
                                ],
                                value='SBER',
                                clearable=False
                            )
                        ], width=4),

                        dbc.Col([
                            html.Label("Период:", className="form-label"),
                            dcc.Dropdown(
                                id="period-selector",
                                options=[
                                    {'label': '1 день', 'value': '1d'},
                                    {'label': '1 неделя', 'value': '1w'},
                                    {'label': '1 месяц', 'value': '1m'},
                                    {'label': '3 месяца', 'value': '3m'},
                                    {'label': '6 месяцев', 'value': '6m'}
                                ],
                                value='1m',
                                clearable=False
                            )
                        ], width=4),

                        dbc.Col([
                            html.Label("Интервал:", className="form-label"),
                            dcc.Dropdown(
                                id="interval-selector",
                                options=[
                                    {'label': '1 минута', 'value': '1min'},
                                    {'label': '5 минут', 'value': '5min'},
                                    {'label': '15 минут', 'value': '15min'},
                                    {'label': '1 час', 'value': '1h'},
                                    {'label': '1 день', 'value': '1d'}
                                ],
                                value='1h',
                                clearable=False
                            )
                        ], width=4)
                    ])
                ])
            ])
        ])
    ], className="mb-4"),

    # Графики
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📊 Цена и объем"),
                dbc.CardBody([
                    dcc.Graph(id="price-volume-chart")
                ])
            ])
        ])
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📉 Технические индикаторы"),
                dbc.CardBody([
                    dcc.Graph(id="indicators-chart")
                ])
            ])
        ], width=6),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📰 Новостной сентимент"),
                dbc.CardBody([
                    dcc.Graph(id="sentiment-chart")
                ])
            ])
        ], width=6)
    ])
])

# Страница настроек
settings_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("⚙️ Настройки системы", className="mb-4")
        ])
    ]),

    # Торговые настройки
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📊 Торговые параметры"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Начальный капитал (₽):", className="form-label"),
                            dbc.Input(
                                id="initial-capital-input",
                                type="number",
                                value=10000,
                                min=1000,
                                step=1000
                            )
                        ], width=4),

                        dbc.Col([
                            html.Label("Макс. позиций:", className="form-label"),
                            dbc.Input(
                                id="max-positions-input",
                                type="number",
                                value=5,
                                min=1,
                                max=20
                            )
                        ], width=4),

                        dbc.Col([
                            html.Label("Макс. вес позиции (%):", className="form-label"),
                            dbc.Input(
                                id="max-weight-input",
                                type="number",
                                value=20,
                                min=5,
                                max=100,
                                step=5
                            )
                        ], width=4)
                    ], className="mb-3"),

                    dbc.Row([
                        dbc.Col([
                            html.Label("Стоп-лосс (%):", className="form-label"),
                            dbc.Input(
                                id="stop-loss-input",
                                type="number",
                                value=3.0,
                                min=0.5,
                                max=10,
                                step=0.5
                            )
                        ], width=4),

                        dbc.Col([
                            html.Label("Тейк-профит (%):", className="form-label"),
                            dbc.Input(
                                id="take-profit-input",
                                type="number",
                                value=6.0,
                                min=1,
                                max=20,
                                step=0.5
                            )
                        ], width=4),

                        dbc.Col([
                            html.Label("Рик на сделку (%):", className="form-label"),
                            dbc.Input(
                                id="risk-per-trade-input",
                                type="number",
                                value=1.5,
                                min=0.5,
                                max=5,
                                step=0.1
                            )
                        ], width=4)
                    ], className="mb-3"),

                    dbc.Button("💾 Сохранить настройки",
                               id="save-settings-btn",
                               color="primary",
                               className="w-100")
                ])
            ])
        ])
    ], className="mb-4"),

    # Настройки торговых сессий
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("⏰ Часы торговли"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Основная сессия:", className="form-label"),
                            dbc.InputGroup([
                                dbc.Input(id="session-start-input",
                                          type="time",
                                          value="06:50"),
                                dbc.InputGroupText("-"),
                                dbc.Input(id="session-end-input",
                                          type="time",
                                          value="18:50")
                            ])
                        ], width=6),

                        dbc.Col([
                            html.Label("Вечерняя сессия:", className="form-label"),
                            dbc.InputGroup([
                                dbc.Input(id="evening-start-input",
                                          type="time",
                                          value="19:00"),
                                dbc.InputGroupText("-"),
                                dbc.Input(id="evening-end-input",
                                          type="time",
                                          value="23:50")
                            ])
                        ], width=6)
                    ], className="mb-3"),

                    dbc.Row([
                        dbc.Col([
                            dbc.Checklist(
                                options=[
                                    {"label": "Торговля в выходные", "value": "weekend"}
                                ],
                                value=[],
                                id="weekend-trading-check",
                                switch=True
                            )
                        ], width=6),

                        dbc.Col([
                            dbc.Checklist(
                                options=[
                                    {"label": "Принудительный режим 24/7", "value": "force"}
                                ],
                                value=[],
                                id="force-trading-check",
                                switch=True
                            )
                        ], width=6)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div(id="session-save-status", className="mt-2 text-center"),
                            dbc.Button("💾 Сохранить часы торговли",
                                       id="session-save-btn",
                                       color="primary",
                                       className="w-100 mt-3")
                        ])
                    ])
                ])
            ])
        ])
    ], className="mb-4"),


    # RSS источники
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📰 RSS источники новостей"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Input(id="new-rss-url", placeholder="Введите URL RSS-ленты...", type="url",
                                      className="mb-2"),
                            dbc.Input(id="new-rss-name", placeholder="Название источника (необязательно)",
                                      className="mb-2"),
                            dbc.Button("➕ Добавить источник", id="add-rss-btn", color="success", className="w-100"),
                            html.Div(id="rss-action-status", className="mt-2 text-center"),  # Для статуса
                            html.Hr(),
                            html.H5("Текущие источники"),
                            html.Div(id="rss-sources-list")  # Список будет обновляться
                        ])
                    ]),

                    dbc.Button("➕ Добавить источник",
                               id="add-rss-btn",
                               color="success",
                               className="mt-3")
                ])
            ])
        ])
    ])
])

# Страница логов
logs_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("📋 Системные логи", className="mb-4")
        ])
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    "Журнал событий",
                    dbc.Button("🗑️ Очистить",
                               id="clear-logs-btn",
                               color="danger",
                               size="sm",
                               className="float-end")
                ]),
                dbc.CardBody([
                    html.Div(id="logs-content",
                             style={
                                 'height': '500px',
                                 'overflowY': 'auto',
                                 'backgroundColor': '#1a1a1a',
                                 'padding': '10px',
                                 'fontFamily': 'monospace',
                                 'fontSize': '12px'
                             })
                ])
            ])
        ])
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Статистика системы"),
                dbc.CardBody([
                    html.Div(id="system-stats")
                ])
            ])
        ], width=6),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Статус компонентов"),
                dbc.CardBody([
                    html.Div(id="components-status")
                ])
            ])
        ], width=6)
    ], className="mt-4")
])


# Коллбэки для навигации
@app.callback(
    Output("page-content", "children"),
    [Input("nav-dashboard", "n_clicks"),
     Input("nav-portfolio", "n_clicks"),
     Input("nav-charts", "n_clicks"),
     Input("nav-settings", "n_clicks"),
     Input("nav-logs", "n_clicks")],
    prevent_initial_call=True
)
def update_page_content(dash_clicks, port_clicks, charts_clicks, settings_clicks, logs_clicks):
    """Обновление контента страницы"""
    ctx_triggered = ctx.triggered_id

    if ctx_triggered == "nav-dashboard":
        return dashboard_layout
    elif ctx_triggered == "nav-portfolio":
        return portfolio_layout
    elif ctx_triggered == "nav-charts":
        return charts_layout
    elif ctx_triggered == "nav-settings":
        return settings_layout
    elif ctx_triggered == "nav-logs":
        return logs_layout

    return dashboard_layout


# Коллбэк для обновления дашборда
@app.callback(
    [Output("system-status", "children"),
     Output("market-status", "children"),
     Output("total-capital", "children"),
     Output("cash-amount", "children"),
     Output("positions-value", "children"),
     Output("pnl-percent", "children"),
     Output("daily-pnl", "children"),
     Output("total-pnl", "children"),
     Output("positions-table", "children"),
     Output("signals-list", "children")],
    [Input("interval-component", "n_intervals"),
     Input("refresh-data-btn", "n_clicks")]
)
def update_dashboard(n_intervals, refresh_clicks):
    """Обновление данных на дашборде"""
    if broker_instance is None:
        return ["ОФФЛАЙН", "Рынок закрыт", "0 ₽", "0 ₽", "0 ₽", "0%", "0 ₽",
                "0 ₽", "Нет данных", "Нет данных"]

    try:
        # Получение данных от брокера
        summary = broker_instance.get_portfolio_summary()
        session_info = summary.get('session_info', {})

        # Статус системы
        system_status = "АКТИВНА" if broker_instance.trading_enabled else "ПАУЗА"

        # Статус рынка
        market_status = "Рынок открыт" if session_info.get('is_trading_time') else "Рынок закрыт"

        # Капитал
        total_value = summary.get('total_value', 0)
        cash = summary.get('cash', 0)
        positions_value = total_value - cash

        total_capital = f"{total_value:,.0f} ₽"
        cash_amount = f"{cash:,.0f} ₽"
        positions_val = f"{positions_value:,.0f} ₽"

        # PnL
        initial = summary.get('initial_capital', total_value)
        pnl_percent = ((total_value / initial) - 1) * 100 if initial > 0 else 0
        pnl_abs = total_value - initial

        pnl_percent_text = f"{pnl_percent:+.2f}%"
        pnl_percent_class = "text-success" if pnl_percent >= 0 else "text-danger"

        daily_pnl = summary.get('risk_metrics', {}).get('daily_pnl', 0)
        daily_pnl_text = f"{daily_pnl:+,.0f} ₽"

        total_pnl_text = f"{pnl_abs:+,.0f} ₽"

        # Позиции
        positions_table = create_positions_table(summary.get('positions', []))

        # Сигналы
        signals_list = create_signals_list(summary.get('current_signals', []))

        return [
            system_status,
            market_status,
            total_capital,
            cash_amount,
            positions_val,
            html.Span(pnl_percent_text, className=pnl_percent_class),
            html.Span(daily_pnl_text, className="text-success" if daily_pnl >= 0 else "text-danger"),
            html.Span(total_pnl_text, className="text-success" if pnl_abs >= 0 else "text-danger"),
            positions_table,
            signals_list
        ]

    except Exception as e:
        logger.error(f"Ошибка обновления дашборда: {e}")
        return ["ОШИБКА", "Ошибка", "0 ₽", "0 ₽", "0 ₽", "0%", "0 ₽",
                "0 ₽", "Ошибка загрузки", "Ошибка загрузки"]


def create_positions_table(positions: List[Dict]) -> html.Table:
    """Создание таблицы позиций"""
    if not positions:
        return html.P("Нет открытых позиций", className="text-muted")

    header = html.Thead(html.Tr([
        html.Th("Тикер"),
        html.Th("Кол-во"),
        html.Th("Ср. цена"),
        html.Th("Тек. цена"),
        html.Th("Стоимость"),
        html.Th("PnL"),
        html.Th("Вес")
    ]))

    rows = []
    for pos in positions:
        pnl_class = "text-success" if pos.get('pnl', 0) >= 0 else "text-danger"

        rows.append(html.Tr([
            html.Td(pos['ticker']),
            html.Td(f"{pos['quantity']:,}"),
            html.Td(f"{pos['avg_price']:.2f}"),
            html.Td(f"{pos.get('current_price', 0):.2f}"),
            html.Td(f"{pos.get('position_value', 0):,.0f} ₽"),
            html.Td(html.Span(f"{pos.get('pnl_percent', 0):+.1f}%", className=pnl_class)),
            html.Td(f"{pos.get('weight', 0):.1f}%")
        ]))

    table = html.Table([header, html.Tbody(rows)],
                       className="table table-sm table-hover")

    return table


def create_signals_list(signals: List[Dict]) -> html.Table:
    """Создание списка сигналов"""
    if not signals:
        return html.P("Нет активных сигналов", className="text-muted")

    header = html.Thead(html.Tr([
        html.Th("Тикер"),
        html.Th("Сигнал"),
        html.Th("Уверенность"),
        html.Th("Причина"),
        html.Th("Время")
    ]))

    rows = []
    for sig in signals[:10]:  # Ограничиваем 10 сигналами
        action = sig.get('action', 'HOLD')
        confidence = sig.get('confidence', 0)

        # Цвет в зависимости от действия
        if action == 'BUY':
            action_class = "text-success"
        elif action == 'SELL':
            action_class = "text-danger"
        else:
            action_class = "text-warning"

        # Цвет уверенности
        if confidence > 0.8:
            conf_class = "text-success"
        elif confidence > 0.6:
            conf_class = "text-warning"
        else:
            conf_class = "text-muted"

        # Время
        timestamp = sig.get('timestamp', '')
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                time_str = dt.strftime('%H:%M:%S')
            except:
                time_str = timestamp[:8]
        else:
            time_str = ''

        rows.append(html.Tr([
            html.Td(sig['ticker']),
            html.Td(html.Span(action, className=action_class)),
            html.Td(html.Span(f"{confidence:.1%}", className=conf_class)),
            html.Td(sig.get('reason', 'N/A')),
            html.Td(time_str)
        ]))

    table = html.Table([header, html.Tbody(rows)],
                       className="table table-sm table-hover")

    return table


# Коллбэки для управления
@app.callback(
    [Output("start-trading-btn", "disabled"),
     Output("pause-trading-btn", "disabled"),
     Output("stop-trading-btn", "disabled")],
    [Input("start-trading-btn", "n_clicks"),
     Input("pause-trading-btn", "n_clicks"),
     Input("stop-trading-btn", "n_clicks")]
)
def control_trading(start_clicks, pause_clicks, stop_clicks):
    """Управление торговлей"""
    if broker_instance is None:
        return [True, True, True]

    ctx_triggered = ctx.triggered_id

    if ctx_triggered == "start-trading-btn":
        broker_instance.trading_enabled = True
        logger.info("Торговля запущена")
        return [True, False, False]

    elif ctx_triggered == "pause-trading-btn":
        broker_instance.trading_enabled = False
        logger.info("Торговля приостановлена")
        return [False, True, False]

    elif ctx_triggered == "stop-trading-btn":
        broker_instance.trading_enabled = False
        logger.info("Торговля остановлена")
        # Здесь можно добавить логику полной остановки
        return [False, True, True]

    # По умолчанию
    return [not broker_instance.trading_enabled,
            broker_instance.trading_enabled,
            False]


# Коллбэк для сохранения состояния
@app.callback(
    Output("save-state-btn", "children"),
    Input("save-state-btn", "n_clicks")
)
def save_system_state(n_clicks):
    """Сохранение состояния системы"""
    if n_clicks and broker_instance is not None:
        try:
            broker_instance._save_portfolio_state()
            broker_instance.model.save_model()
            logger.info("Состояние системы сохранено")
            return "✅ Сохранено!"
        except Exception as e:
            logger.error(f"Ошибка сохранения: {e}")
            return "❌ Ошибка!"

    return "💾 Сохранить состояние"


# Коллбэк для ребалансировки
@app.callback(
    Output("rebalance-btn", "children"),
    Input("rebalance-btn", "n_clicks")
)
def rebalance_portfolio(n_clicks):
    """Ребалансировка портфеля"""
    if n_clicks and broker_instance is not None:
        try:
            # Получаем текущие цены
            prices = broker_instance._get_current_prices()
            if prices:
                securities = broker_instance.moex.get_all_securities()
                broker_instance._rebalance_portfolio(prices, securities)
                logger.info("Ребалансировка выполнена")
                return "✅ Ребалансировано!"
        except Exception as e:
            logger.error(f"Ошибка ребалансировки: {e}")
            return "❌ Ошибка!"

    return "🔄 Ребалансировка"


# Коллбэк для обновления скорости
@app.callback(
    Output("interval-component", "interval"),
    Input("update-speed-slider", "value")
)
def update_refresh_speed(value):
    """Обновление скорости обновления"""
    return value * 1000  # Конвертируем в миллисекунды


# Коллбэк для сохранения торговых настроек
@app.callback(
    Output("save-settings-btn", "children"),
    [Input("save-settings-btn", "n_clicks")],
    [State("initial-capital-input", "value"),
     State("max-positions-input", "value"),
     State("max-weight-input", "value"),
     State("stop-loss-input", "value"),
     State("take-profit-input", "value"),
     State("risk-per-trade-input", "value")]
)
def save_trading_settings(n_clicks, initial_capital, max_positions, max_weight,
                          stop_loss, take_profit, risk_per_trade):
    """Сохранение торговых настроек"""
    if n_clicks is None or n_clicks == 0:
        return "💾 Сохранить настройки"

    try:
        # Загружаем текущие настройки
        with open('config/settings.json', 'r', encoding='utf-8') as f:
            settings = json.load(f)

        # Обновляем настройки
        settings['initial_capital_rub'] = initial_capital
        settings['max_positions'] = max_positions
        settings['max_position_weight_percent'] = max_weight
        settings['stop_loss_percent'] = stop_loss
        settings['take_profit_percent'] = take_profit
        settings['risk_per_trade_percent'] = risk_per_trade

        # Сохраняем в файл
        with open('config/settings.json', 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=4, ensure_ascii=False)

        # Обновляем настройки в брокере
        if broker_instance is not None:
            broker_instance.settings = settings
            broker_instance.portfolio.initial_capital = initial_capital
            broker_instance.portfolio.max_positions = max_positions

        logger.info(f"Настройки сохранены: капитал={initial_capital}, макс.позиций={max_positions}")
        return "✅ Настройки сохранены!"

    except Exception as e:
        logger.error(f"Ошибка сохранения настроек: {e}")
        return "❌ Ошибка сохранения"


@app.callback(
    Output("session-save-status", "children"),  # Нужно добавить этот элемент в layout
    [Input("session-save-btn", "n_clicks")],  # Нужно создать кнопку с этим ID
    [State("session-start-input", "value"),
     State("session-end-input", "value"),
     State("evening-start-input", "value"),
     State("evening-end-input", "value"),
     State("weekend-trading-check", "value"),
     State("force-trading-check", "value")]
)
def save_session_settings(n_clicks, main_start, main_end,
                          evening_start, evening_end, weekend_trading, force_trading):
    """Сохранение настроек торговых сессий"""
    if not n_clicks:
        return ""

    try:
        # 1. Загружаем настройки
        with open('config/settings.json', 'r', encoding='utf-8') as f:
            settings = json.load(f)

        # 2. Обновляем раздел trading_hours
        if 'trading_hours' not in settings:
            settings['trading_hours'] = {}

        settings['trading_hours']['main_session'] = f"{main_start}-{main_end}"
        settings['trading_hours']['evening_session'] = f"{evening_start}-{evening_end}"
        settings['trading_hours']['trade_on_weekend'] = 'weekend' in weekend_trading
        settings['trading_hours']['force_247'] = 'force' in force_trading

        # 3. Сохраняем файл
        with open('config/settings.json', 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)

        # 4. Обновляем планировщик (если брокер инициализирован)
        if broker_instance and hasattr(broker_instance, 'scheduler'):
            broker_instance.scheduler.trading_hours = settings['trading_hours']

        logger.info(f"Часы торговли сохранены: {settings['trading_hours']['main_session']}")
        return "✅ Часы торговли сохранены!"

    except Exception as e:
        logger.error(f"Ошибка сохранения часов: {e}")
        return "❌ Ошибка"

@app.callback(
    [Output("rss-action-status", "children"),
     Output("rss-sources-list", "children"),
     Output("new-rss-url", "value"),
     Output("new-rss-name", "value")],
    [Input("add-rss-btn", "n_clicks")],
    [State("new-rss-url", "value"),
     State("new-rss-name", "value")]
)
def add_rss_source(n_clicks, new_url, new_name):
    if n_clicks is None or not new_url:
        return "", "Загружаем список...", "", ""

    config_path = "config/rss_sources.json"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except:
        config = {'sources': []}

    # Проверяем, нет ли такого URL уже
    if any(source.get('url') == new_url for source in config['sources']):
        return f"❌ Источник {new_url} уже добавлен.", "", new_url, new_name

    # Добавляем новый источник
    new_source = {
        'url': new_url,
        'name': new_name or f"Источник {len(config['sources'])+1}",
        'enabled': True,
        'update_interval': 300
    }
    config['sources'].append(new_source)

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    # Формируем список для отображения
    sources_list = html.Ul([
        html.Li([
            html.Span(f"📰 {src['name']} ", style={'font-weight': 'bold'}),
            html.Span(f"({src['url'][:50]}...)"),
            dbc.Button("🗑️", id={'type': 'del-rss-btn', 'index': i}, color="danger", size="sm", className="ms-2")
        ]) for i, src in enumerate(config['sources'])
    ])

    return f"✅ Добавлен: {new_source['name']}", sources_list, "", ""
# Запуск приложения
if __name__ == '__main__':
    # Тестовый запуск
    print("Запуск веб-интерфейса...")
    # ИСПРАВЛЕНИЕ:
    app.run(debug=True, port=8050, host='127.0.0.1')