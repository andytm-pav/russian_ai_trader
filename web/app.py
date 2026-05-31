"""
Веб-сервер на Dash для торгового интерфейса
"""

import dash
from dash import dcc, html, Input, Output, State, callback, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import threading
import os
from typing import Dict, List, Optional, Any

from web.web_dashboard import dashboard_viz

from utils.logger import get_logger
from models.smart_broker import SmartPortfolioBroker
from models.trainer import model_trainer_instance

logger = get_logger("WEB_APP")


def get_portfolio_tickers():
    """Получаем тикеры из текущего портфеля"""
    try:
        portfolio_path = 'data/portfolio_state.json'
        if os.path.exists(portfolio_path):
            with open(portfolio_path, 'r', encoding='utf-8') as f:
                portfolio = json.load(f)

            positions = portfolio.get('positions', {})
            portfolio_tickers = list(positions.keys())

            if 'MOEX' not in portfolio_tickers:
                portfolio_tickers.append('MOEX')

            return portfolio_tickers
    except Exception as e:
        logger.error(f"Ошибка загрузки портфеля: {e}")

    return ['SBER', 'GAZP', 'LKOH', 'MOEX']


app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    suppress_callback_exceptions=True
)

app.title = "AI Trader - Российский рынок"

broker_instance = None
update_interval = 5000
web_stop_event = threading.Event()


def run_web_server(broker: SmartPortfolioBroker, stop_event: threading.Event):
    """Запуск веб-сервера"""
    global broker_instance, web_stop_event
    broker_instance = broker
    web_stop_event = stop_event

    try:
        with open('config/settings.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
            port = config.get('web_port', 8050)
    except:
        port = 8050

    logger.info(f"Запуск веб-сервера на порту {port}")

    try:
        app.run(
            host='0.0.0.0',
            port=port,
            debug=False
        )
    except Exception as e:
        logger.error(f"Ошибка запуска веб-сервера: {e}")


app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("🤖 AI Trader - Российский рынок",
                    className="text-center mb-4",
                    style={'color': '#00ff88'}),
            html.P("Профессиональная система алгоритмической торговли с AI",
                   className="text-center text-muted mb-4")
        ])
    ]),

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

    dbc.Row([
        dbc.Col([
            html.Div(id="page-content")
        ])
    ]),

    dcc.Interval(
        id='interval-component',
        interval=update_interval,
        n_intervals=0
    ),

    dcc.Store(id='portfolio-store'),
    dcc.Store(id='signals-store'),
    dcc.Store(id='session-store')
], fluid=True, className="p-4")


dashboard_layout = dbc.Container([
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
                    ]),
                    html.Div([
                        html.Span("💎 Накоплено: ", className="text-muted"),
                        html.Span("0 ₽", id="fixated-profit", className="fw-bold text-info")
                    ], className="mt-2")
                ])
            ])
        ], width=4)
    ], className="mb-4"),

    dbc.Row([
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

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🎮 Управление системой"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("▶️ Старт торговли", id="start-trading-btn", color="success", className="w-100 mb-2", disabled=False),
                            dbc.Button("⏸️ Пауза", id="pause-trading-btn", color="warning", className="w-100 mb-2", disabled=True),
                            dbc.Button("⏹️ Стоп", id="stop-trading-btn", color="danger", className="w-100", disabled=False)
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
                                       className="w-100 mb-2"),

                        ], width=4),
                        dbc.Col([
                            html.Div([
                                html.Label("Скорость обновления:", className="form-label"),
                                dcc.Slider(id="update-speed-slider", min=1, max=10, step=1, value=5,
                                           marks={i: f"{i}s" for i in range(1, 11)})
                            ]),
                            html.Div([
                                html.Label("Уровень риска:", className="form-label mt-3"),
                                dcc.Slider(id="risk-level-slider", min=1, max=10, step=1, value=5,
                                           marks={1: 'Консерв.', 5: 'Умерен.', 10: 'Агресс.'})
                            ])
                        ], width=4)
                    ])
                ])
            ])
        ])
    ])
])


portfolio_layout = dbc.Container([
    dbc.Row([dbc.Col([html.H2("💼 Детали портфеля", className="mb-4")])]),
    dbc.Row([dbc.Col([dbc.Card([dbc.CardHeader("📋 Все позиции"), dbc.CardBody([html.Div(id="detailed-positions-table")])])])], className="mb-4"),
    dbc.Row([
        dbc.Col([dbc.Card([dbc.CardHeader("📊 Распределение по секторам"), dbc.CardBody([dcc.Graph(id="sector-pie-chart")])])], width=6),
        dbc.Col([dbc.Card([dbc.CardHeader("📈 Динамика стоимости"), dbc.CardBody([dcc.Graph(id="portfolio-value-chart")])])], width=6)
    ], className="mb-4"),
    dbc.Row([dbc.Col([dbc.Card([dbc.CardHeader("📝 История сделок"), dbc.CardBody([html.Div(id="trade-history-table")])])])])
])


charts_layout = dbc.Container([
    dbc.Row([dbc.Col([html.H2("📈 Аналитические графики", className="mb-4")])]),
    dbc.Row([dbc.Col([dbc.Card([dbc.CardHeader("🔍 Выбор инструмента"), dbc.CardBody([dbc.Row([
        dbc.Col([html.Label("Тикер:", className="form-label"), dcc.Dropdown(id="ticker-selector", options=[], value='MOEX', clearable=False)], width=4),
        dbc.Col([html.Label("Период:", className="form-label"), dcc.Dropdown(id="period-selector", options=[
            {'label': '1 день', 'value': '1d'}, {'label': '1 неделя', 'value': '1w'}, {'label': '1 месяц', 'value': '1m'},
            {'label': '3 месяца', 'value': '3m'}, {'label': '6 месяцев', 'value': '6m'}], value='1m', clearable=False)], width=4),
        dbc.Col([html.Label("Интервал:", className="form-label"), dcc.Dropdown(id="interval-selector", options=[
            {'label': '1 минута', 'value': '1min'}, {'label': '5 минут', 'value': '5min'}, {'label': '15 минут', 'value': '15min'},
            {'label': '1 час', 'value': '1h'}, {'label': '1 день', 'value': '1d'}], value='1h', clearable=False)], width=4)
    ])])])])], className="mb-4"),
    dbc.Row([dbc.Col([dbc.Card([dbc.CardHeader("📊 Цена и объем"), dbc.CardBody([dcc.Graph(id="price-volume-chart")])])])], className="mb-4"),
    dbc.Row([
        dbc.Col([dbc.Card([dbc.CardHeader("📉 Технические индикаторы"), dbc.CardBody([dcc.Graph(id="indicators-chart")])])], width=6),
        dbc.Col([dbc.Card([dbc.CardHeader("📰 Новостной сентимент"), dbc.CardBody([dcc.Graph(id="sentiment-chart")])])], width=6)
    ])
])
# Часть 2 из 4: settings_layout, logs_layout

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
                            dbc.Input(id="initial-capital-input", type="number", value=10000, min=1000, step=1000)
                        ], width=4),
                        dbc.Col([
                            html.Label("Макс. позиций:", className="form-label"),
                            dbc.Input(id="max-positions-input", type="number", value=5, min=1, max=20)
                        ], width=4),
                        dbc.Col([
                            html.Label("Макс. вес позиции (%):", className="form-label"),
                            dbc.Input(id="max-weight-input", type="number", value=20, min=5, max=100, step=5)
                        ], width=4)
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Стоп-лосс (%):", className="form-label"),
                            dbc.Input(id="stop-loss-input", type="number", value=3.0, min=0.5, max=10, step=0.5)
                        ], width=4),
                        dbc.Col([
                            html.Label("Тейк-профит (%):", className="form-label"),
                            dbc.Input(id="take-profit-input", type="number", value=6.0, min=1, max=20, step=0.5)
                        ], width=4),
                        dbc.Col([
                            html.Label("Риск на сделку (%):", className="form-label"),
                            dbc.Input(id="risk-per-trade-input", type="number", value=1.5, min=0.5, max=5, step=0.1)
                        ], width=4)
                    ], className="mb-3"),
                    dbc.Button("💾 Сохранить настройки", id="save-settings-btn", color="primary", className="w-100"),
                    html.Hr(),
                    dbc.Button("🔧 Обновить конфиги",
                               id="reload-configs-btn",
                               color="warning",
                               className="w-100 mt-2")
                ])
            ])
        ])
    ], className="mb-4"),

    # Защитные лимиты
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🛡️ Защитные лимиты"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Кулдаун (сек):", className="form-label"),
                            dbc.Input(id="cooldown-input", type="number", value=1800, min=60, step=60)
                        ], width=4),
                        dbc.Col([
                            html.Label("Мин. сумма сделки (₽):", className="form-label"),
                            dbc.Input(id="min-cash-input", type="number", value=1000, min=100, step=100)
                        ], width=4),
                        dbc.Col([
                            html.Label("Мин. риск на акцию (%):", className="form-label"),
                            dbc.Input(id="min-risk-input", type="number", value=0.01, min=0.001, max=1.0, step=0.001)
                        ], width=4)
                    ], className="mb-3"),
                ])
            ])
        ])
    ], className="mb-4"),

    # Фиксация прибыли
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("💰 Фиксация дневной прибыли"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Checklist(
                                options=[{"label": "Включить фиксацию", "value": "enabled"}],
                                value=["enabled"],
                                id="fixation-enabled-check",
                                switch=True
                            )
                        ], width=4),
                        dbc.Col([
                            html.Label("Мин. прибыль для фиксации (₽):", className="form-label"),
                            dbc.Input(id="fixation-min-profit-input", type="number", value=50, min=10, step=10)
                        ], width=4),
                        dbc.Col([
                            html.Label("Реинвестировать (%):", className="form-label"),
                            dcc.Slider(
                                id="fixation-reinvest-slider",
                                min=0, max=100, step=10, value=0,
                                marks={0: '0%', 50: '50%', 100: '100%'}
                            )
                        ], width=4)
                    ], className="mb-3"),
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
                                dbc.Input(id="session-start-input", type="time", value="06:50"),
                                dbc.InputGroupText("-"),
                                dbc.Input(id="session-end-input", type="time", value="18:50")
                            ])
                        ], width=6),
                        dbc.Col([
                            html.Label("Вечерняя сессия:", className="form-label"),
                            dbc.InputGroup([
                                dbc.Input(id="evening-start-input", type="time", value="19:00"),
                                dbc.InputGroupText("-"),
                                dbc.Input(id="evening-end-input", type="time", value="23:50")
                            ])
                        ], width=6)
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Checklist(
                                options=[{"label": "Торговля в выходные", "value": "weekend"}],
                                value=[], id="weekend-trading-check", switch=True
                            )
                        ], width=6),
                        dbc.Col([
                            dbc.Checklist(
                                options=[{"label": "Принудительный режим 24/7", "value": "force"}],
                                value=[], id="force-trading-check", switch=True
                            )
                        ], width=6)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div(id="session-save-status", className="mt-2 text-center"),
                            dbc.Button("💾 Сохранить часы торговли", id="session-save-btn", color="primary", className="w-100 mt-3")
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
                            dbc.Input(id="new-rss-url", placeholder="Введите URL RSS-ленты...", type="url", className="mb-2"),
                            dbc.Input(id="new-rss-name", placeholder="Название источника (необязательно)", className="mb-2"),
                            dbc.Button("➕ Добавить источник", id="add-rss-btn", color="success", className="w-100"),
                            html.Div(id="rss-action-status", className="mt-2 text-center"),
                            html.Hr(),
                            html.H5("Текущие источники"),
                            html.Div(id="rss-sources-list")
                        ])
                    ])
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

    # Фильтр логов
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Уровень:", className="form-label"),
                            dcc.Dropdown(
                                id='log-level-filter',
                                options=[
                                    {'label': 'Все', 'value': 'ALL'},
                                    {'label': 'INFO', 'value': 'INFO'},
                                    {'label': 'WARNING', 'value': 'WARNING'},
                                    {'label': 'ERROR', 'value': 'ERROR'},
                                    {'label': 'СДЕЛКИ', 'value': 'TRADES'}
                                ],
                                value='ALL',
                                clearable=False
                            )
                        ], width=4),
                        dbc.Col([
                            html.Label("Компонент:", className="form-label"),
                            dcc.Dropdown(
                                id='log-component-filter',
                                options=[
                                    {'label': 'Все', 'value': 'ALL'},
                                    {'label': 'MAIN', 'value': 'MAIN'},
                                    {'label': 'MOEX_FETCHER', 'value': 'MOEX_FETCHER'},
                                    {'label': 'NEWS_FETCHER_OPT', 'value': 'NEWS_FETCHER_OPT'},
                                    {'label': 'PORTFOLIO', 'value': 'PORTFOLIO'},
                                    {'label': 'RISK_MANAGER', 'value': 'RISK_MANAGER'},
                                    {'label': 'SMART_BROKER', 'value': 'SMART_BROKER'},
                                    {'label': 'TECH_CORE', 'value': 'TECH_CORE'},
                                    {'label': 'TRADER_MODEL', 'value': 'TRADER_MODEL'},
                                    {'label': 'TRAINER', 'value': 'TRAINER'}
                                ],
                                value='ALL',
                                clearable=False
                            )
                        ], width=4),
                    ])
                ])
            ], className="mb-2")
        ])
    ]),

    # Журнал событий
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    "Журнал событий",
                    dbc.Button("🗑️ Очистить", id="clear-logs-btn", color="danger", size="sm", className="float-end")
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
                dbc.CardBody([html.Div(id="system-stats")])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Статус компонентов"),
                dbc.CardBody([html.Div(id="components-status")])
            ])
        ], width=6)
    ], className="mt-4")
])
# Часть 3 из 4: коллбэки навигации, дашборда, управления, настроек, логов

# Коллбэк навигации
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


# Коллбэк обновления дашборда
@app.callback(
    [Output("system-status", "children"),
     Output("market-status", "children"),
     Output("total-capital", "children"),
     Output("cash-amount", "children"),
     Output("positions-value", "children"),
     Output("pnl-percent", "children"),
     Output("daily-pnl", "children"),
     Output("total-pnl", "children"),
     Output("fixated-profit", "children"),
     Output("positions-table", "children"),
     Output("signals-list", "children")],
    [Input("interval-component", "n_intervals"),
     Input("refresh-data-btn", "n_clicks")]
)
def update_dashboard(n_intervals, refresh_clicks):
    """Обновление данных на дашборде"""
    if broker_instance is None:
        return ["ОФФЛАЙН", "Рынок закрыт", "0 ₽", "0 ₽", "0 ₽", "0%", "0 ₽",
                "0 ₽", "0 ₽", "Нет данных", "Нет данных"]

    try:
        summary = broker_instance.get_portfolio_summary()
        session_info = summary.get('session_info', {})

        system_status = "АКТИВНА" if broker_instance.trading_enabled else "ПАУЗА"
        market_status = "Рынок открыт" if session_info.get('is_trading_time', False) else "Рынок закрыт"

        total_value = summary.get('total_value', 0)
        cash = summary.get('cash', 0)
        positions_value = total_value - cash

        total_capital = f"{total_value:,.0f} ₽"
        cash_amount = f"{cash:,.0f} ₽"
        positions_val = f"{positions_value:,.0f} ₽"

        initial = summary.get('initial_capital', total_value)
        pnl_percent = ((total_value / initial) - 1) * 100 if initial > 0 else 0
        pnl_abs = total_value - initial
        pnl_percent_text = f"{pnl_percent:+.2f}%"
        pnl_percent_class = "text-success" if pnl_percent >= 0 else "text-danger"

        daily_pnl = summary.get('risk_metrics', {}).get('daily_pnl', 0)
        daily_pnl_text = f"{daily_pnl:+,.0f} ₽"
        total_pnl_text = f"{pnl_abs:+,.0f} ₽"

        positions_table = create_positions_table(summary.get('positions', []))
        signals_list = create_signals_list(summary.get('current_signals', []))

        if n_intervals and n_intervals % 5 == 0:
            save_portfolio_history()

        reserved_cash = summary.get('reserved_cash', 0)
        fixated_text = f"{reserved_cash:+,.0f} ₽"

        return [
            system_status, market_status, total_capital, cash_amount, positions_val,
            html.Span(pnl_percent_text, className=pnl_percent_class),
            html.Span(daily_pnl_text, className="text-success" if daily_pnl >= 0 else "text-danger"),
            html.Span(total_pnl_text, className="text-success" if pnl_abs >= 0 else "text-danger"),
            html.Span(fixated_text, className="text-info fw-bold"),
            positions_table, signals_list
        ]

    except Exception as e:
        logger.error(f"Ошибка обновления дашборда: {e}")
        return ["ОШИБКА", "Ошибка", "0 ₽", "0 ₽", "0 ₽", "0%", "0 ₽",
                "0 ₽", "0 ₽", "Ошибка загрузки", "Ошибка загрузки"]


def create_positions_table(positions: List[Dict]) -> html.Table:
    """Создание таблицы позиций"""
    if not positions:
        return html.P("Нет открытых позиций", className="text-muted")

    header = html.Thead(html.Tr([
        html.Th("Тикер"), html.Th("Кол-во"), html.Th("Ср. цена"),
        html.Th("Тек. цена"), html.Th("Стоимость"), html.Th("PnL"), html.Th("Вес")
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

    return html.Table([header, html.Tbody(rows)], className="table table-sm table-hover")


def create_signals_list(signals: List[Dict]) -> html.Table:
    """Создание списка сигналов"""
    if not signals:
        return html.P("Нет активных сигналов", className="text-muted")

    header = html.Thead(html.Tr([
        html.Th("Тикер"), html.Th("Сигнал"), html.Th("Уверенность"), html.Th("Причина"), html.Th("Время")
    ]))

    rows = []
    for sig in signals[:10]:
        action = sig.get('action', 'HOLD')
        confidence = min(sig.get('confidence', 0), 1.0)

        if action == 'BUY':
            action_class = "text-success"
        elif action == 'SELL':
            action_class = "text-danger"
        else:
            action_class = "text-warning"

        if confidence > 0.8:
            conf_class = "text-success"
        elif confidence > 0.6:
            conf_class = "text-warning"
        else:
            conf_class = "text-muted"

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

    return html.Table([header, html.Tbody(rows)], className="table table-sm table-hover")


# Коллбэки управления
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
        return [False, True, True]

    return [not broker_instance.trading_enabled, broker_instance.trading_enabled, False]


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


@app.callback(
    Output("rebalance-btn", "children"),
    Input("rebalance-btn", "n_clicks")
)
def rebalance_portfolio(n_clicks):
    """Ребалансировка портфеля"""
    if n_clicks and broker_instance is not None:
        try:
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


@app.callback(
    Output("interval-component", "interval"),
    Input("update-speed-slider", "value")
)
def update_refresh_speed(value):
    """Обновление скорости обновления"""
    return value * 1000


# Сохранение настроек
@app.callback(
    Output("save-settings-btn", "children"),
    [Input("save-settings-btn", "n_clicks")],
    [State("initial-capital-input", "value"),
     State("max-positions-input", "value"),
     State("max-weight-input", "value"),
     State("stop-loss-input", "value"),
     State("take-profit-input", "value"),
     State("risk-per-trade-input", "value"),
     State("cooldown-input", "value"),
     State("min-cash-input", "value"),
     State("min-risk-input", "value"),
     State("fixation-enabled-check", "value"),
     State("fixation-min-profit-input", "value"),
     State("fixation-reinvest-slider", "value")]
)
def save_trading_settings(n_clicks, initial_capital, max_positions, max_weight,
                          stop_loss, take_profit, risk_per_trade,
                          cooldown, min_cash, min_risk,
                          fixation_enabled, fixation_min_profit, fixation_reinvest):
    """Сохранение торговых настроек"""
    if n_clicks is None or n_clicks == 0:
        return "💾 Сохранить настройки"

    try:
        with open('config/settings.json', 'r', encoding='utf-8') as f:
            settings = json.load(f)

        settings['initial_capital_rub'] = initial_capital
        settings['max_positions'] = max_positions
        settings['max_position_weight_percent'] = max_weight
        settings['stop_loss_percent'] = stop_loss
        settings['take_profit_percent'] = take_profit
        settings['risk_per_trade_percent'] = risk_per_trade
        settings['cooldown_seconds'] = cooldown
        settings['min_cash_per_trade'] = min_cash
        settings['min_risk_per_share_percent'] = min_risk

        if 'daily_profit_fixation' not in settings:
            settings['daily_profit_fixation'] = {}
        settings['daily_profit_fixation']['enabled'] = 'enabled' in fixation_enabled
        settings['daily_profit_fixation']['min_profit_to_fix'] = fixation_min_profit
        settings['daily_profit_fixation']['reinvest_percent'] = fixation_reinvest / 100.0

        with open('config/settings.json', 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)

        if broker_instance is not None:
            broker_instance.settings = settings
            broker_instance.portfolio.initial_capital = initial_capital
            broker_instance.portfolio.max_positions = max_positions
            if hasattr(broker_instance, 'risk_manager'):
                broker_instance.risk_manager.config['cooldown_seconds'] = cooldown
                broker_instance.risk_manager.config['min_cash_per_trade'] = min_cash
                broker_instance.risk_manager.config['min_risk_per_share_percent'] = min_risk

        logger.info(f"Настройки сохранены: капитал={initial_capital}, макс.позиций={max_positions}")
        return "✅ Настройки сохранены!"

    except Exception as e:
        logger.error(f"Ошибка сохранения настроек: {e}")
        return "❌ Ошибка сохранения"


@app.callback(
    Output("session-save-status", "children"),
    [Input("session-save-btn", "n_clicks")],
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
        with open('config/settings.json', 'r', encoding='utf-8') as f:
            settings = json.load(f)

        if 'trading_hours' not in settings:
            settings['trading_hours'] = {}

        settings['trading_hours']['main_session'] = f"{main_start}-{main_end}"
        settings['trading_hours']['evening_session'] = f"{evening_start}-{evening_end}"
        settings['trading_hours']['trade_on_weekend'] = 'weekend' in weekend_trading
        settings['trading_hours']['force_247'] = 'force' in force_trading

        with open('config/settings.json', 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)

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

    if any(source.get('url') == new_url for source in config['sources']):
        return f"❌ Источник {new_url} уже добавлен.", "", new_url, new_name

    new_source = {
        'url': new_url,
        'name': new_name or f"Источник {len(config['sources'])+1}",
        'enabled': True,
        'update_interval': 300
    }
    config['sources'].append(new_source)

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    sources_list = html.Ul([
        html.Li([
            html.Span(f"📰 {src['name']} ", style={'font-weight': 'bold'}),
            html.Span(f"({src['url'][:50]}...)"),
            dbc.Button("🗑️", id={'type': 'del-rss-btn', 'index': i}, color="danger", size="sm", className="ms-2")
        ]) for i, src in enumerate(config['sources'])
    ])

    return f"✅ Добавлен: {new_source['name']}", sources_list, "", ""


# Коллбэк логов (новый — из буфера памяти)
@app.callback(
    Output("logs-content", "children"),
    [Input("interval-component", "n_intervals"),
     Input("clear-logs-btn", "n_clicks"),
     Input("log-level-filter", "value"),
     Input("log-component-filter", "value")],
    prevent_initial_call=False
)
def update_logs(n_intervals, clear_clicks, level_filter, component_filter):
    """Обновление содержимого логов из буфера памяти"""
    from utils.logger import get_log_buffer, clear_log_buffer

    ctx_triggered = ctx.triggered_id
    if ctx_triggered == "clear-logs-btn" and clear_clicks:
        clear_log_buffer()
        return html.P("Логи очищены", className="text-muted")

    log_buffer = get_log_buffer()
    lines = list(log_buffer)

    if not lines:
        return html.P("Нет записей в журнале", className="text-muted")

    COMPONENT_COLORS = {
        'SMART_BROKER': '#00ff88', 'RISK_MANAGER': '#ffaa00', 'TECH_CORE': '#00aaff',
        'TRAINER': '#aa00ff', 'MOEX_FETCHER': '#ff8800', 'NEWS_FETCHER_OPT': '#00ccff',
        'PORTFOLIO': '#ff00aa', 'MAIN': '#ffffff', 'SYSTEM': '#cccccc',
        'DEEP_DIAG': '#888888', 'TEST_FIXES': '#888888', 'TEST_8_STAGES': '#888888'
    }

    log_entries = []
    for entry in lines:
        name = entry.get('name', 'SYSTEM')
        level = entry.get('level', 'INFO')
        message = entry.get('message', '')
        timestamp = entry.get('timestamp', '')

        if isinstance(timestamp, str):
            try:
                dt = datetime.fromisoformat(timestamp)
                time_str = dt.strftime('%H:%M:%S')
            except:
                time_str = timestamp[:8] if len(timestamp) >= 8 else timestamp
        else:
            time_str = str(timestamp)[:8]

        if level_filter == 'TRADES':
            trade_keywords = ['BUY', 'SELL', 'STOP_LOSS', 'TAKE_PROFIT', 'ДНЕВНАЯ ПРИБЫЛЬ', 'КУПЛЕНО', 'ПРОДАНО']
            if not any(kw in message for kw in trade_keywords):
                continue
        elif level_filter != 'ALL' and level != level_filter:
            continue

        if component_filter != 'ALL' and name != component_filter:
            continue

        color = COMPONENT_COLORS.get(name, '#cccccc')
        font_weight = 'normal'
        bg_color = 'transparent'

        if any(kw in message for kw in ['BUY', 'КУПЛЕНО']):
            font_weight = 'bold'; color = '#00ff88'
        elif any(kw in message for kw in ['SELL', 'ПРОДАНО']):
            font_weight = 'bold'; color = '#ff4444'
        elif 'STOP_LOSS' in message:
            font_weight = 'bold'; color = '#ff8800'
        elif 'TAKE_PROFIT' in message:
            font_weight = 'bold'; color = '#ffaa00'
        elif 'ДНЕВНАЯ ПРИБЫЛЬ' in message:
            font_weight = 'bold'; color = '#00ccff'
        elif level in ['ERROR', 'CRITICAL']:
            bg_color = '#330000'

        log_entries.append(
            html.Div([
                html.Span(f"{time_str} ", style={'color': '#888888'}),
                html.Span(f"[{name}] ", style={'color': color, 'fontWeight': 'bold'}),
                html.Span(message, style={'color': color, 'fontWeight': font_weight})
            ], style={
                'margin': '1px 0', 'fontSize': '11px',
                'backgroundColor': bg_color, 'padding': '1px 4px', 'borderRadius': '2px'
            })
        )

    log_entries.reverse()
    return log_entries


# Коллбэк статистики
@app.callback(
    [Output("system-stats", "children"),
     Output("components-status", "children")],
    [Input("interval-component", "n_intervals")]
)
def update_system_stats(n_intervals):
    """Обновление статистики системы с реальными статусами"""
    if broker_instance is None:
        return [html.P("Нет данных"), html.P("Нет данных")]

    try:
        summary = broker_instance.get_portfolio_summary()
        risk_metrics = summary.get('risk_metrics', {})
        reserved_cash = summary.get('reserved_cash', 0)

        stats_html = html.Div([
            html.P(f"🔢 Циклов: {broker_instance.cycle_count}"),
            html.P(f"📈 Сигналов: {len(broker_instance.signals_cache)}"),
            html.P(f"💼 Позиций: {summary.get('positions_count', 0)}"),
            html.P(f"📊 PnL день: {risk_metrics.get('daily_pnl', 0):+,.0f}₽"),
            html.P(f"🎯 Сделок день: {risk_metrics.get('daily_trades', 0)}"),
            html.P(f"🔄 Обучение: {len(broker_instance.model.memory)} опытов"),
            html.P(f"💎 Накоплено: {reserved_cash:+,.0f}₽")
        ])

        model_ok = broker_instance.model is not None
        news_ok = hasattr(broker_instance, 'news_fetcher') and broker_instance.news_fetcher is not None
        tech_ok = hasattr(broker_instance, 'technical_core') and broker_instance.technical_core is not None
        risk_ok = hasattr(broker_instance, 'risk_manager') and broker_instance.risk_manager is not None
        sched_ok = hasattr(broker_instance, 'scheduler') and broker_instance.scheduler is not None
        portf_ok = hasattr(broker_instance, 'portfolio') and broker_instance.portfolio is not None
        trainer_ok = hasattr(broker_instance, 'trainer') and broker_instance.trainer is not None
        trainer_active = trainer_ok and getattr(broker_instance.trainer, 'training_enabled', False)

        components_html = html.Div([
            html.P(f"🤖 AI модель: {'✅' if model_ok else '❌'} {'Активна' if model_ok else 'Недоступна'}"),
            html.P(f"📰 Новости: {'✅' if news_ok else '❌'} {'Активны' if news_ok else 'Недоступны'}"),
            html.P(f"📊 Тех.анализ: {'✅' if tech_ok else '❌'} {'Активен' if tech_ok else 'Недоступен'}"),
            html.P(f"⚖️ Риск-менеджер: {'✅' if risk_ok else '❌'} {'Активен' if risk_ok else 'Недоступен'}"),
            html.P(f"⏰ Планировщик: {'✅' if sched_ok else '❌'} {'Активен' if sched_ok else 'Недоступен'}"),
            html.P(f"💰 Портфель: {'✅' if portf_ok else '❌'} {'Активен' if portf_ok else 'Недоступен'}"),
            html.P(f"🔄 Trainer: {'✅' if trainer_active else '❌'} {'Активен' if trainer_active else 'Остановлен'}")
        ])

        return [stats_html, components_html]

    except Exception as e:
        logger.error(f"Ошибка обновления статистики: {e}")
        return [html.P(f"Ошибка: {e}"), html.P("Ошибка загрузки")]
# Часть 4 из 4: коллбэки графиков, портфеля, утилиты

# Коллбэк графика цены и объема
@app.callback(
    Output("price-volume-chart", "figure"),
    [Input("ticker-selector", "value"),
     Input("period-selector", "value"),
     Input("interval-selector", "value"),
     Input("interval-component", "n_intervals")]
)
def update_price_volume_chart(ticker, period, interval_str, n_intervals):
    """Обновление графика цены и объема"""
    try:
        if broker_instance is None:
            return dashboard_viz._create_empty_chart("Система не инициализирована")

        interval_map = {"1min": 1, "5min": 10, "15min": 10, "30min": 10, "1h": 60, "1d": 24}
        period_count_map = {"1d": 24, "1w": 7 * 24, "1m": 30 * 24, "3m": 90 * 24, "6m": 180 * 24}

        interval = interval_map.get(interval_str, 60)
        count = period_count_map.get(period, 100)

        if interval == 24:
            count = min(count // 24, 365)

        try:
            moex = broker_instance.moex
            candles_data = moex.get_candles(ticker=ticker, interval=interval, count=count)
        except Exception as e:
            logger.error(f"Ошибка получения данных {ticker}: {e}")
            candles_data = None

        if candles_data is None or candles_data.empty:
            base_prices = {
                'SBER': 280, 'GAZP': 160, 'LKOH': 7500, 'YNDX': 3500, 'VTBR': 0.025,
                'ROSN': 580, 'GMKN': 16000, 'NVTK': 1800, 'TCSG': 4500, 'MOEX': 150,
                'ALRS': 100, 'PHOR': 2800
            }
            base_price = base_prices.get(ticker, 100)

            if interval == 24:
                dates = pd.date_range(end=datetime.now(), periods=count, freq='D')
            else:
                dates = pd.date_range(end=datetime.now(), periods=count, freq='H')

            trend = np.random.uniform(-0.01, 0.01, count).cumsum()
            prices = base_price * (1 + trend)
            volatility = base_price * 0.02
            noise = np.random.randn(count) * volatility
            final_prices = prices + noise

            candles_data = pd.DataFrame({
                'Open': final_prices - np.random.rand(count) * (base_price * 0.01),
                'High': final_prices + np.random.rand(count) * (base_price * 0.015),
                'Low': final_prices - np.random.rand(count) * (base_price * 0.01),
                'Close': final_prices,
                'Volume': np.random.randint(10000, 1000000, count)
            }, index=dates)

        return dashboard_viz.create_price_volume_chart(candles_data, ticker)

    except Exception as e:
        logger.error(f"Ошибка обновления графика цены: {e}")
        return dashboard_viz._create_empty_chart(f"Ошибка: {str(e)}")


# Коллбэк графика индикаторов
@app.callback(
    Output("indicators-chart", "figure"),
    [Input("ticker-selector", "value"),
     Input("period-selector", "value"),
     Input("interval-selector", "value"),
     Input("interval-component", "n_intervals")]
)
def update_indicators_chart(ticker, period, interval_str, n_intervals):
    """График индикаторов из свечных данных"""
    try:
        try:
            import talib
            HAS_TALIB = True
        except ImportError:
            HAS_TALIB = False

        if not HAS_TALIB:
            return dashboard_viz._create_empty_chart(
                "TA-Lib не установлен.\nУстановите: pip install TA-Lib"
            )

        if broker_instance is None:
            return dashboard_viz._create_empty_chart("Система не инициализирована")

        interval_map = {"1min": 1, "5min": 10, "15min": 10, "30min": 10, "1h": 60, "1d": 24}
        interval = interval_map.get(interval_str, 60)
        count = 50

        candles_data = broker_instance.moex.get_candles(ticker=ticker, interval=interval, count=count)

        if candles_data is None or candles_data.empty:
            return dashboard_viz._create_empty_chart(f"Нет данных для {ticker}")

        if 'Close' in candles_data.columns:
            close_prices = candles_data['Close'].values
        elif 'close' in candles_data.columns:
            close_prices = candles_data['close'].values
        else:
            close_prices = candles_data['Open'].values

        if len(close_prices) < 20:
            return dashboard_viz._create_empty_chart(f"Мало данных для расчета индикаторов")

        indicators_data = {'prices': close_prices.tolist()}

        # RSI
        try:
            rsi_values = talib.RSI(close_prices, timeperiod=14)
            indicators_data['rsi'] = rsi_values.tolist()
        except:
            indicators_data['rsi'] = (50 + 20 * np.sin(np.linspace(0, 4 * np.pi, len(close_prices)))).tolist()

        # MACD
        try:
            macd, macd_signal, macd_hist = talib.MACD(close_prices)
            indicators_data['macd'] = macd.tolist()
            indicators_data['macd_signal'] = macd_signal.tolist()
            indicators_data['macd_hist'] = macd_hist.tolist()
        except:
            indicators_data['macd'] = (np.sin(np.linspace(0, 6 * np.pi, len(close_prices))) * 2).tolist()
            indicators_data['macd_signal'] = (np.sin(np.linspace(0, 6 * np.pi, len(close_prices)) - 0.5) * 1.8).tolist()
            indicators_data['macd_hist'] = (np.random.randn(len(close_prices)) * 0.5).tolist()

        # Bollinger Bands
        try:
            upper, middle, lower = talib.BBANDS(close_prices, timeperiod=20)
            indicators_data['bb_upper'] = upper.tolist()
            indicators_data['bb_middle'] = middle.tolist()
            indicators_data['bb_lower'] = lower.tolist()
        except:
            base = np.mean(close_prices)
            indicators_data['bb_upper'] = (base + base * 0.05 + np.sin(np.linspace(0, 4 * np.pi, len(close_prices))) * base * 0.02).tolist()
            indicators_data['bb_middle'] = (base + np.sin(np.linspace(0, 4 * np.pi, len(close_prices))) * base * 0.01).tolist()
            indicators_data['bb_lower'] = (base - base * 0.05 + np.sin(np.linspace(0, 4 * np.pi, len(close_prices))) * base * 0.02).tolist()

        return dashboard_viz.create_indicators_chart(indicators_data, ticker)

    except Exception as e:
        logger.error(f"Ошибка обновления графика индикаторов: {e}")
        return dashboard_viz._create_empty_chart(f"Ошибка: {str(e)}")


# Коллбэк графика сентимента
@app.callback(
    Output("sentiment-chart", "figure"),
    [Input("ticker-selector", "value"),
     Input("interval-component", "n_intervals")]
)
def update_sentiment_chart(ticker, n_intervals):
    """Обновление графика новостного сентимента"""
    try:
        if broker_instance is None:
            return dashboard_viz._create_empty_chart("Система не инициализирована")

        sentiment_data = []

        if hasattr(broker_instance, 'get_sentiment_history'):
            all_sentiment = broker_instance.get_sentiment_history(limit=50)
            for item in all_sentiment:
                item_ticker = item.get('ticker', 'MARKET')
                if item_ticker == ticker or (ticker == 'MOEX' and item_ticker == 'MARKET'):
                    sentiment_data.append(item)
            if not sentiment_data and ticker != 'MOEX':
                sentiment_data = [item for item in all_sentiment if item.get('ticker') == 'MARKET']

        if not sentiment_data:
            now = datetime.now()
            base_sentiment = 0.2 if ticker == 'SBER' else 0.0
            sentiment_data = [
                {
                    'timestamp': (now - timedelta(hours=i)).isoformat(),
                    'sentiment': base_sentiment + 0.1 * np.sin(i * 0.5),
                    'source': 'TEST',
                    'ticker': ticker if ticker != 'MOEX' else 'MARKET'
                }
                for i in range(20, 0, -1)
            ]

        if not sentiment_data:
            return dashboard_viz._create_empty_chart(f"Нет данных по сентименту для {ticker}")

        return dashboard_viz.create_sentiment_chart(sentiment_data)

    except Exception as e:
        logger.error(f"Ошибка обновления графика сентимента: {e}")
        return dashboard_viz._create_empty_chart(f"Ошибка: {str(e)}")


# Коллбэки страницы портфеля
@app.callback(
    [Output("detailed-positions-table", "children"),
     Output("sector-pie-chart", "figure"),
     Output("portfolio-value-chart", "figure"),
     Output("trade-history-table", "children")],
    [Input("interval-component", "n_intervals"),
     Input("nav-portfolio", "n_clicks")]
)
def update_portfolio_page(n_intervals, nav_clicks):
    """Обновление данных на странице портфеля"""
    if broker_instance is None:
        empty_chart = dashboard_viz._create_empty_chart("Система не инициализирована")
        return ["Нет данных", empty_chart, empty_chart, "Нет данных"]

    try:
        summary = get_portfolio_summary_safe()
        positions = summary.get('positions', [])
        positions_table = create_real_positions_table(positions)
        sector_chart = create_real_sector_chart(positions)
        value_chart = create_real_portfolio_value_chart(summary)
        trade_history = get_real_trade_history()
        trade_table = create_real_trade_history_table(trade_history)
        return [positions_table, sector_chart, value_chart, trade_table]

    except Exception as e:
        logger.error(f"Ошибка обновления страницы портфеля: {e}")
        empty_chart = dashboard_viz._create_empty_chart(f"Ошибка загрузки")
        return ["Ошибка загрузки", empty_chart, empty_chart, "Ошибка загрузки"]


def get_portfolio_summary_safe():
    """Безопасное получение сводки портфеля"""
    try:
        if broker_instance and hasattr(broker_instance, 'get_portfolio_summary'):
            summary = broker_instance.get_portfolio_summary()
            if 'positions' in summary and broker_instance.portfolio:
                try:
                    with open('data/portfolio_state.json', 'r', encoding='utf-8') as f:
                        portfolio_data = json.load(f)
                        saved_positions = portfolio_data.get('positions', {})
                    for i, pos in enumerate(summary['positions']):
                        ticker = pos.get('ticker')
                        if ticker and ticker in saved_positions:
                            saved_strategy = saved_positions[ticker].get('strategy')
                            if saved_strategy and pos.get('strategy', 'unknown') == 'unknown':
                                summary['positions'][i]['strategy'] = saved_strategy
                except:
                    pass
            return summary
        return get_portfolio_summary_fallback()
    except:
        return get_portfolio_summary_fallback()


def get_portfolio_summary_fallback():
    """Резервный метод получения сводки портфеля"""
    try:
        portfolio = broker_instance.portfolio if broker_instance else None
        if not portfolio:
            return {"positions": []}
        prices = broker_instance._get_current_prices() if hasattr(broker_instance, '_get_current_prices') else {}
        positions_detail = []
        total_value = 0
        for ticker, pos in portfolio.positions.items():
            current_price = prices.get(ticker, pos.get('avg_price', 0))
            quantity = pos.get('qty', 0)
            avg_price = pos.get('avg_price', 0)
            position_value = quantity * current_price
            total_value += position_value
            positions_detail.append({
                'ticker': ticker, 'quantity': quantity, 'avg_price': avg_price,
                'current_price': current_price, 'position_value': position_value,
                'pnl': (current_price - avg_price) * quantity,
                'pnl_percent': ((current_price / avg_price) - 1) * 100 if avg_price > 0 else 0,
                'weight': (position_value / total_value * 100) if total_value > 0 else 0,
                'buy_time': pos.get('buy_time'), 'strategy': pos.get('strategy', 'unknown')
            })
        return {
            'positions': positions_detail,
            'total_value': total_value + (portfolio.cash if hasattr(portfolio, 'cash') else 0),
            'cash': portfolio.cash if hasattr(portfolio, 'cash') else 0,
            'positions_count': len(portfolio.positions),
            'initial_capital': portfolio.initial_capital if hasattr(portfolio, 'initial_capital') else 10000
        }
    except:
        return {"positions": []}


def create_real_positions_table(positions):
    """Создание таблицы позиций"""
    if not positions:
        return html.Div([
            html.P("Нет открытых позиций", className="text-muted text-center"),
            html.Small("Система ожидает торговых сигналов", className="text-muted d-block text-center")
        ], className="p-4")

    positions_sorted = sorted(positions, key=lambda x: x.get('position_value', 0), reverse=True)

    header = html.Thead(html.Tr([
        html.Th("Тикер"), html.Th("Стратегия"), html.Th("Кол-во"), html.Th("Ср. цена"),
        html.Th("Тек. цена"), html.Th("Стоимость"), html.Th("PnL ₽"), html.Th("PnL %"),
        html.Th("Вес"), html.Th("Дней")
    ]))

    strategy_colors = {
        'news_aggressive': 'badge bg-danger', 'momentum': 'badge bg-warning',
        'mean_reversion': 'badge bg-info', 'balanced': 'badge bg-primary',
        'conservative': 'badge bg-success', 'unknown': 'badge bg-secondary'
    }

    rows = []
    for pos in positions_sorted:
        ticker = pos.get('ticker', 'N/A')
        strategy = pos.get('strategy', 'unknown')
        days_held = 0
        buy_time = pos.get('buy_time')
        if buy_time:
            try:
                if isinstance(buy_time, (int, float)):
                    buy_dt = datetime.fromtimestamp(buy_time)
                elif isinstance(buy_time, str):
                    buy_dt = datetime.fromisoformat(buy_time.replace('Z', '+00:00'))
                else:
                    buy_dt = datetime.now()
                days_held = max(0, (datetime.now() - buy_dt).days)
            except:
                days_held = 0

        pnl = pos.get('pnl', 0)
        pnl_percent = pos.get('pnl_percent', 0)
        pnl_class = "text-success" if pnl >= 0 else "text-danger"
        strategy_class = strategy_colors.get(strategy, 'badge bg-secondary')

        rows.append(html.Tr([
            html.Td(html.Strong(ticker)),
            html.Td(html.Span(strategy, className=f"{strategy_class} badge-sm")),
            html.Td(f"{pos.get('quantity', 0):,}"),
            html.Td(f"{pos.get('avg_price', 0):.2f}"),
            html.Td(f"{pos.get('current_price', 0):.2f}"),
            html.Td(f"{pos.get('position_value', 0):,.0f} ₽"),
            html.Td(html.Span(f"{pnl:+,.0f} ₽", className=pnl_class)),
            html.Td(html.Span(f"{pnl_percent:+.1f}%", className="text-success" if pnl_percent >= 0 else "text-danger")),
            html.Td(f"{pos.get('weight', 0):.1f}%"),
            html.Td(f"{days_held}")
        ]))

    total_positions_value = sum(p.get('position_value', 0) for p in positions_sorted)
    total_pnl = sum(p.get('pnl', 0) for p in positions_sorted)
    total_pnl_class = "text-success" if total_pnl >= 0 else "text-danger"

    summary = html.Div([
        html.Hr(),
        html.Div([
            html.Span("Всего позиций: ", className="text-muted"),
            html.Strong(f"{len(positions)}", className="ms-2"),
            html.Span("Общая стоимость: ", className="text-muted ms-4"),
            html.Strong(f"{total_positions_value:,.0f} ₽", className="ms-2"),
            html.Span("Общий PnL: ", className="text-muted ms-4"),
            html.Strong(f"{total_pnl:+,.0f} ₽", className=f"ms-2 {total_pnl_class}")
        ], className="d-flex justify-content-between")
    ])

    return html.Div([html.Table([header, html.Tbody(rows)], className="table table-sm table-hover table-striped"), summary], className="table-responsive")


def create_real_sector_chart(positions):
    """График распределения по секторам"""
    if not positions:
        return dashboard_viz._create_empty_chart("Нет открытых позиций")

    try:
        sector_info = load_sector_info()
        sector_values = {}
        for pos in positions:
            ticker = pos.get('ticker', '')
            position_value = pos.get('position_value', 0)
            sector = sector_info['sectors'].get(ticker, 'Не определен')
            if sector not in sector_values:
                sector_values[sector] = 0
            sector_values[sector] += position_value

        total_value = sum(sector_values.values())
        filtered_sectors = {}
        other_value = 0
        for sector, value in sector_values.items():
            if total_value > 0 and (value / total_value) >= 0.01:
                filtered_sectors[sector] = value
            else:
                other_value += value
        if other_value > 0:
            filtered_sectors['Другое'] = other_value

        if not filtered_sectors:
            return dashboard_viz._create_empty_chart("Недостаточно данных по секторам")

        sectors = list(filtered_sectors.keys())
        values = list(filtered_sectors.values())
        sector_colors = sector_info['sector_colors']
        colors = [sector_colors.get(s, '#cccccc') for s in sectors]

        fig = go.Figure(data=[go.Pie(
            labels=sectors, values=values, hole=0.4, marker=dict(colors=colors),
            textinfo='label+percent', textposition='inside',
            hovertemplate="<b>%{label}</b><br>Стоимость: %{value:,.0f}₽<br>Доля: %{percent}<extra></extra>"
        )])

        fig.update_layout(
            title=dict(text='📊 Распределение по секторам', font=dict(size=16, color='white')),
            template='plotly_dark', height=400, margin=dict(l=20, r=20, t=60, b=20),
            showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        return fig

    except Exception as e:
        logger.error(f"Ошибка создания графика секторов: {e}")
        return dashboard_viz._create_empty_chart("Ошибка загрузки секторов")


def load_sector_info():
    """Загрузка информации о секторах"""
    try:
        with open('config/ticker_sectors.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {
            'sectors': {
                'SBER': 'Финансы', 'VTBR': 'Финансы', 'GAZP': 'Нефть и газ',
                'LKOH': 'Нефть и газ', 'ROSN': 'Нефть и газ', 'GMKN': 'Металлургия',
                'MTSS': 'Телеком', 'MGNT': 'Потребительские товары', 'PHOR': 'Химия', 'IRAO': 'Энергетика'
            },
            'sector_colors': {
                'Финансы': '#00ff88', 'Нефть и газ': '#0088ff', 'Металлургия': '#ff4444',
                'Телеком': '#ffaa00', 'Потребительские товары': '#00cc66', 'Химия': '#00aaff',
                'Энергетика': '#ff8800', 'Не определен': '#cccccc', 'Другое': '#999999'
            }
        }


def create_real_portfolio_value_chart(summary):
    """График стоимости портфеля"""
    try:
        history_data = load_portfolio_history()
        if not history_data:
            history_data = [{
                'timestamp': datetime.now().isoformat(),
                'total_value': summary.get('total_value', 0),
                'cash': summary.get('cash', 0),
                'positions_value': summary.get('total_value', 0) - summary.get('cash', 0)
            }]

        dates, total_values, cash_values, positions_values = [], [], [], []
        for point in history_data[-30:]:
            try:
                dates.append(datetime.fromisoformat(point.get('timestamp', '').replace('Z', '+00:00')) if point.get('timestamp') else datetime.now())
                total_values.append(point.get('total_value', 0))
                cash_values.append(point.get('cash', 0))
                positions_values.append(point.get('positions_value', 0))
            except:
                continue

        if not dates:
            return dashboard_viz._create_empty_chart("Нет исторических данных")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=total_values, mode='lines+markers', name='Общая стоимость',
                                 line=dict(color='#00ff88', width=3), marker=dict(size=6)))
        if any(v > 0 for v in positions_values):
            fig.add_trace(go.Scatter(x=dates, y=positions_values, mode='lines', name='Стоимость позиций',
                                     line=dict(color='#0088ff', width=2, dash='dash'),
                                     fill='tonexty', fillcolor='rgba(0, 136, 255, 0.1)'))
        if any(v > 0 for v in cash_values):
            fig.add_trace(go.Scatter(x=dates, y=cash_values, mode='lines', name='Кэш',
                                     line=dict(color='#ffaa00', width=2, dash='dot'),
                                     fill='tonexty', fillcolor='rgba(255, 170, 0, 0.1)'))

        initial_capital = summary.get('initial_capital', total_values[0] if total_values else 0)
        fig.update_layout(
            title=dict(text='📈 Динамика стоимости портфеля', font=dict(size=16, color='white')),
            xaxis_title="Дата", yaxis_title="Стоимость, ₽", template='plotly_dark',
            hovermode='x unified', showlegend=True, height=400,
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        if initial_capital > 0:
            fig.add_hline(y=initial_capital, line_dash="dash", line_color="#ff4444",
                          annotation_text=f"Начальный капитал: {initial_capital:,.0f}₽",
                          annotation_position="bottom right", annotation_font=dict(size=10))
        current_value = total_values[-1] if total_values else 0
        fig.add_hline(y=current_value, line_dash="dot", line_color="#00ff88",
                      annotation_text=f"Текущая: {current_value:,.0f}₽",
                      annotation_position="top right", annotation_font=dict(size=10))
        fig.update_yaxes(tickprefix="₽", tickformat=",")
        fig.update_xaxes(tickformat="%d.%m")
        return fig

    except Exception as e:
        logger.error(f"Ошибка создания графика стоимости: {e}")
        return dashboard_viz._create_empty_chart("Ошибка создания графика")


def load_portfolio_history():
    """Загрузка истории портфеля"""
    for file_path in ['data/portfolio_history.json', 'data/daily_report.json']:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif 'history' in data:
                    return data['history']
        except:
            continue
    return create_history_from_trades()


def create_history_from_trades():
    """Создание истории из сделок"""
    try:
        if not broker_instance or not hasattr(broker_instance.portfolio, 'trade_history'):
            return []
        trades = broker_instance.portfolio.trade_history
        if not trades:
            return []
        history = []
        current_cash = broker_instance.portfolio.initial_capital
        positions_value = 0
        for trade in sorted(trades, key=lambda x: x.get('timestamp', '')):
            if trade.get('action') == 'BUY':
                current_cash -= trade.get('cost', 0)
                positions_value += trade.get('cost', 0)
            elif trade.get('action') == 'SELL':
                current_cash += trade.get('revenue', 0)
                positions_value -= trade.get('revenue', 0)
            history.append({
                'timestamp': trade.get('timestamp', ''),
                'total_value': current_cash + positions_value,
                'cash': current_cash, 'positions_value': positions_value
            })
        return history[-30:]
    except:
        return []


def get_real_trade_history():
    """Получение реальной истории сделок"""
    try:
        if broker_instance and hasattr(broker_instance.portfolio, 'trade_history'):
            return broker_instance.portfolio.trade_history[-50:]
        elif broker_instance and hasattr(broker_instance.portfolio, 'get_trade_history_summary'):
            return broker_instance.portfolio.get_trade_history_summary(limit=50)
        else:
            with open('data/portfolio_state.json', 'r', encoding='utf-8') as f:
                return json.load(f).get('trade_history', [])[-50:]
    except:
        return []


def create_real_trade_history_table(trades):
    """Создание таблицы истории сделок"""
    if not trades:
        return html.Div([
            html.P("Нет истории сделок", className="text-muted text-center"),
            html.Small("Сделки появятся после начала торговли", className="text-muted d-block text-center")
        ], className="p-4")

    trades_sorted = sorted(trades, key=lambda x: x.get('timestamp', ''), reverse=True)

    header = html.Thead(html.Tr([
        html.Th("Время"), html.Th("Тикер"), html.Th("Действие"), html.Th("Кол-во"),
        html.Th("Цена"), html.Th("Сумма"), html.Th("PnL"), html.Th("Стратегия")
    ]))

    action_map = {
        'BUY': ("badge bg-success", "↗"), 'SELL': ("badge bg-danger", "↘"),
        'STOP_LOSS': ("badge bg-warning", "⛔"), 'TAKE_PROFIT': ("badge bg-info", "✅")
    }
    strategy_colors = {
        'news_aggressive': 'badge-sm bg-danger', 'momentum': 'badge-sm bg-warning',
        'mean_reversion': 'badge-sm bg-info', 'balanced': 'badge-sm bg-primary',
        'conservative': 'badge-sm bg-success', 'unknown': 'badge-sm bg-secondary'
    }

    rows = []
    for trade in trades_sorted[:20]:
        timestamp = trade.get('timestamp', '')
        time_str = ''
        if timestamp:
            try:
                if isinstance(timestamp, (int, float)):
                    dt = datetime.fromtimestamp(timestamp)
                elif isinstance(timestamp, str):
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                else:
                    dt = datetime.now()
                time_str = dt.strftime('%d.%m %H:%M')
            except:
                time_str = str(timestamp)[:16]

        action = trade.get('action', '')
        action_class, action_icon = action_map.get(action, ("badge bg-secondary", "➡"))
        pnl = trade.get('pnl', 0)
        pnl_text = f"{pnl:+,.0f} ₽" if pnl != 0 else "-"
        amount = trade.get('cost', trade.get('revenue', 0))
        strategy = trade.get('strategy', 'unknown')
        strategy_class = strategy_colors.get(strategy, 'badge-sm bg-secondary')

        rows.append(html.Tr([
            html.Td(time_str), html.Td(html.Strong(trade.get('ticker', ''))),
            html.Td(html.Span(f"{action_icon} {action}", className=f"{action_class}")),
            html.Td(f"{trade.get('quantity', 0):,}"), html.Td(f"{trade.get('price', 0):.2f}"),
            html.Td(f"{amount:,.0f} ₽"),
            html.Td(html.Span(pnl_text, className="text-success" if pnl >= 0 else "text-danger")),
            html.Td(html.Span(strategy, className=f"{strategy_class}"))
        ]))

    buy_count = sum(1 for t in trades_sorted if t.get('action') == 'BUY')
    sell_count = sum(1 for t in trades_sorted if t.get('action') == 'SELL')
    total_pnl = sum(t.get('pnl', 0) for t in trades_sorted)

    summary = html.Div([
        html.Hr(),
        html.Div([
            html.Span("Всего сделок: ", className="text-muted"),
            html.Strong(f"{len(trades_sorted)}", className="ms-2"),
            html.Span("Покупки: ", className="text-muted ms-4"),
            html.Strong(f"{buy_count}", className="ms-2 text-success"),
            html.Span("Продажи: ", className="text-muted ms-4"),
            html.Strong(f"{sell_count}", className="ms-2 text-danger"),
            html.Span("Общий PnL: ", className="text-muted ms-4"),
            html.Strong(f"{total_pnl:+,.0f} ₽", className=f"ms-2 {'text-success' if total_pnl >= 0 else 'text-danger'}")
        ], className="d-flex justify-content-between flex-wrap")
    ])

    return html.Div([html.Table([header, html.Tbody(rows)], className="table table-sm table-hover table-striped"), summary], className="table-responsive")


def save_portfolio_history():
    """Сохранение текущего состояния портфеля в историю"""
    if broker_instance is None:
        return
    try:
        history_file = 'data/portfolio_history.json'
        history_data = []
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
                if not isinstance(history_data, list):
                    history_data = []
        except:
            history_data = []
        summary = get_portfolio_summary_safe()
        history_data.append({
            'timestamp': datetime.now().isoformat(),
            'total_value': summary.get('total_value', 0),
            'cash': summary.get('cash', 0),
            'positions_value': summary.get('total_value', 0) - summary.get('cash', 0),
            'positions_count': summary.get('positions_count', 0),
            'pnl_total': summary.get('total_value', 0) - summary.get('initial_capital', 0)
        })
        history_data = history_data[-100:]
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, default=str)
        logger.debug(f"История портфеля сохранена ({len(history_data)} записей)")
    except Exception as e:
        logger.error(f"Ошибка сохранения истории портфеля: {e}")


@app.callback(
    Output("ticker-selector", "options"),
    [Input("interval-component", "n_intervals")],
    prevent_initial_call=False
)
def update_ticker_dropdown_simple(n_intervals):
    """Обновление выпадающего списка тикеров"""
    try:
        portfolio_tickers = get_portfolio_tickers()
        if not portfolio_tickers:
            return [{'label': f"{ticker}", 'value': ticker} for ticker in ['SBER', 'GAZP', 'LKOH', 'MOEX', 'VTBR', 'ROSN', 'GMKN']]
        options = []
        for ticker in sorted(portfolio_tickers):
            if ticker == 'MOEX':
                options.append({'label': "MOEX (Московская биржа)", 'value': ticker})
            else:
                options.append({'label': f"{ticker} (в портфеле)", 'value': ticker})
        return options
    except:
        return [{'label': 'SBER (Сбербанк)', 'value': 'SBER'}]


# Динамическое обновление списка компонентов в фильтре логов
@app.callback(
    Output("log-component-filter", "options"),
    Input("interval-component", "n_intervals")
)
def update_component_filter(n_intervals):
    """Обновление выпадающего списка компонентов на основе буфера логов"""
    from utils.logger import get_log_buffer

    log_buffer = get_log_buffer()
    names = set()
    for entry in log_buffer:
        name = entry.get('name', '')
        if name:
            names.add(name)

    options = [{'label': 'Все', 'value': 'ALL'}]
    options += [{'label': name, 'value': name} for name in sorted(names)]

    return options

@app.callback(
    Output("reload-configs-btn", "children"),
    Input("reload-configs-btn", "n_clicks")
)
def reload_configs(n_clicks):
    """Перезагрузка конфигов на лету"""
    if n_clicks and broker_instance is not None:
        try:
            broker_instance.reload_configs()
            logger.info("Конфиги перезагружены через дашборд")
            return "✅ Конфиги обновлены!"
        except Exception as e:
            logger.error(f"Ошибка перезагрузки конфигов: {e}")
            return "❌ Ошибка!"
    return "🔧 Обновить конфиги"


if __name__ == '__main__':
    print("Запуск веб-интерфейса...")
    app.run(debug=True, port=8050, host='127.0.0.1')