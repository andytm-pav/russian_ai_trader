"""
Логика веб-дашборда и визуализации данных
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json

from utils.logger import setup_logger

logger = setup_logger("DASHBOARD")


class DashboardVisualizer:
    """Класс для визуализации данных на дашборде"""

    def __init__(self):
        self.color_scheme = {
            'primary': '#00ff88',
            'secondary': '#0088ff',
            'danger': '#ff4444',
            'warning': '#ffaa00',
            'success': '#00cc66',
            'info': '#00aaff',
            'dark': '#1a1a1a',
            'light': '#f8f9fa'
        }

        logger.info("Инициализирован DashboardVisualizer")

    def create_portfolio_value_chart(self,
                                     history_data: List[Dict]) -> go.Figure:
        """Создание графика стоимости портфеля"""
        if not history_data:
            return self._create_empty_chart("Нет данных по стоимости портфеля")

        try:
            # Подготовка данных
            dates = []
            values = []

            for point in history_data:
                if 'timestamp' in point and 'total_value' in point:
                    dates.append(datetime.fromisoformat(point['timestamp'].replace('Z', '+00:00')))
                    values.append(point['total_value'])

            if not dates:
                return self._create_empty_chart("Нет данных по стоимости портфеля")

            # Создание графика
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=dates,
                y=values,
                mode='lines+markers',
                name='Стоимость портфеля',
                line=dict(color=self.color_scheme['primary'], width=2),
                marker=dict(size=4),
                fill='tozeroy',
                fillcolor='rgba(0, 255, 136, 0.1)'
            ))

            # Добавление линии начального капитала
            if len(values) > 0:
                initial_value = values[0]
                fig.add_hline(
                    y=initial_value,
                    line_dash="dash",
                    line_color=self.color_scheme['warning'],
                    annotation_text=f"Начальный капитал: {initial_value:,.0f}₽",
                    annotation_position="bottom right"
                )

            # Настройки графика
            fig.update_layout(
                title=dict(
                    text='📈 Динамика стоимости портфеля',
                    font=dict(size=16, color='white')
                ),
                xaxis_title="Дата",
                yaxis_title="Стоимость, ₽",
                template='plotly_dark',
                hovermode='x unified',
                showlegend=True,
                height=400,
                margin=dict(l=40, r=40, t=60, b=40)
            )

            # Форматирование осей
            fig.update_yaxes(tickprefix="₽", tickformat=",")
            fig.update_xaxes(tickformat="%d.%m %H:%M")

            return fig

        except Exception as e:
            logger.error(f"Ошибка создания графика стоимости: {e}")
            return self._create_empty_chart(f"Ошибка: {str(e)}")

    def create_sector_pie_chart(self,
                                positions: List[Dict],
                                ticker_info: Dict[str, Dict]) -> go.Figure:
        """Создание круговой диаграммы распределения по секторам"""
        if not positions:
            return self._create_empty_chart("Нет открытых позиций")

        try:
            # Группировка по секторам
            sector_values = {}

            for pos in positions:
                ticker = pos['ticker']
                position_value = pos.get('position_value', 0)

                # Получаем сектор из информации о тикере
                sector = ticker_info.get(ticker, {}).get('sector', 'Другое')

                if sector not in sector_values:
                    sector_values[sector] = 0

                sector_values[sector] += position_value

            if not sector_values:
                return self._create_empty_chart("Нет данных по секторам")

            # Подготовка данных
            sectors = list(sector_values.keys())
            values = list(sector_values.values())
            total_value = sum(values)

            # Проценты
            percentages = [v / total_value * 100 for v in values]

            # Цвета для секторов
            colors = ['#00ff88', '#0088ff', '#ff4444', '#ffaa00', '#00cc66',
                      '#00aaff', '#aa00ff', '#ff00aa', '#ff8800', '#88ff00']

            # Создание диаграммы
            fig = go.Figure(data=[go.Pie(
                labels=sectors,
                values=values,
                hole=0.4,
                marker=dict(colors=colors[:len(sectors)]),
                textinfo='label+percent',
                textposition='inside',
                hovertemplate="<b>%{label}</b><br>" +
                              "Стоимость: %{value:,.0f}₽<br>" +
                              "Доля: %{percent}<extra></extra>"
            )])

            # Настройки
            fig.update_layout(
                title=dict(
                    text='📊 Распределение по секторам',
                    font=dict(size=16, color='white')
                ),
                template='plotly_dark',
                height=400,
                margin=dict(l=20, r=20, t=60, b=20),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                )
            )

            return fig

        except Exception as e:
            logger.error(f"Ошибка создания круговой диаграммы: {e}")
            return self._create_empty_chart(f"Ошибка: {str(e)}")

    def create_price_volume_chart(self,
                                  candles_data: pd.DataFrame,
                                  ticker: str) -> go.Figure:
        """Создание графика цены и объема"""
        if candles_data is None or candles_data.empty:
            return self._create_empty_chart(f"Нет данных для {ticker}")

        try:
            # Создание subplot
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.7, 0.3]
            )

            # График цены (свечи)
            fig.add_trace(
                go.Candlestick(
                    x=candles_data.index,
                    open=candles_data['Open'],
                    high=candles_data['High'],
                    low=candles_data['Low'],
                    close=candles_data['Close'],
                    name='Цена',
                    increasing_line_color=self.color_scheme['success'],
                    decreasing_line_color=self.color_scheme['danger']
                ),
                row=1, col=1
            )

            # Добавление скользящих средних
            if len(candles_data) >= 20:
                candles_data['SMA20'] = candles_data['Close'].rolling(window=20).mean()
                fig.add_trace(
                    go.Scatter(
                        x=candles_data.index,
                        y=candles_data['SMA20'],
                        name='SMA 20',
                        line=dict(color=self.color_scheme['warning'], width=1)
                    ),
                    row=1, col=1
                )

            if len(candles_data) >= 50:
                candles_data['SMA50'] = candles_data['Close'].rolling(window=50).mean()
                fig.add_trace(
                    go.Scatter(
                        x=candles_data.index,
                        y=candles_data['SMA50'],
                        name='SMA 50',
                        line=dict(color=self.color_scheme['info'], width=1)
                    ),
                    row=1, col=1
                )

            # График объема
            colors_volume = []
            for i in range(len(candles_data)):
                if i == 0:
                    colors_volume.append(self.color_scheme['info'])
                else:
                    if candles_data['Close'].iloc[i] > candles_data['Close'].iloc[i - 1]:
                        colors_volume.append(self.color_scheme['success'])
                    else:
                        colors_volume.append(self.color_scheme['danger'])

            fig.add_trace(
                go.Bar(
                    x=candles_data.index,
                    y=candles_data['Volume'],
                    name='Объем',
                    marker_color=colors_volume,
                    opacity=0.7
                ),
                row=2, col=1
            )

            # Настройки
            fig.update_layout(
                title=dict(
                    text=f'📈 {ticker} - Цена и объем',
                    font=dict(size=16, color='white')
                ),
                template='plotly_dark',
                height=600,
                hovermode='x unified',
                showlegend=True,
                xaxis_rangeslider_visible=False,
                margin=dict(l=40, r=40, t=60, b=40)
            )

            # Настройка осей
            fig.update_yaxes(title_text="Цена, ₽", row=1, col=1, tickprefix="₽")
            fig.update_yaxes(title_text="Объем", row=2, col=1)
            fig.update_xaxes(title_text="Время", row=2, col=1)

            return fig

        except Exception as e:
            logger.error(f"Ошибка создания графика цены: {e}")
            return self._create_empty_chart(f"Ошибка: {str(e)}")

    def create_indicators_chart(self,
                                indicators_data: Dict[str, Any],
                                ticker: str) -> go.Figure:
        """Создание графика технических индикаторов"""
        if not indicators_data:
            return self._create_empty_chart(f"Нет индикаторов для {ticker}")

        try:
            # Создание subplot для индикаторов
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.4, 0.3, 0.3],
                subplot_titles=('RSI', 'MACD', 'Bollinger Bands')
            )

            # RSI
            if 'rsi' in indicators_data and indicators_data['rsi']:
                rsi_values = indicators_data['rsi']
                timestamps = list(range(len(rsi_values)))

                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=rsi_values,
                        name='RSI',
                        line=dict(color=self.color_scheme['primary'], width=2)
                    ),
                    row=1, col=1
                )

                # Уровни перекупленности/перепроданности
                fig.add_hline(y=70, line_dash="dash",
                              line_color=self.color_scheme['danger'],
                              row=1, col=1)
                fig.add_hline(y=30, line_dash="dash",
                              line_color=self.color_scheme['success'],
                              row=1, col=1)

            # MACD
            if 'macd' in indicators_data and 'macd_signal' in indicators_data:
                macd_values = indicators_data['macd']
                signal_values = indicators_data['macd_signal']
                timestamps = list(range(len(macd_values)))

                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=macd_values,
                        name='MACD',
                        line=dict(color=self.color_scheme['warning'], width=2)
                    ),
                    row=2, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=signal_values,
                        name='Signal',
                        line=dict(color=self.color_scheme['info'], width=2)
                    ),
                    row=2, col=1
                )

                # Гистограмма MACD
                if 'macd_hist' in indicators_data:
                    hist_values = indicators_data['macd_hist']
                    colors_hist = ['green' if v >= 0 else 'red' for v in hist_values]

                    fig.add_trace(
                        go.Bar(
                            x=timestamps,
                            y=hist_values,
                            name='MACD Hist',
                            marker_color=colors_hist,
                            opacity=0.6
                        ),
                        row=2, col=1
                    )

            # Bollinger Bands
            if 'bb_upper' in indicators_data and 'bb_lower' in indicators_data:
                upper = indicators_data['bb_upper']
                lower = indicators_data['bb_lower']
                middle = indicators_data.get('bb_middle',
                                             [(u + l) / 2 for u, l in zip(upper, lower)])
                prices = indicators_data.get('prices', middle)
                timestamps = list(range(len(upper)))

                # Цена
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=prices,
                        name='Цена',
                        line=dict(color=self.color_scheme['primary'], width=2)
                    ),
                    row=3, col=1
                )

                # Bollinger Bands
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=upper,
                        name='BB Upper',
                        line=dict(color=self.color_scheme['warning'], width=1, dash='dash'),
                        opacity=0.7
                    ),
                    row=3, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=middle,
                        name='BB Middle',
                        line=dict(color=self.color_scheme['info'], width=1),
                        opacity=0.7
                    ),
                    row=3, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=lower,
                        name='BB Lower',
                        line=dict(color=self.color_scheme['success'], width=1, dash='dash'),
                        opacity=0.7,
                        fill='tonexty',
                        fillcolor='rgba(0, 255, 136, 0.1)'
                    ),
                    row=3, col=1
                )

            # Настройки
            fig.update_layout(
                title=dict(
                    text=f'📊 {ticker} - Технические индикаторы',
                    font=dict(size=16, color='white')
                ),
                template='plotly_dark',
                height=600,
                hovermode='x unified',
                showlegend=True,
                margin=dict(l=40, r=40, t=60, b=40)
            )

            return fig

        except Exception as e:
            logger.error(f"Ошибка создания графика индикаторов: {e}")
            return self._create_empty_chart(f"Ошибка: {str(e)}")

    def create_sentiment_chart(self,
                               sentiment_data: List[Dict]) -> go.Figure:
        """Создание графика новостного сентимента"""
        if not sentiment_data:
            return self._create_empty_chart("Нет данных по сентименту")

        try:
            # Подготовка данных
            timestamps = []
            sentiments = []
            sources = []
            tickers = []

            for item in sentiment_data:
                if 'timestamp' in item and 'sentiment' in item:
                    timestamps.append(
                        datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00'))
                    )
                    sentiments.append(item['sentiment'])
                    sources.append(item.get('source', 'Unknown'))
                    tickers.append(item.get('ticker', 'Unknown'))

            if not timestamps:
                return self._create_empty_chart("Нет данных по сентименту")

            # Создание графика
            fig = go.Figure()

            # Точки сентимента
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=sentiments,
                mode='markers',
                name='Сентимент новостей',
                marker=dict(
                    size=8,
                    color=sentiments,
                    colorscale='RdYlGn',
                    cmin=-1,
                    cmax=1,
                    showscale=True,
                    colorbar=dict(
                        title="Сентимент",
                        titleside="right",
                        tickvals=[-1, -0.5, 0, 0.5, 1],
                        ticktext=["Очень негат.", "Негат.", "Нейтр.", "Позит.", "Очень позит."]
                    )
                ),
                text=[f"{ticker} ({source})" for ticker, source in zip(tickers, sources)],
                hovertemplate="<b>%{text}</b><br>" +
                              "Время: %{x}<br>" +
                              "Сентимент: %{y:.2f}<extra></extra>"
            ))

            # Линия среднего
            if len(sentiments) > 1:
                avg_sentiment = np.mean(sentiments)
                fig.add_hline(
                    y=avg_sentiment,
                    line_dash="dash",
                    line_color=self.color_scheme['warning'],
                    annotation_text=f"Средний: {avg_sentiment:.2f}",
                    annotation_position="bottom right"
                )

            # Нулевая линия
            fig.add_hline(
                y=0,
                line_color='white',
                line_width=1,
                opacity=0.5
            )

            # Настройки
            fig.update_layout(
                title=dict(
                    text='📰 Новостной сентимент',
                    font=dict(size=16, color='white')
                ),
                xaxis_title="Время",
                yaxis_title="Сентимент",
                template='plotly_dark',
                hovermode='x unified',
                showlegend=True,
                height=400,
                margin=dict(l=40, r=40, t=60, b=40),
                yaxis=dict(
                    range=[-1.1, 1.1],
                    tickvals=[-1, -0.5, 0, 0.5, 1]
                )
            )

            return fig

        except Exception as e:
            logger.error(f"Ошибка создания графика сентимента: {e}")
            return self._create_empty_chart(f"Ошибка: {str(e)}")

    def create_performance_metrics(self,
                                   metrics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Создание карточек с метриками производительности"""
        try:
            cards = []

            # Основные метрики
            basic_metrics = [
                ('💰 Общая стоимость',
                 f"{metrics_data.get('total_value', 0):,.0f}₽",
                 'primary'),

                ('💵 Кэш',
                 f"{metrics_data.get('cash', 0):,.0f}₽",
                 'success'),

                ('📊 Позиций',
                 f"{metrics_data.get('positions_count', 0)}",
                 'info'),

                ('📈 Доходность',
                 f"{metrics_data.get('pnl_percent', 0):+.2f}%",
                 'success' if metrics_data.get('pnl_percent', 0) >= 0 else 'danger'),

                ('🎯 Успешность',
                 f"{metrics_data.get('success_rate', 0):.1%}",
                 'success' if metrics_data.get('success_rate', 0) >= 0.5 else 'warning'),

                ('⚡ Активность',
                 f"{metrics_data.get('activity_score', 0):.1f}",
                 'info')
            ]

            for title, value, color in basic_metrics:
                cards.append({
                    'title': title,
                    'value': value,
                    'color': color
                })

            return cards

        except Exception as e:
            logger.error(f"Ошибка создания метрик: {e}")
            return []

    def _create_empty_chart(self, message: str) -> go.Figure:
        """Создание пустого графика с сообщением"""
        fig = go.Figure()

        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color='gray')
        )

        fig.update_layout(
            template='plotly_dark',
            height=400,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=40, r=40, t=40, b=40)
        )

        return fig

    def create_trade_history_table(self,
                                   trade_history: List[Dict]) -> Dict[str, Any]:
        """Создание таблицы истории сделок"""
        if not trade_history:
            return {'columns': [], 'data': []}

        try:
            # Определяем колонки
            columns = [
                {'name': 'Время', 'id': 'time'},
                {'name': 'Тикер', 'id': 'ticker'},
                {'name': 'Действие', 'id': 'action'},
                {'name': 'Кол-во', 'id': 'quantity'},
                {'name': 'Цена', 'id': 'price'},
                {'name': 'Сумма', 'id': 'amount'},
                {'name': 'PnL', 'id': 'pnl'}
            ]

            # Подготовка данных
            data = []
            for trade in trade_history[-50:]:  # Последние 50 сделок
                # Форматирование времени
                timestamp = trade.get('timestamp', '')
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        time_str = dt.strftime('%d.%m %H:%M')
                    except:
                        time_str = timestamp[:16]
                else:
                    time_str = ''

                # Определение цвета действия
                action = trade.get('action', '')
                action_color = 'text-success' if action == 'BUY' else 'text-danger'

                # Расчет суммы
                quantity = trade.get('quantity', 0)
                price = trade.get('price', 0)
                amount = quantity * price

                # PnL
                pnl = trade.get('pnl', 0)
                pnl_color = 'text-success' if pnl >= 0 else 'text-danger'

                data.append({
                    'time': time_str,
                    'ticker': trade.get('ticker', ''),
                    'action': {'value': action, 'color': action_color},
                    'quantity': f"{quantity:,}",
                    'price': f"{price:.2f}",
                    'amount': f"{amount:,.0f}₽",
                    'pnl': {'value': f"{pnl:+,.0f}₽", 'color': pnl_color}
                })

            return {'columns': columns, 'data': data}

        except Exception as e:
            logger.error(f"Ошибка создания таблицы сделок: {e}")
            return {'columns': [], 'data': []}


# Глобальный экземпляр
dashboard_viz = DashboardVisualizer()