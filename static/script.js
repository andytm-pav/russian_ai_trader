/**
 * Основной JavaScript файл для AI Trader
 * Обработка интерфейса и взаимодействие с бэкендом
 */

// Конфигурация
const CONFIG = {
    API_BASE_URL: window.location.origin,
    UPDATE_INTERVAL: 5000, // 5 секунд
    CHART_UPDATE_INTERVAL: 30000, // 30 секунд
    MAX_LOG_ENTRIES: 1000
};

// Глобальное состояние
let APP_STATE = {
    systemActive: false,
    tradingEnabled: false,
    marketOpen: false,
    lastUpdate: null,
    portfolioData: null,
    signalsData: null,
    chartsData: {},
    updateIntervals: {},
    currentPage: 'dashboard'
};

// DOM элементы
const DOM = {
    systemStatus: document.getElementById('system-status'),
    marketStatus: document.getElementById('market-status'),
    modelStatus: document.getElementById('model-status'),
    newsStatus: document.getElementById('news-status'),
    totalCapital: document.getElementById('total-capital'),
    cashAmount: document.getElementById('cash-amount'),
    positionsValue: document.getElementById('positions-value'),
    pnlPercent: document.getElementById('pnl-percent'),
    dailyPnl: document.getElementById('daily-pnl'),
    totalPnl: document.getElementById('total-pnl'),
    pageContent: document.getElementById('page-content'),
    pageTitle: document.getElementById('page-title'),
    pageSubtitle: document.getElementById('page-subtitle'),
    currentTime: document.getElementById('current-time'),
    lastUpdate: document.getElementById('last-update')
};

// Инициализация приложения
document.addEventListener('DOMContentLoaded', function() {
    console.log('AI Trader - Инициализация приложения');

    // Настройка временной зоны
    setLocaleTime();

    // Загрузка начального состояния
    loadInitialState();

    // Настройка навигации
    setupNavigation();

    // Настройка кнопок управления
    setupControlButtons();

    // Запуск периодических обновлений
    startPeriodicUpdates();

    // Загрузка начальной страницы
    loadPage('dashboard');
});

/**
 * Установка локального времени
 */
function setLocaleTime() {
    const now = new Date();
    const options = {
        timeZone: 'Europe/Moscow',
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    };

    DOM.currentTime.textContent = now.toLocaleTimeString('ru-RU', options);
    DOM.lastUpdate.textContent = `Последнее обновление: ${now.toLocaleTimeString('ru-RU', {hour: '2-digit', minute:'2-digit'})}`;

    // Обновление времени каждую секунду
    setInterval(() => {
        const now = new Date();
        DOM.currentTime.textContent = now.toLocaleTimeString('ru-RU', options);
    }, 1000);
}

/**
 * Загрузка начального состояния системы
 */
async function loadInitialState() {
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/api/status`);
        const data = await response.json();

        if (data.success) {
            APP_STATE.systemActive = data.data.system_active;
            APP_STATE.tradingEnabled = data.data.trading_enabled;
            APP_STATE.marketOpen = data.data.market_open;
            APP_STATE.lastUpdate = new Date();

            updateSystemStatus();
            updateMarketStatus();
        } else {
            console.error('Ошибка загрузки состояния:', data.message);
        }
    } catch (error) {
        console.error('Ошибка сети:', error);
        showToast('Ошибка соединения с сервером', 'danger');
    }
}

/**
 * Обновление статуса системы
 */
function updateSystemStatus() {
    if (APP_STATE.systemActive) {
        DOM.systemStatus.textContent = 'Система активна';
        DOM.systemStatus.className = 'badge bg-success me-2';
    } else {
        DOM.systemStatus.textContent = 'Система неактивна';
        DOM.systemStatus.className = 'badge bg-danger me-2';
    }
}

/**
 * Обновление статуса рынка
 */
function updateMarketStatus() {
    if (APP_STATE.marketOpen) {
        DOM.marketStatus.textContent = 'Рынок открыт';
        DOM.marketStatus.className = 'badge bg-success me-2';
    } else {
        DOM.marketStatus.textContent = 'Рынок закрыт';
        DOM.marketStatus.className = 'badge bg-danger me-2';
    }
}

/**
 * Настройка навигации
 */
function setupNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');

    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();

            // Удаляем активный класс у всех ссылок
            navLinks.forEach(l => l.classList.remove('active'));

            // Добавляем активный класс текущей ссылке
            this.classList.add('active');

            // Загружаем контент страницы
            const page = this.getAttribute('href').substring(1);
            loadPage(page);
        });
    });
}

/**
 * Настройка кнопок управления
 */
function setupControlButtons() {
    // Старт торговли
    document.getElementById('start-btn')?.addEventListener('click', async function() {
        if (await confirmAction('Запустить торговлю?')) {
            await startTrading();
        }
    });

    // Пауза торговли
    document.getElementById('pause-btn')?.addEventListener('click', async function() {
        if (await confirmAction('Приостановить торговлю?')) {
            await pauseTrading();
        }
    });

    // Стоп торговли
    document.getElementById('stop-btn')?.addEventListener('click', async function() {
        if (await confirmAction('Остановить торговлю? Все активные операции будут завершены.')) {
            await stopTrading();
        }
    });

    // Обновить данные
    document.getElementById('refresh-btn')?.addEventListener('click', function() {
        refreshData();
    });

    // Сохранить состояние
    document.getElementById('save-state-btn')?.addEventListener('click', async function() {
        await saveSystemState();
    });

    // Ребалансировка
    document.getElementById('rebalance-btn')?.addEventListener('click', async function() {
        if (await confirmAction('Выполнить ребалансировку портфеля?')) {
            await rebalancePortfolio();
        }
    });
}

/**
 * Загрузка страницы
 */
async function loadPage(page) {
    APP_STATE.currentPage = page;

    try {
        // Показываем индикатор загрузки
        showLoading();

        // Загружаем контент страницы
        const response = await fetch(`${CONFIG.API_BASE_URL}/api/pages/${page}`);
        const data = await response.json();

        if (data.success) {
            // Обновляем контент
            DOM.pageContent.innerHTML = data.html;
            DOM.pageTitle.textContent = data.title;
            DOM.pageSubtitle.textContent = data.subtitle;

            // Инициализируем страницу
            initializePage(page);

            // Загружаем данные для страницы
            loadPageData(page);
        } else {
            throw new Error(data.message);
        }
    } catch (error) {
        console.error(`Ошибка загрузки страницы ${page}:`, error);
        DOM.pageContent.innerHTML = `
            <div class="alert alert-danger">
                <h4>Ошибка загрузки страницы</h4>
                <p>${error.message}</p>
                <button class="btn btn-primary mt-3" onclick="loadPage('${page}')">
                    Повторить попытку
                </button>
            </div>
        `;
    } finally {
        hideLoading();
    }
}

/**
 * Инициализация страницы
 */
function initializePage(page) {
    switch(page) {
        case 'dashboard':
            initializeDashboard();
            break;
        case 'portfolio':
            initializePortfolio();
            break;
        case 'charts':
            initializeCharts();
            break;
        case 'news':
            initializeNews();
            break;
        case 'settings':
            initializeSettings();
            break;
        case 'logs':
            initializeLogs();
            break;
    }
}

/**
 * Загрузка данных для страницы
 */
async function loadPageData(page) {
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/api/${page}/data`);
        const data = await response.json();

        if (data.success) {
            updatePageData(page, data.data);
        }
    } catch (error) {
        console.error(`Ошибка загрузки данных для страницы ${page}:`, error);
    }
}

/**
 * Обновление данных страницы
 */
function updatePageData(page, data) {
    switch(page) {
        case 'dashboard':
            updateDashboardData(data);
            break;
        case 'portfolio':
            updatePortfolioData(data);
            break;
        case 'charts':
            updateChartsData(data);
            break;
        case 'news':
            updateNewsData(data);
            break;
    }
}

/**
 * Инициализация дашборда
 */
function initializeDashboard() {
    // Настройка обновления данных дашборда
    if (APP_STATE.updateIntervals.dashboard) {
        clearInterval(APP_STATE.updateIntervals.dashboard);
    }

    APP_STATE.updateIntervals.dashboard = setInterval(() => {
        loadPageData('dashboard');
    }, CONFIG.UPDATE_INTERVAL);

    // Настройка слайдеров
    const speedSlider = document.getElementById('update-speed-slider');
    const riskSlider = document.getElementById('risk-level-slider');

    if (speedSlider) {
        speedSlider.addEventListener('input', function() {
            const speed = this.value;
            updateRefreshSpeed(speed * 1000); // Конвертируем в миллисекунды
        });
    }

    if (riskSlider) {
        riskSlider.addEventListener('input', function() {
            const riskLevel = this.value;
            updateRiskLevel(riskLevel);
        });
    }
}

/**
 * Обновление данных дашборда
 */
function updateDashboardData(data) {
    if (!data) return;

    // Обновление капитала
    if (data.total_capital !== undefined) {
        DOM.totalCapital.textContent = formatCurrency(data.total_capital);
    }

    if (data.cash !== undefined) {
        DOM.cashAmount.textContent = formatCurrency(data.cash);
    }

    if (data.positions_value !== undefined) {
        DOM.positionsValue.textContent = formatCurrency(data.positions_value);
    }

    // Обновление PnL
    if (data.pnl_percent !== undefined) {
        const pnlPercent = data.pnl_percent;
        DOM.pnlPercent.textContent = `${pnlPercent >= 0 ? '+' : ''}${pnlPercent.toFixed(2)}%`;
        DOM.pnlPercent.className = pnlPercent >= 0 ? 'metric-value metric-positive' : 'metric-value metric-negative';
    }

    if (data.daily_pnl !== undefined) {
        const dailyPnl = data.daily_pnl;
        DOM.dailyPnl.textContent = `${dailyPnl >= 0 ? '+' : ''}${formatCurrency(dailyPnl)}`;
        DOM.dailyPnl.className = dailyPnl >= 0 ? 'metric-positive' : 'metric-negative';
    }

    if (data.total_pnl !== undefined) {
        const totalPnl = data.total_pnl;
        DOM.totalPnl.textContent = `${totalPnl >= 0 ? '+' : ''}${formatCurrency(totalPnl)}`;
        DOM.totalPnl.className = totalPnl >= 0 ? 'metric-positive' : 'metric-negative';
    }

    // Обновление таблицы позиций
    if (data.positions && Array.isArray(data.positions)) {
        updatePositionsTable(data.positions);
    }

    // Обновление списка сигналов
    if (data.signals && Array.isArray(data.signals)) {
        updateSignalsList(data.signals);
    }

    // Обновление времени последнего обновления
    DOM.lastUpdate.textContent = `Последнее обновление: ${new Date().toLocaleTimeString('ru-RU', {hour: '2-digit', minute:'2-digit'})}`;
}

/**
 * Обновление таблицы позиций
 */
function updatePositionsTable(positions) {
    const tableContainer = document.getElementById('positions-table');
    if (!tableContainer) return;

    if (!positions || positions.length === 0) {
        tableContainer.innerHTML = '<p class="text-muted text-center">Нет открытых позиций</p>';
        return;
    }

    let html = `
        <table class="table table-sm table-hover">
            <thead>
                <tr>
                    <th>Тикер</th>
                    <th>Кол-во</th>
                    <th>Ср. цена</th>
                    <th>Тек. цена</th>
                    <th>Стоимость</th>
                    <th>PnL</th>
                    <th>Вес</th>
                </tr>
            </thead>
            <tbody>
    `;

    positions.forEach(pos => {
        const pnlClass = pos.pnl >= 0 ? 'price-up' : 'price-down';
        const pnlPercent = pos.pnl_percent || 0;

        html += `
            <tr>
                <td><strong>${pos.ticker}</strong></td>
                <td>${formatNumber(pos.quantity)}</td>
                <td>${formatCurrency(pos.avg_price, 2)}</td>
                <td>${formatCurrency(pos.current_price, 2)}</td>
                <td>${formatCurrency(pos.position_value)}</td>
                <td class="${pnlClass}">${pnlPercent >= 0 ? '+' : ''}${pnlPercent.toFixed(1)}%</td>
                <td>${(pos.weight || 0).toFixed(1)}%</td>
            </tr>
        `;
    });

    html += `
            </tbody>
        </table>
    `;

    tableContainer.innerHTML = html;
}

/**
 * Обновление списка сигналов
 */
function updateSignalsList(signals) {
    const listContainer = document.getElementById('signals-list');
    if (!listContainer) return;

    if (!signals || signals.length === 0) {
        listContainer.innerHTML = '<p class="text-muted text-center">Нет активных сигналов</p>';
        return;
    }

    let html = `
        <table class="table table-sm table-hover">
            <thead>
                <tr>
                    <th>Тикер</th>
                    <th>Сигнал</th>
                    <th>Уверенность</th>
                    <th>Причина</th>
                    <th>Время</th>
                </tr>
            </thead>
            <tbody>
    `;

    signals.slice(0, 10).forEach(signal => {
        const actionClass = signal.action === 'BUY' ? 'trade-buy' :
                          signal.action === 'SELL' ? 'trade-sell' : 'text-warning';

        const confidenceClass = signal.confidence > 0.8 ? 'text-success' :
                              signal.confidence > 0.6 ? 'text-warning' : 'text-muted';

        const timeStr = signal.timestamp ?
            new Date(signal.timestamp).toLocaleTimeString('ru-RU', {hour: '2-digit', minute:'2-digit'}) : '';

        html += `
            <tr>
                <td><strong>${signal.ticker}</strong></td>
                <td class="${actionClass}">${signal.action}</td>
                <td class="${confidenceClass}">${(signal.confidence * 100).toFixed(1)}%</td>
                <td>${signal.reason || 'N/A'}</td>
                <td>${timeStr}</td>
            </tr>
        `;
    });

    html += `
            </tbody>
        </table>
    `;

    listContainer.innerHTML = html;
}

/**
 * Запуск торговли
 */
async function startTrading() {
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/api/trading/start`, {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success) {
            APP_STATE.tradingEnabled = true;
            showToast('Торговля запущена', 'success');
            updateSystemStatus();
        } else {
            showToast(`Ошибка: ${data.message}`, 'danger');
        }
    } catch (error) {
        showToast('Ошибка сети', 'danger');
        console.error('Ошибка запуска торговли:', error);
    }
}

/**
 * Приостановка торговли
 */
async function pauseTrading() {
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/api/trading/pause`, {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success) {
            APP_STATE.tradingEnabled = false;
            showToast('Торговля приостановлена', 'warning');
            updateSystemStatus();
        } else {
            showToast(`Ошибка: ${data.message}`, 'danger');
        }
    } catch (error) {
        showToast('Ошибка сети', 'danger');
        console.error('Ошибка приостановки торговли:', error);
    }
}

/**
 * Остановка торговли
 */
async function stopTrading() {
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/api/trading/stop`, {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success) {
            APP_STATE.tradingEnabled = false;
            showToast('Торговля остановлена', 'danger');
            updateSystemStatus();
        } else {
            showToast(`Ошибка: ${data.message}`, 'danger');
        }
    } catch (error) {
        showToast('Ошибка сети', 'danger');
        console.error('Ошибка остановки торговли:', error);
    }
}

/**
 * Обновление данных
 */
async function refreshData() {
    try {
        showLoading();

        const response = await fetch(`${CONFIG.API_BASE_URL}/api/refresh`);
        const data = await response.json();

        if (data.success) {
            showToast('Данные обновлены', 'success');

            // Перезагружаем текущую страницу
            loadPageData(APP_STATE.currentPage);
        } else {
            showToast(`Ошибка: ${data.message}`, 'danger');
        }
    } catch (error) {
        showToast('Ошибка сети', 'danger');
        console.error('Ошибка обновления данных:', error);
    } finally {
        hideLoading();
    }
}

/**
 * Сохранение состояния системы
 */
async function saveSystemState() {
    try {
        const button = document.getElementById('save-state-btn');
        const originalText = button?.textContent;

        if (button) {
            button.disabled = true;
            button.innerHTML = '<span class="loader"></span> Сохранение...';
        }

        const response = await fetch(`${CONFIG.API_BASE_URL}/api/system/save`, {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success) {
            showToast('Состояние системы сохранено', 'success');

            if (button) {
                button.innerHTML = '<i class="fas fa-check"></i> Сохранено!';
                setTimeout(() => {
                    button.disabled = false;
                    button.textContent = originalText;
                }, 2000);
            }
        } else {
            showToast(`Ошибка: ${data.message}`, 'danger');

            if (button) {
                button.disabled = false;
                button.textContent = originalText;
            }
        }
    } catch (error) {
        showToast('Ошибка сети', 'danger');
        console.error('Ошибка сохранения состояния:', error);

        const button = document.getElementById('save-state-btn');
        if (button) {
            button.disabled = false;
            button.textContent = '💾 Сохранить состояние';
        }
    }
}

/**
 * Ребалансировка портфеля
 */
async function rebalancePortfolio() {
    try {
        const button = document.getElementById('rebalance-btn');
        const originalText = button?.textContent;

        if (button) {
            button.disabled = true;
            button.innerHTML = '<span class="loader"></span> Ребалансировка...';
        }

        const response = await fetch(`${CONFIG.API_BASE_URL}/api/portfolio/rebalance`, {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success) {
            showToast('Портфель ребалансирован', 'success');

            if (button) {
                button.innerHTML = '<i class="fas fa-check"></i> Ребалансировано!';
                setTimeout(() => {
                    button.disabled = false;
                    button.textContent = originalText;
                }, 2000);
            }
        } else {
            showToast(`Ошибка: ${data.message}`, 'danger');

            if (button) {
                button.disabled = false;
                button.textContent = originalText;
            }
        }
    } catch (error) {
        showToast('Ошибка сети', 'danger');
        console.error('Ошибка ребалансировки:', error);

        const button = document.getElementById('rebalance-btn');
        if (button) {
            button.disabled = false;
            button.textContent = '🔄 Ребалансировка';
        }
    }
}

/**
 * Обновление скорости обновления
 */
function updateRefreshSpeed(speed) {
    CONFIG.UPDATE_INTERVAL = speed;

    // Перезапускаем интервалы обновления
    restartUpdateIntervals();

    showToast(`Скорость обновления: ${speed/1000}с`, 'info');
}

/**
 * Обновление уровня риска
 */
async function updateRiskLevel(level) {
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/api/settings/risk`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ level: parseInt(level) })
        });

        const data = await response.json();

        if (data.success) {
            showToast(`Уровень риска обновлен: ${level}/10`, 'success');
        }
    } catch (error) {
        console.error('Ошибка обновления уровня риска:', error);
    }
}

/**
 * Перезапуск интервалов обновления
 */
function restartUpdateIntervals() {
    // Останавливаем все интервалы
    Object.values(APP_STATE.updateIntervals).forEach(interval => {
        clearInterval(interval);
    });

    APP_STATE.updateIntervals = {};

    // Запускаем заново для текущей страницы
    if (APP_STATE.currentPage === 'dashboard') {
        APP_STATE.updateIntervals.dashboard = setInterval(() => {
            loadPageData('dashboard');
        }, CONFIG.UPDATE_INTERVAL);
    }
}

/**
 * Запуск периодических обновлений
 */
function startPeriodicUpdates() {
    // Обновление статуса системы каждую минуту
    APP_STATE.updateIntervals.status = setInterval(() => {
        loadInitialState();
    }, 60000);

    // Обновление времени
    APP_STATE.updateIntervals.time = setInterval(() => {
        setLocaleTime();
    }, 1000);
}

/**
 * Показать уведомление
 */
function showToast(message, type = 'info') {
    // Создаем контейнер для тостов если его нет
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'toast-container';
        document.body.appendChild(container);
    }

    // Создаем тост
    const toast = document.createElement('div');
    toast.className = `toast toast-${type} animate-fade-in`;
    toast.innerHTML = `
        <div class="toast-content">
            <i class="fas fa-${getToastIcon(type)}"></i>
            <span>${message}</span>
        </div>
        <button class="toast-close" onclick="this.parentElement.remove()">
            <i class="fas fa-times"></i>
        </button>
    `;

    container.appendChild(toast);

    // Автоматическое удаление через 5 секунд
    setTimeout(() => {
        if (toast.parentElement) {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(100%)';
            setTimeout(() => {
                if (toast.parentElement) {
                    toast.remove();
                }
            }, 300);
        }
    }, 5000);
}

/**
 * Получить иконку для тоста
 */
function getToastIcon(type) {
    switch(type) {
        case 'success': return 'check-circle';
        case 'danger': return 'exclamation-circle';
        case 'warning': return 'exclamation-triangle';
        case 'info': return 'info-circle';
        default: return 'bell';
    }
}

/**
 * Показать индикатор загрузки
 */
function showLoading() {
    let overlay = document.getElementById('loading-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'loading-overlay';
        overlay.className = 'loading-overlay';
        overlay.innerHTML = `
            <div class="loading-spinner">
                <div class="loader loader-lg"></div>
                <p>Загрузка...</p>
            </div>
        `;
        document.body.appendChild(overlay);
    }
    overlay.style.display = 'flex';
}

/**
 * Скрыть индикатор загрузки
 */
function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
}

/**
 * Подтверждение действия
 */
async function confirmAction(message) {
    return new Promise((resolve) => {
        // Создаем модальное окно подтверждения
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">Подтверждение</h5>
                </div>
                <div class="modal-body">
                    <p>${message}</p>
                </div>
                <div class="modal-footer">
                    <button class="btn btn-secondary" id="confirm-cancel">Отмена</button>
                    <button class="btn btn-danger" id="confirm-ok">Подтвердить</button>
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        // Обработчики кнопок
        document.getElementById('confirm-cancel').addEventListener('click', () => {
            modal.remove();
            resolve(false);
        });

        document.getElementById('confirm-ok').addEventListener('click', () => {
            modal.remove();
            resolve(true);
        });

        // Закрытие по клику вне модального окна
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.remove();
                resolve(false);
            }
        });
    });
}

/**
 * Форматирование валюты
 */
function formatCurrency(value, decimals = 0) {
    if (value === null || value === undefined) return '0 ₽';

    const formatted = new Intl.NumberFormat('ru-RU', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    }).format(value);

    return `${formatted} ₽`;
}

/**
 * Форматирование числа
 */
function formatNumber(value) {
    if (value === null || value === undefined) return '0';

    return new Intl.NumberFormat('ru-RU').format(value);
}

/**
 * Инициализация портфеля (заглушка)
 */
function initializePortfolio() {
    console.log('Инициализация портфеля');
}

/**
 * Инициализация графиков (заглушка)
 */
function initializeCharts() {
    console.log('Инициализация графиков');
}

/**
 * Инициализация новостей (заглушка)
 */
function initializeNews() {
    console.log('Инициализация новостей');
}

/**
 * Инициализация настроек (заглушка)
 */
function initializeSettings() {
    console.log('Инициализация настроек');
}

/**
 * Инициализация логов (заглушка)
 */
function initializeLogs() {
    console.log('Инициализация логов');
}

/**
 * Обновление данных портфеля (заглушка)
 */
function updatePortfolioData(data) {
    console.log('Обновление данных портфеля:', data);
}

/**
 * Обновление данных графиков (заглушка)
 */
function updateChartsData(data) {
    console.log('Обновление данных графиков:', data);
}

/**
 * Обновление данных новостей (заглушка)
 */
function updateNewsData(data) {
    console.log('Обновление данных новостей:', data);
}

// Экспорт функций для глобального доступа
window.AITrader = {
    loadPage,
    startTrading,
    pauseTrading,
    stopTrading,
    refreshData,
    saveSystemState,
    rebalancePortfolio,
    showToast,
    confirmAction
};

// Автоскролл журнала событий
setInterval(function() {
    var logDiv = document.getElementById('logs-content');
    if (logDiv) {
        logDiv.scrollTop = logDiv.scrollHeight;
    }
}, 1000);