/* ═══════════════════════════════════════════════════════════════
   AI Portfolio Optimizer — Frontend Application
   ═══════════════════════════════════════════════════════════════ */

const API = '';
let eventSource = null;
const charts = {};
const STEP_ORDER = ['data_ingestion','preprocessing','feature_engineering','lstm_preparation','lstm_training','predictions','backtesting','packaging','completed'];

// ─── Navigation ──────────────────────────────────────────────
document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', () => {
        document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
        item.classList.add('active');
        document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
        const sec = document.getElementById('section-' + item.dataset.section);
        if (sec) sec.classList.add('active');
    });
});

// ─── Pipeline Control ────────────────────────────────────────
async function startPipeline() {
    const btn = document.getElementById('btn-start');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Running...';
    try {
        await fetch(API + '/api/pipeline/start', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({years: 7}) });
        startSSE();
    } catch(e) { btn.disabled = false; btn.textContent = '🚀 Start Pipeline'; alert('Failed to start: ' + e.message); }
}

async function resetPipeline() {
    await fetch(API + '/api/pipeline/reset', { method: 'POST' });
    document.getElementById('progress-bar').style.width = '0%';
    document.getElementById('progress-pct').textContent = '0%';
    document.getElementById('progress-step').textContent = 'Waiting to start...';
    document.getElementById('status-label').textContent = 'Idle';
    document.getElementById('status-text').className = 'status-dot idle';
    document.querySelectorAll('.step-pill').forEach(p => { p.classList.remove('active','completed','error'); });
    document.getElementById('btn-start').disabled = false;
    document.getElementById('btn-start').innerHTML = '🚀 Start Pipeline';
    if (eventSource) { eventSource.close(); eventSource = null; }
}

// ─── SSE Streaming ───────────────────────────────────────────
function startSSE() {
    if (eventSource) eventSource.close();
    eventSource = new EventSource(API + '/api/pipeline/stream');
    eventSource.onmessage = (e) => {
        const d = JSON.parse(e.data);
        updateProgress(d);
        if (d.status === 'completed') { eventSource.close(); eventSource = null; loadResults(); }
        if (d.status === 'error') { eventSource.close(); eventSource = null; showError(d.error); }
    };
    eventSource.onerror = () => { setTimeout(pollStatus, 2000); };
}

async function pollStatus() {
    try {
        const r = await fetch(API + '/api/pipeline/status');
        const d = await r.json();
        updateProgress(d);
        if (d.status === 'running') setTimeout(pollStatus, 1500);
        else if (d.status === 'completed') loadResults();
        else if (d.status === 'error') showError(d.error);
    } catch(e) { setTimeout(pollStatus, 3000); }
}

function updateProgress(d) {
    document.getElementById('progress-bar').style.width = d.progress + '%';
    document.getElementById('progress-pct').textContent = d.progress + '%';
    document.getElementById('progress-step').textContent = d.details || d.step || '';
    document.getElementById('status-label').textContent = d.status.charAt(0).toUpperCase() + d.status.slice(1);
    document.getElementById('status-text').className = 'status-dot ' + d.status;
    // Update step pills
    const cur = d.step || d.current_step || '';
    const idx = STEP_ORDER.indexOf(cur);
    document.querySelectorAll('.step-pill').forEach(p => {
        const si = STEP_ORDER.indexOf(p.dataset.step);
        p.classList.remove('active','completed','error');
        if (d.status === 'error' && p.dataset.step === cur) p.classList.add('error');
        else if (si < idx || d.status === 'completed') p.classList.add('completed');
        else if (si === idx) p.classList.add('active');
    });
    // Add log entry
    if (d.details) addLogEntry(d.step || '', d.details);
}

function showError(msg) {
    document.getElementById('btn-start').disabled = false;
    document.getElementById('btn-start').innerHTML = '🚀 Retry Pipeline';
    document.getElementById('progress-step').textContent = '❌ Error: ' + msg;
}

function addLogEntry(step, msg) {
    const c = document.getElementById('logs-container');
    const now = new Date().toLocaleTimeString();
    const el = document.createElement('div');
    el.className = 'log-entry';
    el.innerHTML = `<span class="log-time">${now}</span><span class="log-step">${step}</span><span class="log-message">${msg}</span>`;
    c.appendChild(el);
    c.scrollTop = c.scrollHeight;
}

// ─── Load & Render Results ───────────────────────────────────
async function loadResults() {
    document.getElementById('btn-start').disabled = false;
    document.getElementById('btn-start').innerHTML = '🚀 Start Pipeline';
    try {
        const r = await fetch(API + '/api/pipeline/results');
        const data = await r.json();
        // Extract backtest period for display
        const tp = data.backtest.test_period || {};
        const backtestRange = tp.start && tp.end ? `${tp.start} → ${tp.end} (${tp.trading_days} days)` : '';
        renderMetrics(data.backtest.metrics, backtestRange);
        renderEquityCurve(data.backtest.equity_curve, backtestRange);
        renderDrawdown(data.backtest.equity_curve);
        renderRollingReturns(data.backtest.equity_curve);
        renderAllocation(data.backtest.rebalance_logs);
        renderComparison(data.backtest.metrics, backtestRange);
        renderRebalanceLogs(data.backtest.rebalance_logs);
        renderTraining(data.training);
        // Auto-navigate to overview
        document.querySelector('[data-section="overview"]').click();
    } catch(e) { console.error('Failed to load results:', e); }
}

// ─── Render Metrics ──────────────────────────────────────────
function renderMetrics(m, backtestRange) {
    const g = document.getElementById('metrics-grid');
    // Show backtest period banner at top
    const periodBanner = backtestRange
        ? `<div class="metric-card" style="grid-column:1/-1;background:rgba(212,168,67,0.06);border-color:rgba(212,168,67,0.15)">
             <div class="metric-label">Backtest Period (held-out test set)</div>
             <div class="metric-value neutral" style="font-size:16px">${backtestRange}</div>
             <div class="metric-comparison">Both portfolio and Nifty measured over this exact window</div>
           </div>` : '';
    const items = [
        { label: 'Total Return', value: m.portfolio_total_return + '%', cls: m.portfolio_total_return >= 0 ? 'positive' : 'negative', cmp: `NIFTY: ${m.nifty_total_return}%` },
        { label: 'Annual Return', value: m.portfolio_annual_return + '%', cls: m.portfolio_annual_return >= 0 ? 'positive' : 'negative', cmp: `NIFTY: ${m.nifty_annual_return}%` },
        { label: 'Sharpe Ratio', value: m.portfolio_sharpe_ratio, cls: m.portfolio_sharpe_ratio > 0 ? 'positive' : 'negative', cmp: `NIFTY: ${m.nifty_sharpe_ratio}` },
        { label: 'Sortino Ratio', value: m.portfolio_sortino_ratio, cls: m.portfolio_sortino_ratio > 0 ? 'positive' : 'negative', cmp: 'Downside risk adjusted' },
        { label: 'Max Drawdown', value: m.portfolio_max_drawdown + '%', cls: 'negative', cmp: `NIFTY: ${m.nifty_max_drawdown}%` },
        { label: 'Alpha', value: m.alpha + '%', cls: m.alpha >= 0 ? 'positive' : 'negative', cmp: 'vs NIFTY benchmark' },
        { label: 'Beta', value: m.beta, cls: 'neutral', cmp: 'Market sensitivity' },
        { label: 'Info Ratio', value: m.information_ratio, cls: m.information_ratio > 0 ? 'positive' : 'negative', cmp: 'Risk-adjusted excess' },
        { label: 'Win Rate', value: m.portfolio_win_rate + '%', cls: m.portfolio_win_rate > 50 ? 'positive' : 'negative', cmp: 'Profitable days' },
        { label: 'Volatility', value: m.portfolio_annual_volatility + '%', cls: 'neutral', cmp: `NIFTY: ${m.nifty_annual_volatility}%` },
    ];
    g.innerHTML = periodBanner + items.map(i => `
        <div class="metric-card">
            <div class="metric-label">${i.label}</div>
            <div class="metric-value ${i.cls}">${i.value}</div>
            <div class="metric-comparison">${i.cmp}</div>
        </div>`).join('');
}

// ─── Charts ──────────────────────────────────────────────────
const COLORS = { port: '#d4a843', nifty: '#7b9db8', portBg: 'rgba(212,168,67,0.08)', niftyBg: 'rgba(123,157,184,0.06)' };
const chartDefaults = { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: '#a8a29e', font: { family: 'Inter', size: 11 } } } }, scales: { x: { ticks: { color: '#78716c', maxTicksLimit: 12, font: { size: 10, family: 'Inter' } }, grid: { color: 'rgba(255,255,255,0.03)' } }, y: { ticks: { color: '#78716c', font: { size: 10, family: 'Inter' } }, grid: { color: 'rgba(255,255,255,0.03)' } } } };

function destroyChart(id) { if (charts[id]) { charts[id].destroy(); delete charts[id]; } }

function renderEquityCurve(eq, backtestRange) {
    destroyChart('equity');
    // Update chart title with date range
    const titleEl = document.getElementById('equity-chart-title');
    if (titleEl) titleEl.textContent = `📈 Equity Curve — Portfolio vs NIFTY 50 [${backtestRange || ''}]`;
    const labels = eq.dates.filter((_, i) => i % 5 === 0);
    const portData = eq.portfolio.filter((_, i) => i % 5 === 0);
    const niftyData = eq.nifty.filter((_, i) => i % 5 === 0);
    charts.equity = new Chart(document.getElementById('chart-equity'), {
        type: 'line',
        data: { labels, datasets: [
            { label: 'AI Portfolio', data: portData, borderColor: COLORS.port, backgroundColor: COLORS.portBg, fill: true, tension: 0.3, pointRadius: 0, borderWidth: 2.5 },
            { label: 'NIFTY 50 (Buy & Hold)', data: niftyData, borderColor: COLORS.nifty, backgroundColor: COLORS.niftyBg, fill: true, tension: 0.3, pointRadius: 0, borderWidth: 2 },
        ]},
        options: { ...chartDefaults, plugins: { ...chartDefaults.plugins, tooltip: { mode: 'index', intersect: false } } }
    });
}

function renderDrawdown(eq) {
    destroyChart('drawdown');
    const portDD = [], niftyDD = [];
    let pMax = 0, nMax = 0;
    eq.portfolio.forEach((v, i) => { pMax = Math.max(pMax, v); portDD.push(((v - pMax) / pMax) * 100); });
    eq.nifty.forEach((v, i) => { nMax = Math.max(nMax, v); niftyDD.push(((v - nMax) / nMax) * 100); });
    const labels = eq.dates.filter((_, i) => i % 5 === 0);
    charts.drawdown = new Chart(document.getElementById('chart-drawdown'), {
        type: 'line',
        data: { labels, datasets: [
            { label: 'Portfolio DD', data: portDD.filter((_, i) => i % 5 === 0), borderColor: COLORS.port, backgroundColor: 'rgba(99,102,241,0.15)', fill: true, tension: 0.3, pointRadius: 0, borderWidth: 1.5 },
            { label: 'NIFTY DD', data: niftyDD.filter((_, i) => i % 5 === 0), borderColor: COLORS.nifty, backgroundColor: 'rgba(245,158,11,0.1)', fill: true, tension: 0.3, pointRadius: 0, borderWidth: 1.5 },
        ]},
        options: chartDefaults
    });
}

function renderRollingReturns(eq) {
    destroyChart('rolling');
    const window = 60;
    const portRoll = [], niftyRoll = [];
    for (let i = window; i < eq.portfolio.length; i++) {
        portRoll.push(((eq.portfolio[i] / eq.portfolio[i - window]) - 1) * 100);
        niftyRoll.push(((eq.nifty[i] / eq.nifty[i - window]) - 1) * 100);
    }
    const labels = eq.dates.slice(window).filter((_, i) => i % 5 === 0);
    charts.rolling = new Chart(document.getElementById('chart-rolling'), {
        type: 'line',
        data: { labels, datasets: [
            { label: 'Portfolio 60D', data: portRoll.filter((_, i) => i % 5 === 0), borderColor: COLORS.port, tension: 0.3, pointRadius: 0, borderWidth: 1.5 },
            { label: 'NIFTY 60D', data: niftyRoll.filter((_, i) => i % 5 === 0), borderColor: COLORS.nifty, tension: 0.3, pointRadius: 0, borderWidth: 1.5 },
        ]},
        options: chartDefaults
    });
}

function renderAllocation(logs) {
    if (!logs || logs.length === 0) return;
    const last = logs[logs.length - 1];
    destroyChart('allocation');
    const palette = ['#d4a843','#7b9db8','#6b9e7a','#c47060','#b08cce','#c9a54e','#8ba89e','#d49876'];
    charts.allocation = new Chart(document.getElementById('chart-allocation'), {
        type: 'doughnut',
        data: { labels: last.stocks_selected, datasets: [{ data: last.weights.map(w => (w * 100).toFixed(1)), backgroundColor: palette.slice(0, last.stocks_selected.length), borderWidth: 0, hoverOffset: 8 }] },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'right', labels: { color: '#9ca3af', padding: 12, font: { family: 'Inter', size: 12 } } } } }
    });
    destroyChart('returnsWeights');
    charts.returnsWeights = new Chart(document.getElementById('chart-returns-weights'), {
        type: 'bar',
        data: { labels: last.stocks_selected.map(t => t.replace('.NS', '')), datasets: [
            { label: 'Weight %', data: last.weights.map(w => (w * 100).toFixed(1)), backgroundColor: 'rgba(99,102,241,0.6)', borderRadius: 6 },
            { label: 'Pred Return (bps)', data: last.predicted_returns.map(r => (r * 10000).toFixed(1)), backgroundColor: 'rgba(16,185,129,0.6)', borderRadius: 6 },
        ]},
        options: { ...chartDefaults, plugins: { ...chartDefaults.plugins, legend: { labels: { color: '#9ca3af' } } } }
    });
}

function renderComparison(m, backtestRange) {
    const rows = [
        ['Total Return', m.portfolio_total_return + '%', m.nifty_total_return + '%'],
        ['Annual Return', m.portfolio_annual_return + '%', m.nifty_annual_return + '%'],
        ['Sharpe Ratio', m.portfolio_sharpe_ratio, m.nifty_sharpe_ratio],
        ['Volatility', m.portfolio_annual_volatility + '%', m.nifty_annual_volatility + '%'],
        ['Max Drawdown', m.portfolio_max_drawdown + '%', m.nifty_max_drawdown + '%'],
        ['Alpha (vs NIFTY)', m.alpha + '%', '—'],
        ['Beta', m.beta, '1.000'],
        ['Information Ratio', m.information_ratio, '—'],
        ['Excess Return', m.excess_return + '%', '—'],
        ['Backtest Period', backtestRange || '—', backtestRange || '—'],
    ];
    const c = document.getElementById('comparison-content');
    let html = `<div class="comparison-row header"><div class="comparison-cell">Metric</div><div class="comparison-cell">AI Portfolio</div><div class="comparison-cell">NIFTY 50</div><div class="comparison-cell">Winner</div></div>`;
    rows.forEach(([label, port, nifty]) => {
        const pv = parseFloat(port), nv = parseFloat(nifty);
        let winner = '—';
        if (!isNaN(pv) && !isNaN(nv)) {
            if (label.includes('Drawdown') || label.includes('Volatility')) winner = pv > nv ? '🏆 AI Portfolio' : (pv < nv ? '⚡ NIFTY' : 'Tie');
            else winner = pv > nv ? '🏆 AI Portfolio' : (pv < nv ? '⚡ NIFTY' : 'Tie');
        }
        if (label === 'Backtest Period') winner = 'Same window';
        html += `<div class="comparison-row"><div class="comparison-cell label">${label}</div><div class="comparison-cell highlight">${port}</div><div class="comparison-cell dim">${nifty}</div><div class="comparison-cell">${winner}</div></div>`;
    });
    c.innerHTML = html;
}

function renderRebalanceLogs(logs) {
    const el = document.getElementById('rebalance-table');
    document.getElementById('rebalance-count').textContent = logs.length + ' rebalances';
    if (!logs.length) return;
    let html = '<table><thead><tr><th>#</th><th>Date</th><th>Stocks Selected</th><th>Weights</th><th>Added</th><th>Removed</th><th>Turnover</th><th>Exp. Sharpe</th></tr></thead><tbody>';
    logs.forEach(l => {
        const stocks = l.stocks_selected.map(t => `<span class="ticker-badge">${t.replace('.NS','')}</span>`).join(' ');
        const weights = l.weights.map(w => (w * 100).toFixed(0) + '%').join(', ');
        const added = (l.stocks_added || []).map(t => t.replace('.NS', '')).join(', ') || '—';
        const removed = (l.stocks_removed || []).map(t => t.replace('.NS', '')).join(', ') || '—';
        html += `<tr><td>${l.rebalance_number}</td><td>${l.date}</td><td>${stocks}</td><td style="font-family:var(--font-mono);font-size:11px">${weights}</td><td style="color:var(--success)">${added}</td><td style="color:var(--danger)">${removed}</td><td>${l.turnover}</td><td>${l.expected_sharpe}</td></tr>`;
    });
    html += '</tbody></table>';
    el.innerHTML = html;
}

function renderTraining(t) {
    destroyChart('trainingLoss');
    destroyChart('overfitGap');
    document.getElementById('training-epochs').textContent = t.epochs_trained + ' epochs';
    const labels = Array.from({ length: t.epochs_trained }, (_, i) => i + 1);

    // Loss chart
    charts.trainingLoss = new Chart(document.getElementById('chart-training-loss'), {
        type: 'line',
        data: { labels, datasets: [
            { label: 'Train Loss', data: t.train_loss, borderColor: '#6366f1', tension: 0.3, pointRadius: 0, borderWidth: 2.5, fill: false },
            { label: 'Val Loss',   data: t.val_loss,   borderColor: '#f59e0b', tension: 0.3, pointRadius: 0, borderWidth: 2.5, fill: false },
        ]},
        options: { ...chartDefaults,
            plugins: { ...chartDefaults.plugins,
                annotation: {},
                tooltip: { mode: 'index', intersect: false },
                legend: { labels: { color: '#9ca3af' } }
            }
        }
    });

    // Overfitting gap chart
    if (t.overfit_gap && t.overfit_gap.length > 0) {
        const gapCanvas = document.getElementById('chart-overfit-gap');
        if (gapCanvas) {
            const gapColors = t.overfit_gap.map(g => g > 50 ? 'rgba(196,112,96,0.7)' : g > 20 ? 'rgba(201,165,78,0.7)' : 'rgba(107,158,122,0.6)');
            charts.overfitGap = new Chart(gapCanvas, {
                type: 'bar',
                data: { labels, datasets: [{
                    label: 'Overfit Gap % (val-train)/train',
                    data: t.overfit_gap,
                    backgroundColor: gapColors,
                    borderRadius: 2,
                }]},
                options: { ...chartDefaults,
                    plugins: { ...chartDefaults.plugins,
                        legend: { labels: { color: '#9ca3af' } },
                        tooltip: { callbacks: { label: ctx => `Gap: ${ctx.raw > 50 ? '⚠️ OVERFIT ' : ctx.raw > 20 ? '⚡ WATCH ' : '✅ OK '}${ctx.raw.toFixed(1)}%` } }
                    },
                    scales: { ...chartDefaults.scales,
                        y: { ...chartDefaults.scales.y, grid: { color: 'rgba(99,102,241,0.06)' },
                            ticks: { color: '#6b7280', callback: v => v + '%' } }
                    }
                }
            });
        }
    }

    const gapPct = t.final_overfit_gap_pct || 0;
    const gapStatus = gapPct > 50 ? '⚠️ OVERFIT' : gapPct > 20 ? '⚡ WATCH' : '✅ Healthy';
    const splits = t.split_dates || {};
    const tm = document.getElementById('training-metrics');
    tm.innerHTML = [
        { l: 'Epochs Trained',   v: t.epochs_trained,                    c: 'neutral' },
        { l: 'Best Val Loss',    v: t.best_val_loss.toFixed(6),           c: 'neutral' },
        { l: 'Final Overfit Gap', v: `${gapPct.toFixed(1)}% ${gapStatus}`, c: gapPct > 50 ? 'negative' : gapPct > 20 ? 'neutral' : 'positive' },
        { l: 'Train Samples',    v: (t.train_samples || 0).toLocaleString(), c: 'neutral' },
        { l: 'Val Samples',      v: (t.val_samples  || 0).toLocaleString(), c: 'neutral' },
        { l: 'Test Samples',     v: (t.test_samples || 0).toLocaleString(), c: 'neutral' },
        { l: 'Val Start Date',   v: splits.val_start  || '—',             c: 'neutral' },
        { l: 'Test Start Date',  v: splits.test_start || '—',             c: 'neutral' },
    ].map(i => `<div class="metric-card"><div class="metric-label">${i.l}</div><div class="metric-value ${i.c}" style="font-size:18px">${i.v}</div></div>`).join('');
}

// ─── Initial Status Check ────────────────────────────────────
(async () => {
    try {
        const r = await fetch(API + '/api/pipeline/status');
        const d = await r.json();
        if (d.status === 'running') { updateProgress(d); startSSE(); }
        else if (d.status === 'completed') { updateProgress(d); loadResults(); }
    } catch(e) { /* server not ready */ }
})();
