/* ── State ──────────────────────────────────────────────────────────────── */
let activeTicker  = null;
let activeItvl    = '1m';
let chart         = null;
let cSeries       = null;   // candlestick
let vSeries       = null;   // volume
let vwapSeries    = null;   // VWAP line
let indData       = {};     // latest indicator payload from /api/indicators
let activeInds    = new Set(); // all indicators off by default
let arTimer       = null;
let arCountdown   = 5;
let arEnabled     = true;

// RAF-loop state for canvas redraw detection
let _rafTestTime  = 0;
let _rafTestPrice = 0;
let _prevX        = null;
let _prevY        = null;
let _vrDebounce   = null;
// Track candle count so we can shift the logical range on refresh
let _candleCount  = 0;
let _stateSaveDebounce = null;
let _chartLoadSeq = 0;
let _signalLoadSeq = 0;
let _mtfLoadSeq = 0;

const UI_STATE_KEY = 'makeshifttrades.chartState';

const PERIOD = {
  '1m':'5d', '3m':'5d', '5m':'5d', '15m':'1mo',
};

// Expected seconds between consecutive candles for each interval.
// Used to detect when the backend returns a different granularity than requested.
const ITVL_SECS = { '1m': 60, '3m': 180, '5m': 300, '15m': 900 };

function getStoredState() {
  try {
    const raw = window.localStorage.getItem(UI_STATE_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch (err) {
    console.warn('Unable to read chart state:', err);
    return {};
  }
}

function saveStoredState(nextState) {
  try {
    window.localStorage.setItem(UI_STATE_KEY, JSON.stringify(nextState));
  } catch (err) {
    console.warn('Unable to save chart state:', err);
  }
}

function persistUIState(overrides = {}) {
  const current = getStoredState();
  saveStoredState({
    ...current,
    ticker: activeTicker,
    interval: activeItvl,
    arEnabled,
    ...overrides,
  });
}

function clearPersistedView() {
  persistUIState({ view: null });
}

function persistViewState(candleCount = _candleCount) {
  if (!chart || !activeTicker || !activeItvl || candleCount <= 0) return;
  const logicalRange = chart.timeScale().getVisibleLogicalRange();
  if (!logicalRange) return;

  persistUIState({
    view: {
      ticker:    activeTicker,
      interval:  activeItvl,
      period:    PERIOD[activeItvl] || '5d',
      candleCount,
      logicalRange: {
        from: logicalRange.from,
        to:   logicalRange.to,
      },
    },
  });
}

function queuePersistViewState() {
  clearTimeout(_stateSaveDebounce);
  _stateSaveDebounce = setTimeout(() => persistViewState(), 120);
}

function getRestorableViewState(ticker = activeTicker, interval = activeItvl) {
  const state = getStoredState();
  const view = state.view;
  if (!view) return null;
  if (view.ticker !== ticker || view.interval !== interval) return null;
  // Reject stale views saved under a different data period (e.g. 1d → 5d change).
  const currentPeriod = PERIOD[interval] || '5d';
  if (view.period && view.period !== currentPeriod) return null;
  if (!view.logicalRange ||
      typeof view.logicalRange.from !== 'number' ||
      typeof view.logicalRange.to   !== 'number') {
    return null;
  }
  return view;
}

function syncTickerButtons() {
  document.querySelectorAll('.ticker-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.ticker === activeTicker);
  });
}

function syncIntervalButtons() {
  document.querySelectorAll('.itvl-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.itvl === activeItvl);
  });
}

function applyPersistedUIState() {
  const state = getStoredState();
  if (state.interval && PERIOD[state.interval]) {
    activeItvl = state.interval;
  }
  if (typeof state.arEnabled === 'boolean') {
    arEnabled = state.arEnabled;
  }
  if (state.ticker) {
    activeTicker = state.ticker;
  }

  syncIntervalButtons();
  syncTickerButtons();
}


/* ── Chart init ─────────────────────────────────────────────────────────── */
function initChart() {
  const wrap = document.getElementById('chart-wrap');

  chart = LightweightCharts.createChart(wrap, {
    layout: { background: { color: '#131722' }, textColor: '#94a3b8' },
    grid:   { vertLines: { color: '#1e293b' }, horzLines: { color: '#1e293b' } },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: { borderColor: '#2a3045' },
    timeScale: { borderColor: '#2a3045', timeVisible: true, secondsVisible: false },
    width:  wrap.clientWidth,
    height: wrap.clientHeight,
  });

  cSeries = chart.addCandlestickSeries({
    upColor: '#089981', downColor: '#f23645',
    borderUpColor: '#089981', borderDownColor: '#f23645',
    wickUpColor: '#089981', wickDownColor: '#f23645',
  });

  vSeries = chart.addHistogramSeries({
    color: '#26a69a30', priceFormat: { type: 'volume' },
    priceScaleId: '', scaleMargins: { top: 0.82, bottom: 0 },
  });

  // Canvas overlay
  const canvas = document.getElementById('overlay-canvas');
  canvas.width  = wrap.clientWidth;
  canvas.height = wrap.clientHeight;

  // Resize everything when the container resizes
  new ResizeObserver(() => {
    chart.applyOptions({ width: wrap.clientWidth, height: wrap.clientHeight });
    canvas.width  = wrap.clientWidth;
    canvas.height = wrap.clientHeight;
    drawCanvas();
  }).observe(wrap);

  // VWAP line series
  vwapSeries = chart.addLineSeries({
    color: '#ff6b6b', lineWidth: 1,
    lastValueVisible: false, priceLineVisible: false,
    crosshairMarkerVisible: false, title: 'VWAP',
  });

  // Re-clip all zoom-aware indicators whenever the visible range changes
  chart.timeScale().subscribeVisibleTimeRangeChange(() => {
    clearTimeout(_vrDebounce);
    _vrDebounce = setTimeout(() => { applyMarkers(); drawCanvas(); }, 30);
  });

  chart.timeScale().subscribeVisibleLogicalRangeChange(() => {
    queuePersistViewState();
  });

  // RAF loop detects Y price-scale drag (not covered by subscribeVisibleTimeRangeChange)
  requestAnimationFrame(rafLoop);
}

function rafLoop() {
  if (chart && cSeries && _rafTestTime > 0) {
    const x = chart.timeScale().timeToCoordinate(_rafTestTime);
    const y = cSeries.priceToCoordinate(_rafTestPrice);
    if (x !== _prevX || y !== _prevY) {
      _prevX = x; _prevY = y;
      drawCanvas();
    }
  }
  requestAnimationFrame(rafLoop);
}


/* ── Canvas overlay (FVG + OB rectangles) ───────────────────────────────── */
/* ── Canvas overlay ──────────────────────────────────────────────────────────────── */
function drawCanvas() {
  const canvas = document.getElementById('overlay-canvas');
  if (!canvas || !chart || !cSeries) return;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const vr = chart.timeScale().getVisibleRange();
  if (!vr) return;

  // ── Session Kill Zone bands (painted first, behind everything else) ───
  if (activeInds.has('sessions') && indData.sessions) {
    for (const s of indData.sessions) {
      if (s.end_time < vr.from || s.start_time > vr.to) continue;
      let x0 = chart.timeScale().timeToCoordinate(s.start_time) ?? 0;
      let x1 = chart.timeScale().timeToCoordinate(s.end_time)   ?? canvas.width;
      x0 = Math.max(0, x0); x1 = Math.min(canvas.width, x1);
      const W = Math.abs(x1 - x0);
      if (W < 1) continue;
      ctx.fillStyle = s.color;
      ctx.fillRect(Math.min(x0, x1), 0, W, canvas.height);
      ctx.fillStyle = 'rgba(255,255,255,0.35)';
      ctx.font = 'bold 10px sans-serif';
      ctx.fillText(s.name, Math.min(x0, x1) + 4, 14);
    }
  }

  // ── Equilibrium / Premium-Discount bands ───────────────────────────
  if (activeInds.has('equilibrium') && indData.equilibrium) {
    const eq = indData.equilibrium;
    const yH = cSeries.priceToCoordinate(eq.high);
    const yP = cSeries.priceToCoordinate(eq.premium);
    const yD = cSeries.priceToCoordinate(eq.discount);
    const yL = cSeries.priceToCoordinate(eq.low);
    if (yP !== null && yH !== null) {
      ctx.fillStyle = 'rgba(242,54,69,0.05)';
      ctx.fillRect(0, Math.min(yH, yP), canvas.width, Math.abs(yP - yH));
    }
    if (yD !== null && yL !== null) {
      ctx.fillStyle = 'rgba(8,153,129,0.05)';
      ctx.fillRect(0, Math.min(yL, yD), canvas.width, Math.abs(yD - yL));
    }
    const drawEqLine = (price, label, color, dash) => {
      const y = cSeries.priceToCoordinate(price);
      if (y === null) return;
      ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 1;
      ctx.setLineDash(dash);
      ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = color; ctx.font = '10px sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(label, canvas.width - 5, y - 3);
      ctx.textAlign = 'left';
    };
    drawEqLine(eq.high,     'Range H',   'rgba(242,54,69,0.7)',   [4, 4]);
    drawEqLine(eq.low,      'Range L',   'rgba(8,153,129,0.7)',   [4, 4]);
    drawEqLine(eq.eq,       'EQ 50%',    'rgba(148,163,184,0.8)', [2, 2]);
    drawEqLine(eq.premium,  'Prem 75%',  'rgba(242,54,69,0.5)',   [8, 4]);
    drawEqLine(eq.discount, 'Disc 25%',  'rgba(8,153,129,0.5)',   [8, 4]);
  }

  // ── FVG zones (only zones starting in, or active zones extending into, viewport) ─
  if (activeInds.has('fvg') && indData.fvg) {
    for (const z of indData.fvg) {
      if (z.end_time   < vr.from) continue;              // ended before viewport
      if (z.start_time > vr.to)   continue;              // starts after viewport
      if (z.start_time < vr.from && z.ifvg) continue;   // old mitigated zone, skip
      const bull = z.type === 'bullish';
      const fill = bull
        ? (z.ifvg ? 'rgba(100,149,237,.08)' : 'rgba(100,149,237,.22)')
        : (z.ifvg ? 'rgba(255,165,0,.08)'   : 'rgba(255,165,0,.22)');
      drawRect(ctx, z.start_time, z.end_time, z.top, z.bottom,
               fill, bull ? '#6495ed' : '#ffa500', 0.5, z.ifvg ? [4,3] : [], vr, canvas);
    }
  }

  // ── Order Blocks ─────────────────────────────────────────────────────
  if (activeInds.has('ob') && indData.ob) {
    for (const ob of indData.ob) {
      if (ob.end_time < vr.from || ob.start_time > vr.to) continue;
      const bull = ob.type === 'bullish';
      drawRect(ctx, ob.start_time, ob.end_time, ob.top, ob.bottom,
               bull ? 'rgba(8,153,129,.32)' : 'rgba(242,54,69,.32)',
               bull ? '#089981' : '#f23645', 1, [], vr, canvas);
    }
  }

  // ── Liquidity (canvas-drawn; line starts at pivot time, extends right) ────
  if (activeInds.has('liquidity') && indData.liquidity) {
    for (const lv of indData.liquidity) {
      if (lv.start_time > vr.to) continue;       // pivot in the future
      const y = cSeries.priceToCoordinate(lv.price);
      if (y === null) continue;
      let x0 = chart.timeScale().timeToCoordinate(lv.start_time);
      if (x0 === null) x0 = 0;
      x0 = Math.max(0, x0);
      const col = lv.dir === 'high' ? '#ffb300' : '#ab47bc';
      ctx.beginPath(); ctx.strokeStyle = col; ctx.lineWidth = 1;
      ctx.setLineDash([4, 3]);
      ctx.moveTo(x0, y); ctx.lineTo(canvas.width, y);
      ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = col; ctx.font = '10px sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(lv.dir === 'high' ? '↑ Sell' : '↓ Buy', canvas.width - 4, y - 3);
      ctx.textAlign = 'left';
    }
  }

  // ── Key Levels  PDH / PDL / PDC / PWH / PWL ─────────────────────────────
  if (activeInds.has('key_levels') && indData.key_levels) {
    for (const lv of indData.key_levels) {
      const y = cSeries.priceToCoordinate(lv.price);
      if (y === null) continue;
      const col = (lv.type === 'pdh' || lv.type === 'pwh') ? '#f23645'
                : (lv.type === 'pdl' || lv.type === 'pwl') ? '#089981'
                : '#94a3b8';
      ctx.beginPath(); ctx.strokeStyle = col; ctx.lineWidth = 1;
      ctx.setLineDash([6, 3]);
      ctx.moveTo(0, y); ctx.lineTo(canvas.width, y);
      ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = col; ctx.font = 'bold 10px sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(lv.label, canvas.width - 5, y - 3);
      ctx.textAlign = 'left';
    }
  }
}

function drawRect(ctx, t0, t1, pTop, pBot, fill, stroke, lw, dash, vr, canvas) {
  if (t1 < vr.from || t0 > vr.to) return;
  const mid = (vr.from + vr.to) / 2;
  let x0 = chart.timeScale().timeToCoordinate(t0);
  let x1 = chart.timeScale().timeToCoordinate(t1);
  if (x0 === null) x0 = t0 < mid ? 0 : canvas.width;
  if (x1 === null) x1 = t1 < mid ? 0 : canvas.width;
  let y0 = cSeries.priceToCoordinate(pTop);
  let y1 = cSeries.priceToCoordinate(pBot);
  if (y0 === null) y0 = 0;
  if (y1 === null) y1 = canvas.height;
  const L = Math.min(x0,x1), T = Math.min(y0,y1);
  const W = Math.abs(x1-x0),  H = Math.abs(y1-y0);
  if (W < 0.5) return;
  ctx.fillStyle = fill;
  ctx.fillRect(L, T, W, H);
  ctx.strokeStyle = stroke; ctx.lineWidth = lw;
  if (dash.length) ctx.setLineDash(dash);
  ctx.strokeRect(L, T, W, H);
  ctx.setLineDash([]);
}


/* ── Price lines (Liquidity) ────────────────────────────────────────────── */
/* ── Markers (Engulfing + Swings + MS)  —  filtered to visible range ───────── */
function applyMarkers() {
  if (!cSeries) return;
  const vr   = chart.timeScale().getVisibleRange();
  const from = vr ? vr.from : 0;
  const to   = vr ? vr.to   : Infinity;
  const m    = [];

  if (activeInds.has('engulfing') && indData.engulfing) {
    for (const e of indData.engulfing) {
      if (e.time < from || e.time > to) continue;
      m.push({ time: e.time,
        position: e.type === 'bullish' ? 'belowBar' : 'aboveBar',
        color:    e.type === 'bullish' ? '#00bcd4'  : '#e040fb',
        shape:    e.type === 'bullish' ? 'arrowUp'  : 'arrowDown', size: 1 });
    }
  }

  if (activeInds.has('swings') && indData.swings) {
    for (const s of indData.swings) {
      if (s.time < from || s.time > to) continue;
      m.push({ time: s.time,
        position: s.type === 'high' ? 'aboveBar'   : 'belowBar',
        color:    s.type === 'high' ? '#f23645'     : '#089981',
        shape:    s.type === 'high' ? 'arrowDown'   : 'arrowUp',
        text:     s.type === 'high' ? 'SH' : 'SL',  size: 0.8 });
    }
  }

  if (activeInds.has('ms') && indData.ms) {
    for (const ev of indData.ms) {
      if (ev.time < from || ev.time > to) continue;
      m.push({ time: ev.time, position: 'belowBar',
        color: ev.color, shape: 'circle', text: ev.label, size: 0.5 });
    }
  }

  m.sort((a, b) => a.time - b.time);
  cSeries.setMarkers(m);
}

function applyAll() { applyMarkers(); drawCanvas(); }


/* ── Data loading ───────────────────────────────────────────────────────── */
async function loadChart(preserveView = false) {
  if (!activeTicker) return;
  const requestId = ++_chartLoadSeq;
  const requestedTicker = activeTicker;
  const requestedItvl = activeItvl;
  const period = PERIOD[requestedItvl] || '5d';
  const cover  = document.getElementById('loading-cover');

  if (!preserveView) {
    cover.classList.add('show');
    document.getElementById('chart-title').textContent = requestedTicker + ' — loading…';
  }

  // Capture the visible LOGICAL RANGE (bar indices) before any async work.
  // We only apply the restored range ONCE, after ALL setData() calls are done,
  // so LightweightCharts cannot override it between intermediate data updates.
  let savedLogicalRange = null;
  const prevCandleCount = _candleCount;
  if (preserveView && chart && prevCandleCount > 0) {
    savedLogicalRange = chart.timeScale().getVisibleLogicalRange();
  }

  try {
    // ── Candles ──
    const cr = await fetch(`/api/candles?ticker=${requestedTicker}&interval=${requestedItvl}&period=${period}`);
    const cd = await cr.json();
    if (requestId !== _chartLoadSeq || requestedTicker !== activeTicker || requestedItvl !== activeItvl) return;
    if (!cd.ok) { showErr(cd.error); return; }
    const candles = cd.candles;
    if (!candles || !candles.length) { showErr('No data for this ticker / interval.'); return; }

    // Guard: verify the returned candles actually match the requested interval.
    // yfinance can silently return aggregated (e.g. 5m) data when 1m+5d is
    // unavailable.  Detect this by checking the first candle gap and reject
    // before it pollutes the chart.
    if (candles.length >= 2) {
      const actualSecs   = candles[1].time - candles[0].time;
      const expectedSecs = ITVL_SECS[requestedItvl];
      if (expectedSecs && actualSecs > expectedSecs * 2) {
        console.warn(
          `[chart] Interval mismatch for ${requestedTicker}: ` +
          `requested ${requestedItvl} but first candle gap is ${actualSecs}s (~${actualSecs/60}m). Discarding.`
        );
        // On auto-refresh silently skip; on a manual load surface the error.
        if (!preserveView) showErr(`Data mismatch — server returned ${actualSecs/60}m candles for ${requestedItvl}.`);
        return;
      }
    }

    const candleDelta = candles.length - prevCandleCount;

    // Safety cap: if the candle count changed dramatically (e.g. because the
    // previous fetch returned a different interval), don't try to offset the
    // viewport — fall through to the localStorage-restore path instead.
    if (preserveView && prevCandleCount > 0 && Math.abs(candleDelta) > prevCandleCount * 0.5) {
      savedLogicalRange = null;
    }

    cSeries.setData(candles.map(c => ({ time: c.time, open: c.open, high: c.high, low: c.low, close: c.close })));
    vSeries.setData(candles.map(c => ({
      time: c.time, value: c.volume,
      color: c.close >= c.open ? '#26a69a30' : '#ef444430',
    })));

    _rafTestTime  = candles[candles.length - 1].time;
    _rafTestPrice = candles[candles.length - 1].close;
    _candleCount  = candles.length;

    // Lock viewport immediately so the chart doesn't snap to auto-fit while
    // the indicator fetch is in-flight (the final restore happens below).
    if (preserveView && savedLogicalRange !== null) {
      chart.timeScale().setVisibleLogicalRange({
        from: savedLogicalRange.from + candleDelta,
        to:   savedLogicalRange.to   + candleDelta,
      });
    }

    // Title bar
    const last = candles[candles.length - 1];
    const prev = candles.length > 1 ? candles[candles.length - 2] : last;
    const chg  = ((last.close - prev.close) / prev.close * 100).toFixed(2);
    const col  = chg >= 0 ? '#089981' : '#f23645';
    const arr  = chg >= 0 ? '▲' : '▼';
    document.getElementById('chart-title').innerHTML =
      `<span>${activeTicker}</span>` +
      `<span style="font-size:13px;font-weight:400;color:#94a3b8;margin-left:6px">$${last.close.toFixed(2)}</span>` +
      `<span style="font-size:12px;color:${col};margin-left:4px">${arr} ${Math.abs(chg)}%</span>`;
    document.getElementById('status-bar').textContent =
      `${candles.length} candles · Vol: ${last.volume.toLocaleString()} · ${requestedTicker} ${requestedItvl}`;

    // ── Indicators ──
    const indStr = [...activeInds].join(',');
    const ir = await fetch(`/api/indicators?ticker=${requestedTicker}&interval=${requestedItvl}&period=${period}&indicators=${indStr}`);
    const id = await ir.json();
    if (requestId !== _chartLoadSeq || requestedTicker !== activeTicker || requestedItvl !== activeItvl) return;
    if (id.ok) {
      indData = id;
      if (vwapSeries) {
        vwapSeries.setData(id.vwap && activeInds.has('vwap') ? id.vwap : []);
        vwapSeries.applyOptions({ visible: activeInds.has('vwap') });
      }
      applyAll();
    }

    // ── Restore / initialise viewport ─────────────────────────────────────
    // All setData() calls are done — now apply the final range exactly once.
    let targetFrom, targetTo;
    if (preserveView && savedLogicalRange !== null) {
      // Shift bar indices by however many new candles were appended.
      targetFrom = savedLogicalRange.from + candleDelta;
      targetTo   = savedLogicalRange.to   + candleDelta;
    } else {
      const restoredView = getRestorableViewState(requestedTicker, requestedItvl);
      if (restoredView) {
        const viewDelta = candles.length - (restoredView.candleCount || candles.length);
        targetFrom = restoredView.logicalRange.from + viewDelta;
        targetTo   = restoredView.logicalRange.to   + viewDelta;
      } else {
        targetFrom = Math.max(0, candles.length - 150);
        targetTo   = candles.length;
      }
    }
    chart.timeScale().setVisibleLogicalRange({ from: targetFrom, to: targetTo });

    persistViewState(candles.length);
    persistUIState();

    loadSignal();         // refresh AI + MTF panels (MTF fired inside loadSignal)

    // One deferred frame to catch the render LWC schedules after setData().
    requestAnimationFrame(() => {
      if (requestId !== _chartLoadSeq || !chart) return;
      chart.timeScale().setVisibleLogicalRange({ from: targetFrom, to: targetTo });
    });

  } catch (err) {
    if (requestId !== _chartLoadSeq) return;
    showErr('Network error — check console.');
    console.error(err);
  } finally {
    if (requestId === _chartLoadSeq) {
      cover.classList.remove('show');
    }
  }
}

function showErr(msg) {
  document.getElementById('chart-title').textContent = '⚠ ' + msg;
  document.getElementById('status-bar').textContent  = msg;
  document.getElementById('loading-cover').classList.remove('show');
}


/* ── UI handlers ────────────────────────────────────────────────────────── */
function resetPriceScale() {
  // Re-enables auto-fit after a manual Y-axis drag, so the new ticker's
  // candles are always visible at a sensible price range.
  if (chart) {
    chart.priceScale('right').applyOptions({ autoScale: true });
  }
}

function selectTicker(t) {
  activeTicker = t;
  syncTickerButtons();
  resetPriceScale();
  _candleCount = 0;
  _lastAutoTradeKey = null; // new ticker — allow fresh auto-trade
  clearPersistedView();
  indData = {}; if (cSeries) cSeries.setMarkers([]);
  if (vwapSeries) vwapSeries.setData([]);
  // Reset paper trading panel for new ticker
  _lastSignal = null;
  setPtMsg('');
  refreshExecuteButton();
  refreshPaperStatus();
  loadChart(false); startAR();
}

function setItvl(i) {
  activeItvl = i;
  syncIntervalButtons();
  clearPersistedView();
  if (activeTicker) {
    resetPriceScale();
    _candleCount = 0;
    _lastAutoTradeKey = null; // new interval — allow fresh auto-trade
    indData = {}; if (cSeries) cSeries.setMarkers([]);
    if (vwapSeries) vwapSeries.setData([]);
    loadChart(false); startAR();
  }
}

function toggleInd(name) {
  const btn = document.getElementById('ind-' + name);
  if (activeInds.has(name)) {
    activeInds.delete(name); btn.classList.remove('on');
    if (name === 'vwap' && vwapSeries) vwapSeries.applyOptions({ visible: false });
  } else {
    activeInds.add(name); btn.classList.add('on');
    if (name === 'vwap' && vwapSeries && indData.vwap) {
      vwapSeries.setData(indData.vwap);
      vwapSeries.applyOptions({ visible: true });
    }
  }
  applyAll();
}

function manualRefresh() {
  if (!activeTicker) return;
  loadChart(true);
  if (arEnabled) { arCountdown = 5; updateARLabel(); }
}


/* ── Auto-refresh ───────────────────────────────────────────────────────── */
function startAR() {
  clearInterval(arTimer);
  if (!arEnabled) { updateARLabel(); return; }
  arCountdown = 5; updateARLabel();
  arTimer = setInterval(() => {
    arCountdown--;
    if (arCountdown <= 0) { arCountdown = 5; loadChart(true); }
    updateARLabel();
  }, 1000);
}

function updateARLabel() {
  document.getElementById('ar-lbl').textContent =
    arEnabled ? `Next: ${arCountdown}s` : 'Auto-refresh off';
}

function toggleAR() {
  arEnabled = !arEnabled;
  persistUIState();
  const label = arEnabled ? 'Auto: On' : 'Auto: Off';
  for (const id of ['ar-btn', 'hdr-ar']) {
    const el = document.getElementById(id);
    if (!el) continue;
    el.textContent = label;
    el.classList.toggle('active', arEnabled);
  }
  if (arEnabled) { if (activeTicker) loadChart(true); startAR(); }
  else           { clearInterval(arTimer); updateARLabel(); }
}
/* ── MTF Consensus panel ───────────────────────────────────────────── */
async function loadMTFSignal() {
  if (!activeTicker) return;
  const requestId = ++_mtfLoadSeq;
  const requestedTicker = activeTicker;
  const lbl = document.getElementById('mtf-loading');
  if (lbl) lbl.textContent = 'loading…';

  try {
    const r = await fetch(`/api/mtf_signal?ticker=${requestedTicker}&interval=${activeItvl}`);
    const d = await r.json();
    if (requestId !== _mtfLoadSeq || requestedTicker !== activeTicker) return;
    if (lbl) lbl.textContent = '';

    if (!d.ok) {
      const badge = document.getElementById('mtf-badge');
      badge.className   = 'mtf-badge mtf-neutral';
      badge.textContent = '!';
      return;
    }

    // ── Consensus badge ──────────────────────────────────────────────
    const badge = document.getElementById('mtf-badge');
    const cons  = d.consensus || 'NEUTRAL';
    const clsMap = {
      'LONG':       'mtf-long',
      'SHORT':      'mtf-short',
      'LEAN_LONG':  'mtf-lean-long',
      'LEAN_SHORT': 'mtf-lean-short',
      'NEUTRAL':    'mtf-neutral',
    };
    badge.className   = 'mtf-badge ' + (clsMap[cons] || 'mtf-neutral');
    badge.textContent = cons.replace('_', ' ');

    // ── Long / Short percentages & bar ──────────────────────────────
    const lp = (d.long_pct  || 0);
    const sp = (d.short_pct || 0);
    document.getElementById('mtf-long-pct').textContent  = lp.toFixed(0);
    document.getElementById('mtf-short-pct').textContent = sp.toFixed(0);
    document.getElementById('mtf-bar-long').style.width  = lp + '%';
    document.getElementById('mtf-bar-short').style.width = sp + '%';

    // ── Per-TF rows ──────────────────────────────────────────────────
    const grid = document.getElementById('mtf-tf-rows');
    grid.innerHTML = '';
    const allTFs  = ['1m', '3m', '5m', '15m'];
    const tfCutoff = allTFs.indexOf(activeItvl);
    const tfOrder  = allTFs.slice(tfCutoff >= 0 ? tfCutoff : 0);
    for (const itvl of tfOrder) {
      const tf      = (d.timeframes || {})[itvl];
      const isEntry = (itvl === activeItvl);
      const isSupp  = (itvl === '15m');

      const dir    = tf ? tf.direction : null;
      const dirCls = dir === 'BUY'  ? 'mtf-dir-buy'
                   : dir === 'SELL' ? 'mtf-dir-sell'
                   : 'mtf-dir-none';

      const trend    = tf ? tf.trend : null;
      const trendDot = trend === 'bullish'
        ? `<span class="mtf-trend-dot" style="color:var(--bull)">●</span>`
        : trend === 'bearish'
        ? `<span class="mtf-trend-dot" style="color:var(--bear)">●</span>`
        : `<span class="mtf-trend-dot" style="color:var(--muted)">○</span>`;

      const rrTxt  = tf && tf.risk_reward != null
        ? `<span class="mtf-tf-rr">${tf.risk_reward.toFixed(1)}:1</span>` : '';
      const zoneTxt = tf && tf.at_zone
        ? `<span class="mtf-zone-flag" title="Price at FVG zone">⚡</span>` : '';
      const suppTxt = isEntry ? `<span class="mtf-supp-lbl">(entry)</span>`
                    : isSupp  ? `<span class="mtf-supp-lbl">(htf)</span>`
                    : '';

      const row = document.createElement('div');
      row.className = 'mtf-tf-row';
      row.innerHTML = `
        <span class="mtf-tf-label">${itvl}</span>
        ${trendDot}
        <span class="mtf-dir ${dirCls}">${dir || '—'}</span>
        ${rrTxt}${zoneTxt}${suppTxt}`;
      grid.appendChild(row);
    }

    // ── Best consensus levels ────────────────────────────────────────
    if (d.entry != null) {
      document.getElementById('mtf-entry').textContent = '$' + d.entry.toFixed(2);
      document.getElementById('mtf-sl').textContent    = d.stop_loss   != null ? '$' + d.stop_loss.toFixed(2)   : '—';
      document.getElementById('mtf-tp').textContent    = d.take_profit != null ? '$' + d.take_profit.toFixed(2) : '—';
      document.getElementById('mtf-rr').textContent    = d.risk_reward != null ? d.risk_reward.toFixed(2) + ':1' : '—';
    } else {
      ['mtf-entry','mtf-sl','mtf-tp','mtf-rr'].forEach(id => {
        document.getElementById(id).textContent = '—';
      });
    }

    // ── Source TF info ───────────────────────────────────────────────
    const parts = [];
    if (d.entry_tf)  parts.push('Entry: ' + d.entry_tf);
    if (d.target_tf) parts.push('Target: ' + d.target_tf);
    document.getElementById('mtf-src').textContent = parts.join(' · ');

  } catch (err) {
    if (requestId !== _mtfLoadSeq) return;
    if (lbl) lbl.textContent = 'error';
    console.error('loadMTFSignal error:', err);
  }
}
/* ── AI Signal panel ────────────────────────────────────────────────── */
async function loadSignal() {
  if (!activeTicker) return;
  const requestId = ++_signalLoadSeq;
  const requestedTicker = activeTicker;
  const requestedItvl = activeItvl;
  const period = PERIOD[requestedItvl] || '5d';
  const badge  = document.getElementById('sig-badge');
  const status = document.getElementById('sig-status');

  badge.className   = 'signal-badge signal-wait';
  badge.textContent = '···';
  status.textContent = 'Analyzing…';

  try {
    const r = await fetch(
      `/api/signal?ticker=${requestedTicker}&interval=${requestedItvl}&period=${period}`
    );
    const d = await r.json();
    if (requestId !== _signalLoadSeq || requestedTicker !== activeTicker || requestedItvl !== activeItvl) return;

    if (!d.ok) {
      badge.textContent = '!';
      status.textContent = d.error || 'Error';
      return;
    }

    const sig  = d.signal;
    const conf = d.confidence;
    const smc  = d.smc || {};

    // ── Badge ──────────────────────────────────────────────────
    // WAIT = confidence below 60% but a nearby SMC setup exists
    const isPending = smc.entry != null && sig === 'HOLD' && conf < 60.0;
    const cls = sig === 'BUY'  ? 'signal-buy'
              : sig === 'SELL' ? 'signal-sell'
              : 'signal-hold';
    badge.className   = 'signal-badge ' + cls;
    badge.textContent = isPending ? 'WAIT' : sig;

    // ── Confidence bar ────────────────────────────────────────────
    document.getElementById('sig-conf-val').textContent = conf.toFixed(1) + '%';
    const fill = document.getElementById('sig-bar');
    fill.style.width      = Math.min(conf, 100) + '%';
    fill.style.background = sig === 'BUY'  ? 'var(--bull)'
                          : sig === 'SELL' ? 'var(--bear)'
                          : conf >= 60     ? 'var(--muted)'
                          : '#64748b';

    // 60% threshold tick mark on the bar
    const barWrap = fill.parentElement;
    if (!barWrap.querySelector('.conf-threshold')) {
      const mk = document.createElement('div');
      mk.className = 'conf-threshold';
      mk.style.cssText = 'position:absolute;left:60%;top:0;width:1px;height:100%;background:rgba(255,255,255,0.35);pointer-events:none;';
      barWrap.style.position = 'relative';
      barWrap.appendChild(mk);
    }

    // ── Probabilities ────────────────────────────────────────────
    const p       = ((d.ml || {}).probabilities) || {};
    const buyPct  = p.BUY  ?? 0;
    const holdPct = p.HOLD ?? 0;
    const sellPct = p.SELL ?? 0;
    const maxPct  = Math.max(buyPct, holdPct, sellPct);

    document.getElementById('pb-buy').textContent  = buyPct.toFixed(1)  + '%';
    document.getElementById('pb-hold').textContent = holdPct.toFixed(1) + '%';
    document.getElementById('pb-sell').textContent = sellPct.toFixed(1) + '%';

    document.getElementById('pb-buy-bar').style.width  = buyPct  + '%';
    document.getElementById('pb-hold-bar').style.width = holdPct + '%';
    document.getElementById('pb-sell-bar').style.width = sellPct + '%';

    // Highlight the dominant class; dim the others
    document.getElementById('pb-buy').style.color  = buyPct  === maxPct ? 'var(--bull)' : 'var(--border)';
    document.getElementById('pb-hold').style.color = holdPct === maxPct ? 'var(--text)'  : 'var(--border)';
    document.getElementById('pb-sell').style.color = sellPct === maxPct ? 'var(--bear)' : 'var(--border)';

    // ── SMC levels (always populated when an FVG setup exists) ───
    if (smc.entry != null) {
      const atZone = smc.price_at_zone;
      // ~ prefix when price hasn't reached the zone yet (pending entry)
      document.getElementById('lv-entry').textContent =
        (atZone ? '' : '~') + '$' + smc.entry.toFixed(2);
      document.getElementById('lv-sl').textContent    = '$' + smc.stop_loss.toFixed(2);
      document.getElementById('lv-tp').textContent    = '$' + smc.take_profit.toFixed(2);
      document.getElementById('lv-rr').textContent    = smc.risk_reward.toFixed(2) + ':1';
    } else {
      document.getElementById('lv-entry').textContent = '—';
      document.getElementById('lv-sl').textContent    = '—';
      document.getElementById('lv-tp').textContent    = '—';
      document.getElementById('lv-rr').textContent    = '—';
    }

    // ── SMC score pips (0–6) + alignment tag ──────────────────────
    const score    = smc.smc_score || 0;
    const maxScore = 6;
    let pipHtml = '';
    for (let i = 0; i < maxScore; i++) {
      pipHtml += `<span class="smc-score-pip ${i < score ? 'pip-on' : 'pip-off'}"></span>`;
    }

    const alignCls = 'align-' + (d.alignment || 'ml_only');
    const alignLbl = (d.alignment || 'ml_only').replace(/_/g, ' ');
    const trainTxt = d.ml.trained ? '' : ' · Training ML…';
    const zoneTxt  = smc.price_at_zone === false && smc.entry != null
                     ? ' · ⚡ Pending' : '';

    status.innerHTML = `
      <span class="align-tag ${alignCls}">${alignLbl.toUpperCase()}</span>
       SMC ${pipHtml}${zoneTxt}${trainTxt}`;

    // ── Reason text ──────────────────────────────────────────────
    document.getElementById('sig-reason').textContent = smc.reason || '';

    // Cache for paper trading execute button
    _cacheSignal(d);

    // Kick off MTF fetch — updates the inline MTF section below
    loadMTFSignal();

  } catch (err) {
    if (requestId !== _signalLoadSeq) return;
    document.getElementById('sig-badge').textContent  = '!';
    document.getElementById('sig-status').textContent = 'Network error';
    console.error('loadSignal error:', err);
  }
}

/* ── Startup ────────────────────────────────────────────────────────────── */
window.addEventListener('DOMContentLoaded', () => {
  initChart();
  applyPersistedUIState();
  ['ar-btn','hdr-ar'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.classList.toggle('active', arEnabled);
  });
  updateARLabel();
  if (activeTicker) {
    loadChart(false);
    startAR();
  }
  // Init paper trading panel
  refreshPaperStatus();
});

window.addEventListener('beforeunload', () => {
  persistViewState();
  persistUIState();
});

/* ── Paper Trading ──────────────────────────────────────────────────────── */

// Latest signal cache — populated by loadSignal(), consumed by executePaperTrade()
let _lastSignal = null;
// Dedup key: prevents re-executing the same signal on every auto-refresh cycle
let _lastAutoTradeKey = null;
// Set to true while any open orders exist; blocks auto-trade until they clear
let _hasOpenOrders = false;

// Track the paper-status polling timer
let _ptStatusTimer = null;

function setPtMsg(text, type = '') {
  const el = document.getElementById('pt-msg');
  if (!el) return;
  el.textContent = text;
  el.className   = 'pt-msg' + (type ? ' ' + type : '');
}

function setConnDot(state) {
  // state: 'ok' | 'err' | 'wait'
  const dot = document.getElementById('pt-conn-dot');
  if (!dot) return;
  dot.className = 'pt-conn-dot pt-conn-' + state;
  dot.title = state === 'ok'   ? 'API connected'
            : state === 'err'  ? 'API error / disconnected'
            : 'Checking…';
}

async function refreshPaperStatus() {
  try {
    const ticker = activeTicker || '';
    const r      = await fetch(`/api/paper/status${ticker ? '?ticker=' + encodeURIComponent(ticker) : ''}`);
    const d      = await r.json();

    setConnDot(d.connected ? 'ok' : 'err');

    // Sync toggle for the current ticker
    const toggle = document.getElementById('pt-toggle');
    if (toggle && ticker) {
      const isOn = !!(d.enabled || {})[ticker];
      toggle.checked = isOn;
      syncToggleUI(isOn);
    }

    // Update account info
    if (d.connected) {
      const ar = await fetch('/api/paper/account');
      const ad = await ar.json();
      if (ad.ok) {
        document.getElementById('pt-cash-limit').textContent    =
          '$' + (ad.cash_limit || 0).toLocaleString('en-US', {maximumFractionDigits: 2});
        document.getElementById('pt-buying-power').textContent  =
          '$' + (ad.buying_power || 0).toLocaleString('en-US', {maximumFractionDigits: 2});
        const pnl = ad.total_pnl || 0;
        const pnlEl = document.getElementById('pt-pnl');
        pnlEl.textContent = (pnl >= 0 ? '+' : '') + '$' + pnl.toFixed(2);
        pnlEl.style.color = pnl >= 0 ? 'var(--bull)' : 'var(--bear)';
        document.getElementById('pt-acct-info').style.display = 'block';
      }
      // Refresh orders list
      refreshOrdersList();
    }
  } catch (err) {
    setConnDot('err');
    console.warn('refreshPaperStatus error:', err);
  }
}

async function refreshOrdersList() {
  try {
    const r = await fetch('/api/paper/orders');
    const d = await r.json();
    if (!d.ok) return;
    const list = document.getElementById('pt-orders-list');
    const wrap = document.getElementById('pt-orders-wrap');
    if (!list || !wrap) return;

    list.innerHTML = '';
    if (!d.orders || d.orders.length === 0) {
      _hasOpenOrders = false;
      _lastAutoTradeKey = null; // allow fresh auto-trade once orders clear
      wrap.style.display = 'none';
      return;
    }
    _hasOpenOrders = true;
    wrap.style.display = 'block';
    for (const o of d.orders) {
      const sideClass = o.side && o.side.toLowerCase().includes('buy')
                        ? 'pt-order-side-buy' : 'pt-order-side-sell';
      const row = document.createElement('div');
      row.className = 'pt-order-row';

      // Build DOM nodes (no innerHTML) to avoid XSS from API response data
      const sideSpan   = document.createElement('span');
      sideSpan.className = sideClass;
      sideSpan.textContent = (o.side || '').toUpperCase();

      const symSpan    = document.createElement('span');
      symSpan.textContent = o.symbol || '';

      const qtySpan    = document.createElement('span');
      qtySpan.style.color = 'var(--muted)';
      // Show dollar notional when available, fall back to qty for older orders
      qtySpan.textContent = o.notional != null
        ? '$' + Number(o.notional).toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})
        : '×' + (o.qty || '');

      const stSpan     = document.createElement('span');
      stSpan.style.cssText = 'font-size:10px;color:var(--muted)';
      stSpan.textContent   = o.status || '';

      const cancelBtn  = document.createElement('button');
      cancelBtn.className   = 'pt-cancel-btn';
      cancelBtn.title       = 'Cancel order';
      cancelBtn.textContent = '✕';
      cancelBtn.addEventListener('click', () => cancelPaperOrder(o.id));

      row.appendChild(sideSpan);
      row.appendChild(symSpan);
      row.appendChild(qtySpan);
      row.appendChild(stSpan);
      row.appendChild(cancelBtn);
      list.appendChild(row);
    }
  } catch (err) {
    console.warn('refreshOrdersList error:', err);
  }
}

async function cancelPaperOrder(orderId) {
  setPtMsg('Cancelling…');
  try {
    const r = await fetch(`/api/paper/cancel/${encodeURIComponent(orderId)}`, { method: 'DELETE' });
    const d = await r.json();
    if (d.ok) {
      setPtMsg('Order cancelled.', 'ok');
      refreshOrdersList();
    } else {
      setPtMsg('Cancel failed: ' + (d.error || 'unknown'), 'err');
    }
  } catch (err) {
    setPtMsg('Network error during cancel.', 'err');
    console.error('cancelPaperOrder error:', err);
  }
}

function syncToggleUI(isOn) {
  const lbl = document.getElementById('pt-status-lbl');
  if (lbl) {
    lbl.textContent = isOn ? 'ON — Auto Trading' : 'OFF';
    lbl.className   = 'pt-status-lbl ' + (isOn ? 'pt-status-on' : 'pt-status-off');
  }
  // Switching OFF clears the dedup key so re-enabling fires on the current signal
  if (!isOn) _lastAutoTradeKey = null;
  refreshExecuteButton();
}

async function onPaperToggle(enabled) {
  syncToggleUI(enabled);
  if (!activeTicker) return;
  setPtMsg(enabled ? 'Enabling paper trading…' : 'Disabling paper trading…');
  try {
    const r = await fetch('/api/paper/toggle', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ ticker: activeTicker, enabled }),
    });
    const d = await r.json();
    if (d.ok) {
      setPtMsg(enabled
        ? `Paper trading ENABLED for ${activeTicker}.`
        : `Paper trading DISABLED for ${activeTicker}.`,
        enabled ? 'ok' : '');
      refreshPaperStatus();
    } else {
      setPtMsg('Toggle failed: ' + (d.error || 'unknown'), 'err');
      // Revert checkbox
      const toggle = document.getElementById('pt-toggle');
      if (toggle) toggle.checked = !enabled;
      syncToggleUI(!enabled);
    }
  } catch (err) {
    setPtMsg('Network error.', 'err');
    const toggle = document.getElementById('pt-toggle');
    if (toggle) toggle.checked = !enabled;
    syncToggleUI(!enabled);
    console.error('onPaperToggle error:', err);
  }
}

function refreshExecuteButton() {
  const btn      = document.getElementById('pt-execute-btn');
  const toggle   = document.getElementById('pt-toggle');
  if (!btn) return;

  const isOn = toggle && toggle.checked;
  const sig  = _lastSignal;
  const hasActionable = sig && sig.smc && sig.smc.entry != null
                        && (sig.signal === 'BUY' || sig.signal === 'SELL');

  if (!isOn) {
    btn.disabled     = true;
    btn.textContent  = 'Toggle ON to auto-trade';
    btn.className    = 'pt-execute-btn';
    return;
  }

  if (!hasActionable) {
    btn.disabled    = true;
    btn.textContent = '⚡ Auto: Watching for signal…';
    btn.className   = 'pt-execute-btn';
    return;
  }

  // Actionable — button still works as a manual override
  btn.disabled    = false;
  btn.textContent = `⚡ Auto ▶ ${sig.signal}  ${activeTicker}`;
  btn.className   = 'pt-execute-btn ' + (sig.signal === 'BUY' ? 'buy' : 'sell');
}

async function executePaperTrade() {
  const sig = _lastSignal;
  if (!sig || !sig.smc || sig.smc.entry == null) {
    setPtMsg('No valid signal to execute.', 'err');
    return;
  }

  const payload = {
    ticker:      activeTicker,
    side:        sig.signal,           // "BUY" or "SELL"
    entry:       sig.smc.entry,
    stop_loss:   sig.smc.stop_loss,
    take_profit: sig.smc.take_profit,
    confidence:  sig.confidence,
  };

  setPtMsg('Submitting bracket order…');
  document.getElementById('pt-execute-btn').disabled = true;

  try {
    const r = await fetch('/api/paper/execute', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });
    const d = await r.json();

    if (d.ok) {
      const notional = d.notional != null
        ? '$' + Number(d.notional).toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})
        : (d.qty ? d.qty + '×' : '');
      setPtMsg(
        `✓ Order placed! ${d.side} ${notional} ${d.symbol} | ID: ${d.order_id.slice(0, 8)}…`,
        'ok'
      );
      refreshOrdersList();
      refreshPaperStatus();
    } else {
      setPtMsg('✗ ' + (d.error || 'Order failed'), 'err');
      console.error('executePaperTrade error:', d.error);
    }
  } catch (err) {
    setPtMsg('Network error — order not sent.', 'err');
    console.error('executePaperTrade network error:', err);
  } finally {
    refreshExecuteButton();
  }
}

// Hook into loadSignal so the execute button always reflects the latest signal
const _origLoadSignal = loadSignal;
loadSignal = async function() {
  await _origLoadSignal.apply(this, arguments);
};

// Patch the existing loadSignal to cache the result (we wrap the fetch section)
// The cache is set via a small helper called from inside loadSignal.
function _cacheSignal(data) {
  _lastSignal = data;
  refreshExecuteButton();
  _maybeAutoTrade();
}

// Auto-execute a trade when the toggle is ON and a new actionable signal arrives.
// Deduplication is done via a key of ticker|side|entry|interval so the same
// setup is never submitted twice across consecutive auto-refresh cycles.
function _maybeAutoTrade() {
  const toggle = document.getElementById('pt-toggle');
  if (!toggle || !toggle.checked) return;

  const sig = _lastSignal;
  if (!sig || !sig.smc || sig.smc.entry == null) return;
  if (sig.signal !== 'BUY' && sig.signal !== 'SELL') return;

  // Don't attempt a new entry while any orders are still open
  if (_hasOpenOrders) return;

  // Key on direction only (not entry price) — entry fluctuates every candle on
  // 1m bars and would otherwise re-fire on every auto-refresh cycle.
  const key = `${activeTicker}|${sig.signal}|${activeItvl}`;
  if (key === _lastAutoTradeKey) return; // already executed this setup

  _lastAutoTradeKey = key;
  setPtMsg(`⚡ Auto-executing ${sig.signal} signal…`);
  executePaperTrade();
}

// Poll paper status every 15 s while the page is open
setInterval(refreshPaperStatus, 15_000);
