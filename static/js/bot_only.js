tInterval(refreshPaperStatus, 15_000);

/* ΓöÇΓöÇ Autonomous Bot Dashboard Tab Controls & Logic ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ */
let activeTab = 'chart';
let botStatusTimer = null;
let lastLogTimestamp = 0;
let knownLogs = new Set();

function switchTab(tabName) {
  activeTab = tabName;
  
  // Toggle tab buttons
  document.getElementById('tab-btn-chart').classList.toggle('active', tabName === 'chart');
  document.getElementById('tab-btn-bot').classList.toggle('active', tabName === 'bot');
  
  // Toggle layouts
  document.getElementById('view-chart').style.display = tabName === 'chart' ? 'grid' : 'none';
  document.getElementById('view-bot').style.display = tabName === 'bot' ? 'grid' : 'none';
  
  if (tabName === 'bot') {
    // Start polling bot status when dashboard is active
    refreshBotStatus();
    if (!botStatusTimer) {
      botStatusTimer = setInterval(refreshBotStatus, 2000);
    }
  } else {
    // Stop polling when not in use
    if (botStatusTimer) {
      clearInterval(botStatusTimer);
      botStatusTimer = null;
    }
  }
}

async function refreshBotStatus() {
  try {
    const res = await fetch('/api/bot/status');
    const data = await res.json();
    if (!data.ok) return;
    
    // Update Running State
    const statusVal = document.getElementById('bot-status-val');
    const toggleBtn = document.getElementById('bot-toggle-btn');
    
    const pulseDot = document.getElementById('bot-pulse');

    if (data.running) {
      statusVal.textContent = 'RUNNING';
      statusVal.style.color = 'var(--bull)';
      toggleBtn.textContent = 'Stop Bot';
      toggleBtn.style.backgroundColor = 'var(--bear)';
      if (pulseDot) pulseDot.style.display = 'inline-block';
    } else {
      statusVal.textContent = 'STOPPED';
      statusVal.style.color = 'var(--bear)';
      toggleBtn.textContent = 'Start Bot';
      toggleBtn.style.backgroundColor = 'var(--bull)';
      if (pulseDot) pulseDot.style.display = 'none';
    }
    
    // Update Stats
    if (document.getElementById('bot-regime-val')) {
      let rText = 'Unknown';
      if (typeof data.ai_regime === 'object') {
        const eq = data.ai_regime.Equity ? data.ai_regime.Equity.split(' ')[0] : '?';
        const cr = data.ai_regime.Crypto ? data.ai_regime.Crypto.split(' ')[0] : '?';
        const co = data.ai_regime.Commodity ? data.ai_regime.Commodity.split(' ')[0] : '?';
        rText = `EQ: ${eq} | BTC: ${cr} | GLD: ${co}`;
      } else {
        rText = data.ai_regime || 'Unknown';
      }
      document.getElementById('bot-regime-val').textContent = rText;
    }
    document.getElementById('bot-uptime-val').textContent = formatUptime(data.uptime);
    document.getElementById('bot-cycles-val').textContent = data.cycle_count || 0;
    document.getElementById('bot-signals-val').textContent = data.signals_today || 0;
    document.getElementById('bot-orders-val').textContent = data.orders_today || 0;
    
    // Update Config Input controls (only if they aren't currently being modified by the user)
    if (document.activeElement !== document.getElementById('bot-risk-slider')) {
      document.getElementById('bot-risk-slider').value = data.config.max_risk_pct * 100;
      document.getElementById('bot-risk-label').textContent = (data.config.max_risk_pct * 100).toFixed(1) + '%';
    }
    if (document.activeElement !== document.getElementById('bot-max-positions-slider')) {
      document.getElementById('bot-max-positions-slider').value = data.config.max_positions;
      document.getElementById('bot-max-positions-label').textContent = data.config.max_positions;
    }
    if (document.activeElement !== document.getElementById('bot-scan-interval-slider')) {
      document.getElementById('bot-scan-interval-slider').value = data.config.scan_interval;
      document.getElementById('bot-scan-interval-label').textContent = data.config.scan_interval + 's';
    }
    document.getElementById('bot-dry-run-toggle').checked = data.config.dry_run;
    
    // Render Instruments Table
    const tableBody = document.getElementById('bot-instruments-table');
    if (data.instruments && data.instruments.length > 0) {
      tableBody.innerHTML = data.instruments.map(inst => {
        const lastScanText = inst.last_scan > 0 
          ? new Date(inst.last_scan * 1000).toLocaleTimeString() 
          : 'Never';
        return `
          <tr style="border-bottom: 1px solid var(--border);">
            <td style="padding: 8px 0; font-weight: 600;">${inst.ticker}</td>
            <td style="padding: 8px 0; color: var(--muted);">${inst.strategy}</td>
            <td style="padding: 8px 0;"><span class="itvl-btn active" style="padding: 2px 6px; font-size: 10px;">${inst.timeframe}</span></td>
            <td style="padding: 8px 0; text-align: right; color: var(--muted);">${lastScanText}</td>
          </tr>
        `;
      }).join('');
    } else {
      tableBody.innerHTML = `
        <tr>
          <td colspan="4" style="text-align: center; color: var(--muted); padding: 20px 0;">No instruments loaded</td>
        </tr>
      `;
    }
    
    // Fetch logs
    await refreshBotLogs();
    
    // Fetch live broker state (Positions & Orders)
    await refreshBotBrokerState();

  } catch (err) {
    console.error('Error refreshing bot status:', err);
  }
}

async function refreshBotLogs() {
  try {
    const res = await fetch('/api/bot/logs');
    const data = await res.json();
    if (!data.ok) return;
    
    const consoleEl = document.getElementById('bot-log-console');
    let shouldScroll = consoleEl.scrollHeight - consoleEl.clientHeight <= consoleEl.scrollTop + 50;
    
    let added = false;
    data.logs.forEach(line => {
      if (!knownLogs.has(line)) {
        knownLogs.add(line);
        const lineEl = document.createElement('div');
        lineEl.style.marginBottom = '4px';
        lineEl.style.display = 'flex';
        lineEl.style.alignItems = 'flex-start';
        
        // Parse the log line: "2026-06-10 12:00:00 [INFO] src.engine ΓÇö Message"
        // Regex to separate timestamp, level, and message
        const match = line.match(/^(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})\s\[(.*?)\]\s.*?ΓÇö\s(.*)$/);
        
        if (match) {
          const [, timestamp, level, msg] = match;
          
          let lvlClass = 'info';
          if (level === 'ERROR') lvlClass = 'error';
          if (level === 'WARNING') lvlClass = 'warn';
          
          let msgClass = '';
          if (msg.includes('[SIGNAL]')) msgClass = 'highlight-signal';
          if (msg.includes('[ORDER PLACED]')) msgClass = 'highlight-order';
          if (msg.includes('[REJECTED]')) msgClass = 'highlight-reject';

          lineEl.innerHTML = `
            <span class="log-time" style="flex-shrink:0;">${timestamp.split(' ')[1]}</span>
            <span class="log-level ${lvlClass}" style="flex-shrink:0;">${level}</span>
            <span class="log-msg ${msgClass}">${msg}</span>
          `;
        } else {
          // Fallback if regex fails to match structure
          lineEl.textContent = line;
          if (line.includes('[ERROR]')) lineEl.style.color = '#ef4444';
        }
        
        consoleEl.appendChild(lineEl);
        added = true;
      }
    });
    
    if (added && shouldScroll) {
      consoleEl.scrollTop = consoleEl.scrollHeight;
    }
  } catch (err) {
    console.error('Error fetching bot logs:', err);
  }
}

async function toggleBotEngine() {
  const statusVal = document.getElementById('bot-status-val');
  const currentlyRunning = statusVal.textContent === 'RUNNING';
  
  try {
    const res = await fetch('/api/bot/toggle', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled: !currentlyRunning })
    });
    const data = await res.json();
    if (data.ok) {
      refreshBotStatus();
    } else {
      alert('Error toggling bot: ' + data.error);
    }
  } catch (err) {
    console.error('Error toggling bot:', err);
  }
}

function updateBotConfigLabels() {
  const riskVal = parseFloat(document.getElementById('bot-risk-slider').value);
  document.getElementById('bot-risk-label').textContent = riskVal.toFixed(2) + '%';
  
  const positionsVal = parseInt(document.getElementById('bot-max-positions-slider').value, 10);
  document.getElementById('bot-max-positions-label').textContent = positionsVal;
  
  const intervalVal = parseInt(document.getElementById('bot-scan-interval-slider').value, 10);
  document.getElementById('bot-scan-interval-label').textContent = intervalVal + 's';
}

async function updateBotConfig() {
  const dryRun = document.getElementById('bot-dry-run-toggle').checked;
  const riskPct = parseFloat(document.getElementById('bot-risk-slider').value) / 100.0;
  const maxPositions = parseInt(document.getElementById('bot-max-positions-slider').value, 10);
  const scanInterval = parseFloat(document.getElementById('bot-scan-interval-slider').value);
  
  try {
    const res = await fetch('/api/bot/configure', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dry_run: dryRun,
        max_risk_pct: riskPct,
        max_positions: maxPositions,
        scan_interval: scanInterval
      })
    });
    const data = await res.json();
    if (!data.ok) {
      console.error('Failed to update bot config:', data.error);
    }
  } catch (err) {
    console.error('Error updating bot config:', err);
  }
}

function clearConsoleLog() {
  const consoleEl = document.getElementById('bot-log-console');
  consoleEl.innerHTML = '';
  knownLogs.clear();
}

async function refreshBotBrokerState() {
  try {
    const res = await fetch('/api/bot/broker');
    const data = await res.json();
    if (!data.ok) return;

    // Render Positions
    const posTable = document.getElementById('bot-positions-table');
      if (data.positions && data.positions.length > 0) {
        posTable.innerHTML = data.positions.map(pos => {
          const sideColor = pos.side.toLowerCase() === 'long' || pos.side.toLowerCase() === 'buy' ? 'var(--bull)' : 'var(--bear)';
          const pnlColor = pos.unrealized_pl >= 0 ? 'var(--bull)' : 'var(--bear)';
          const pnlSign = pos.unrealized_pl >= 0 ? '+' : '';
          
          const slStr = pos.stop_loss ? `$${pos.stop_loss.toFixed(2)}` : 'ΓÇö';
          const tpStr = pos.take_profit ? `$${pos.take_profit.toFixed(2)}` : 'ΓÇö';
          
          return `
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.02);">
              <td style="padding: 8px 4px; font-weight: 600;">${pos.symbol}</td>
              <td style="padding: 8px 4px; color: ${sideColor}; font-weight: bold; text-transform: uppercase;">${pos.side}</td>
              <td style="padding: 8px 4px; text-align: right;">${pos.qty}</td>
              <td style="padding: 8px 4px; text-align: right;">$${pos.avg_entry_price.toFixed(2)}</td>
              <td style="padding: 8px 4px; text-align: right;">${slStr}</td>
              <td style="padding: 8px 4px; text-align: right;">${tpStr}</td>
              <td style="padding: 8px 4px; text-align: right; color: ${pnlColor}; font-weight: bold;">${pnlSign}$${pos.unrealized_pl.toFixed(2)}</td>
            </tr>
          `;
        }).join('');
      } else {
        posTable.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--muted); padding: 20px 0;">No active positions</td></tr>`;
      }

    // Render Orders
    const ordTable = document.getElementById('bot-orders-table');
    if (data.orders && data.orders.length > 0) {
      ordTable.innerHTML = data.orders.map(ord => {
        const sideColor = ord.side.toLowerCase() === 'buy' ? 'var(--bull)' : 'var(--bear)';
        
        let priceStr = 'MKT';
        if (ord.limit_price) {
          priceStr = `$${ord.limit_price.toFixed(2)}`;
        } else if (ord.stop_price) {
          priceStr = `STP $${ord.stop_price.toFixed(2)}`;
        }
        
        return `
          <tr style="border-bottom: 1px solid rgba(255,255,255,0.02);">
            <td style="padding: 8px 4px; font-weight: 600;">${ord.symbol}</td>
            <td style="padding: 8px 4px; color: ${sideColor}; font-weight: bold; text-transform: uppercase;">${ord.side}</td>
            <td style="padding: 8px 4px; color: var(--muted); text-transform: capitalize;">${ord.type}</td>
            <td style="padding: 8px 4px; text-align: right; font-family: monospace;">${priceStr}</td>
          </tr>
        `;
      }).join('');
    } else {
      ordTable.innerHTML = `<tr><td colspan="4" style="text-align: center; color: var(--muted); padding: 20px 0;">No pending orders</td></tr>`;
    }

  } catch (err) {
    console.error('Error refreshing broker state:', err);
  }
}

function formatUptime(secs) {
  if (!secs) return '0s';
  const hrs = Math.floor(secs / 3600);
  const mins = Math.floor((secs % 3600) / 60);
  const s = secs % 60;
  
  let res = '';
  if (hrs > 0) res += `${hrs}h `;
  if (mins > 0 || hrs > 0) res += `${mins}m `;
  res += `${s}s`;
  return res;
}
