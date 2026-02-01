#!/usr/bin/env python3
"""
PROTEUS Industrial Dashboard - Professional Corporate UI
Mobile-first responsive design with real-time monitoring
"""

from nicegui import ui, app
import asyncio
import json
import csv
import io
from datetime import datetime, timezone
from collections import deque

from proteus_gateway import ProteusGateway, classify_iso_zone

# =============================================================================
# Help Content
# =============================================================================

HELP = {
    'health': '''
### Equipment Health Score

A comprehensive indicator of your machinery's condition (0-100%).

| Score | Status | Action Required |
|-------|--------|-----------------|
| 80-100% | Healthy | Continue normal operation |
| 50-80% | Attention | Schedule preventive maintenance |
| 0-50% | Critical | Immediate inspection needed |

The score combines vibration analysis, temperature trends, and signal patterns to predict potential failures before they occur.
''',
    'vibration': '''
### Vibration Severity (ISO 10816-1)

International standard for machine vibration assessment:

| Zone | Velocity | Condition |
|------|----------|-----------|
| A | < 1.12 mm/s | Excellent |
| B | 1.12 - 2.8 mm/s | Good |
| C | 2.8 - 7.1 mm/s | Alert |
| D | > 7.1 mm/s | Danger |

Higher vibration typically indicates bearing wear, imbalance, or misalignment.
''',
    'impact': '''
### Impact Detection (Crest Factor)

Detects sudden mechanical shocks and impacts in rotating equipment.

| Value | Meaning |
|-------|---------|
| < 3.0 | Normal smooth operation |
| 3.0 - 4.5 | Monitor - possible early wear |
| > 4.5 | Alert - impacts detected |

High values suggest bearing defects, gear damage, or loose components.
''',
    'wear': '''
### Wear Detection (Kurtosis)

Measures signal "spikiness" to detect developing faults.

| Value | Meaning |
|-------|---------|
| < 4.0 | Normal wear pattern |
| 4.0 - 6.0 | Early stage degradation |
| > 6.0 | Active damage occurring |

Sensitive to bearing surface defects and contamination.
''',
    'export': '''
### Data Export

Save sensor data for analysis or AI model training.

**CSV Format** - Opens in Excel, ideal for reports and manual analysis.

**JSON Format** - Structured data for software integration and machine learning.

Click "Start Recording" to begin capturing data, then export when ready.
''',
    'fft': '''
### Frequency Spectrum (FFT)

Shows **what frequencies are present** in your machine's vibration signature.

**Why it matters:**
- Each fault type creates vibration at specific frequencies
- Helps identify the ROOT CAUSE of problems
- Essential for predictive maintenance

| Pattern | Likely Cause |
|---------|--------------|
| Low freq spike (1x RPM) | Imbalance |
| 2x RPM spike | Misalignment |
| High freq harmonics | Bearing wear |
| Random broadband | Cavitation/turbulence |

**How to read:**
- X-axis: Frequency (Hz)
- Y-axis: Amplitude (vibration intensity)
- Peaks indicate dominant frequencies
- Monitor for NEW peaks appearing over time
'''
}


class Dashboard:
    def __init__(self, gateway: ProteusGateway):
        self.gw = gateway
        self.history = {k: deque(maxlen=120) for k in
                       ['velocity', 'health', 'temp', 'pressure', 'cf', 'kurt', 'time']}
        self.recording = False
        self.data = []

    def help_dialog(self, key):
        with ui.dialog() as d, ui.card().classes('max-w-lg'):
            ui.markdown(HELP.get(key, '')).classes('prose prose-sm prose-invert')
            ui.button('Close', on_click=d.close).props('flat')
        d.open()

    def export_dialog(self):
        with ui.dialog() as d, ui.card().classes('p-6'):
            ui.markdown(HELP['export']).classes('prose prose-sm prose-invert mb-4')
            if not self.data:
                ui.label('No data recorded. Start recording first.').classes('text-amber-400')
            else:
                ui.label(f'{len(self.data)} samples ready for export').classes('text-emerald-400 mb-4')
            with ui.row().classes('gap-3 mt-4'):
                ui.button('CSV', on_click=lambda: self._export('csv', d), icon='table_view').props(
                    'color=primary' if self.data else 'color=grey disable')
                ui.button('JSON', on_click=lambda: self._export('json', d), icon='code').props(
                    'color=secondary' if self.data else 'color=grey disable')
                ui.button('Cancel', on_click=d.close).props('flat')
        d.open()

    def _export(self, fmt, dialog):
        if not self.data:
            return
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        if fmt == 'csv':
            out = io.StringIO()
            w = csv.DictWriter(out, fieldnames=self.data[0].keys())
            w.writeheader()
            w.writerows(self.data)
            ui.download(out.getvalue().encode(), f'proteus_{ts}.csv')
        else:
            content = json.dumps({'device': 'PROTEUS', 'exported': ts, 'samples': len(self.data), 'data': self.data}, indent=2)
            ui.download(content.encode(), f'proteus_{ts}.json')
        ui.notify(f'Exported {len(self.data)} records', type='positive')
        dialog.close()

    def toggle_record(self):
        self.recording = not self.recording
        if self.recording:
            self.data = []
            self.rec_btn.props('color=negative')
            self.rec_icon.classes(replace='text-red-500 animate-pulse')
            ui.notify('Recording started', type='positive')
        else:
            self.rec_btn.props('color=primary')
            self.rec_icon.classes(replace='text-slate-400')
            ui.notify(f'Stopped - {len(self.data)} samples captured', type='info')

    def build(self):
        ui.add_head_html('''
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap" rel="stylesheet">
<style>
:root {
    --bg-primary: #0f172a;
    --bg-card: #1e293b;
    --bg-card-hover: #334155;
    --border: #334155;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent: #3b82f6;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
}
body {
    font-family: 'Inter', system-ui, sans-serif !important;
    background: var(--bg-primary) !important;
    color: var(--text-primary);
}
.mono { font-family: 'JetBrains Mono', monospace; }
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    transition: all 0.2s ease;
}
.card:hover { border-color: #475569; }
.card-header {
    padding: 16px 20px;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.card-title {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
}
.card-body { padding: 20px; }
.metric-lg {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.5rem;
    font-weight: 700;
    line-height: 1;
}
.metric-unit {
    font-size: 0.875rem;
    color: var(--text-muted);
    margin-left: 4px;
}
.status-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
}
.status-dot.live { background: var(--success); box-shadow: 0 0 8px var(--success); animation: pulse 2s infinite; }
.status-dot.stale { background: var(--warning); }
.status-dot.offline { background: var(--danger); }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
.gauge-container { position: relative; width: 100%; padding-bottom: 75%; }
.gauge-chart { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }
.help-btn {
    width: 20px; height: 20px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; color: var(--text-muted);
    border: 1px solid var(--border);
    cursor: pointer; transition: all 0.2s;
}
.help-btn:hover { color: var(--text-primary); border-color: var(--text-secondary); }
.grid-dashboard {
    display: grid;
    gap: 16px;
    grid-template-columns: 1fr;
}
@media (min-width: 768px) {
    .grid-dashboard { grid-template-columns: repeat(2, 1fr); }
}
@media (min-width: 1280px) {
    .grid-dashboard { grid-template-columns: repeat(4, 1fr); }
}
.chart-container { width: 100%; height: 280px; }
@media (min-width: 768px) { .chart-container { height: 320px; } }
.stream-indicator {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    border-radius: 6px;
    background: var(--card-bg);
}
.stream-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}
.stream-dot.active { background: var(--success); box-shadow: 0 0 6px var(--success); }
.stream-dot.stale { background: var(--warning); }
.stream-dot.inactive { background: #475569; }
.connection-card {
    background: linear-gradient(135deg, rgba(30,41,59,0.8), rgba(15,23,42,0.9));
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
}
</style>
''')
        # Header
        with ui.header().classes('bg-slate-900 border-b border-slate-700 px-4 py-3'):
            with ui.row().classes('w-full items-center justify-between max-w-7xl mx-auto'):
                with ui.row().classes('items-center gap-3'):
                    ui.icon('precision_manufacturing').classes('text-2xl text-blue-500')
                    with ui.column().classes('gap-0'):
                        ui.label('PROTEUS').classes('text-lg font-bold leading-tight')
                        ui.label('Predictive Maintenance').classes('text-xs text-slate-400 leading-tight hidden sm:block')
                with ui.row().classes('items-center gap-2 sm:gap-4'):
                    with ui.row().classes('items-center gap-2'):
                        self.rec_icon = ui.icon('fiber_manual_record').classes('text-slate-400')
                        self.rec_btn = ui.button('Record', on_click=self.toggle_record).props('flat dense').classes('hidden sm:flex')
                        ui.button('', icon='fiber_manual_record', on_click=self.toggle_record).props('flat dense round').classes('sm:hidden')
                    ui.button('', icon='download', on_click=self.export_dialog).props('flat dense round')
                    ui.separator().props('vertical').classes('h-6 hidden sm:block')
                    with ui.row().classes('items-center gap-2'):
                        self.status_dot = ui.element('span').classes('status-dot offline')
                        self.status_text = ui.label('Offline').classes('text-sm font-medium text-red-400 hidden sm:block')

        # Main
        with ui.column().classes('w-full max-w-7xl mx-auto p-4 gap-4'):
            # Top metrics row
            with ui.element('div').classes('grid-dashboard'):
                # Health Score
                with ui.element('div').classes('card'):
                    with ui.element('div').classes('card-header'):
                        ui.label('Equipment Health').classes('card-title')
                        ui.label('?').classes('help-btn').on('click', lambda: self.help_dialog('health'))
                    with ui.element('div').classes('card-body text-center'):
                        self.health_value = ui.label('--').classes('metric-lg text-emerald-400')
                        ui.label('%').classes('metric-unit')
                        self.health_status = ui.label('Calculating...').classes('text-sm text-slate-400 mt-2 block')

                # Vibration
                with ui.element('div').classes('card'):
                    with ui.element('div').classes('card-header'):
                        ui.label('Vibration').classes('card-title')
                        ui.label('?').classes('help-btn').on('click', lambda: self.help_dialog('vibration'))
                    with ui.element('div').classes('card-body text-center'):
                        with ui.row().classes('items-baseline justify-center'):
                            self.vib_value = ui.label('--').classes('metric-lg text-violet-400')
                            ui.label('mm/s').classes('metric-unit')
                        self.vib_zone = ui.label('Zone --').classes('text-sm text-slate-400 mt-2 block')

                # Temperature
                with ui.element('div').classes('card'):
                    with ui.element('div').classes('card-header'):
                        ui.label('Temperature').classes('card-title')
                    with ui.element('div').classes('card-body text-center'):
                        with ui.row().classes('items-baseline justify-center'):
                            self.temp_value = ui.label('--').classes('metric-lg text-orange-400')
                            ui.label('°C').classes('metric-unit')
                        self.temp_delta = ui.label('+0.0°C from baseline').classes('text-sm text-slate-400 mt-2 block')

                # Pressure
                with ui.element('div').classes('card'):
                    with ui.element('div').classes('card-header'):
                        ui.label('Pressure').classes('card-title')
                    with ui.element('div').classes('card-body text-center'):
                        with ui.row().classes('items-baseline justify-center'):
                            self.pres_value = ui.label('--').classes('metric-lg text-cyan-400')
                            ui.label('mbar').classes('metric-unit')

            # Connection Status Panel
            with ui.element('div').classes('connection-card'):
                with ui.row().classes('items-center justify-between flex-wrap gap-4'):
                    # Device Status
                    with ui.row().classes('items-center gap-3'):
                        ui.icon('sensors').classes('text-2xl text-blue-400')
                        with ui.column().classes('gap-0'):
                            ui.label('Device Connection').classes('text-sm font-semibold text-slate-200')
                            self.device_status = ui.label('Searching for PROTEUS sensor...').classes('text-xs text-slate-400')

                    # Data Stream Indicators
                    with ui.row().classes('items-center gap-3 flex-wrap'):
                        # Temperature stream
                        with ui.element('div').classes('stream-indicator'):
                            self.stream_temp = ui.element('span').classes('stream-dot inactive')
                            ui.label('Temperature').classes('text-xs text-slate-300')

                        # Pressure stream
                        with ui.element('div').classes('stream-indicator'):
                            self.stream_pres = ui.element('span').classes('stream-dot inactive')
                            ui.label('Pressure').classes('text-xs text-slate-300')

                        # Vibration stream
                        with ui.element('div').classes('stream-indicator'):
                            self.stream_vib = ui.element('span').classes('stream-dot inactive')
                            ui.label('Vibration').classes('text-xs text-slate-300')

                        # Acceleration stream
                        with ui.element('div').classes('stream-indicator'):
                            self.stream_accel = ui.element('span').classes('stream-dot inactive')
                            ui.label('Acceleration').classes('text-xs text-slate-300')

            # Charts row - FULL WIDTH
            with ui.element('div').classes('grid gap-4 grid-cols-1'):
                # Vibration Chart
                with ui.element('div').classes('card'):
                    with ui.element('div').classes('card-header'):
                        ui.label('Vibration Trend').classes('card-title')
                        ui.label('Real-time velocity RMS monitoring').classes('text-xs text-slate-500 ml-2')
                    with ui.element('div').classes('card-body'):
                        self.vib_chart = ui.echart(self._chart_opts('Velocity (mm/s)')).classes('chart-container')

                # Health Chart
                with ui.element('div').classes('card'):
                    with ui.element('div').classes('card-header'):
                        ui.label('Health Index Trend').classes('card-title')
                        ui.label('Composite bearing health score').classes('text-xs text-slate-500 ml-2')
                    with ui.element('div').classes('card-body'):
                        self.health_chart = ui.echart(self._chart_opts('Health %', 0, 100, True)).classes('chart-container')

                # FFT Spectrum Chart
                with ui.element('div').classes('card'):
                    with ui.element('div').classes('card-header'):
                        ui.label('Frequency Spectrum (FFT)').classes('card-title')
                        ui.label('?').classes('help-btn').on('click', lambda: self.help_dialog('fft'))
                        ui.label('Identifies vibration sources by frequency').classes('text-xs text-slate-500 ml-2')
                    with ui.element('div').classes('card-body'):
                        self.fft_chart = ui.echart(self._fft_chart_opts()).classes('chart-container')

            # Analysis metrics
            with ui.element('div').classes('grid gap-4 grid-cols-1 md:grid-cols-2'):
                # Impact Detection
                with ui.element('div').classes('card'):
                    with ui.element('div').classes('card-header'):
                        ui.label('Impact Detection').classes('card-title')
                        ui.label('?').classes('help-btn').on('click', lambda: self.help_dialog('impact'))
                    with ui.element('div').classes('card-body'):
                        with ui.row().classes('items-center justify-between'):
                            with ui.column().classes('gap-1'):
                                with ui.row().classes('items-baseline gap-1'):
                                    self.cf_value = ui.label('--').classes('text-3xl font-bold mono text-slate-100')
                                ui.label('Crest Factor').classes('text-xs text-slate-500')
                            self.cf_status = ui.label('Normal').classes('px-3 py-1 rounded-full text-sm font-medium bg-emerald-500/20 text-emerald-400')
                        self.cf_bar = ui.linear_progress(value=0).props('rounded color=green').classes('mt-4')

                # Wear Detection
                with ui.element('div').classes('card'):
                    with ui.element('div').classes('card-header'):
                        ui.label('Wear Detection').classes('card-title')
                        ui.label('?').classes('help-btn').on('click', lambda: self.help_dialog('wear'))
                    with ui.element('div').classes('card-body'):
                        with ui.row().classes('items-center justify-between'):
                            with ui.column().classes('gap-1'):
                                with ui.row().classes('items-baseline gap-1'):
                                    self.kurt_value = ui.label('--').classes('text-3xl font-bold mono text-slate-100')
                                ui.label('Kurtosis').classes('text-xs text-slate-500')
                            self.kurt_status = ui.label('Normal').classes('px-3 py-1 rounded-full text-sm font-medium bg-emerald-500/20 text-emerald-400')
                        self.kurt_bar = ui.linear_progress(value=0).props('rounded color=green').classes('mt-4')

        # Footer
        with ui.footer().classes('bg-slate-900 border-t border-slate-700 py-2 px-4'):
            with ui.row().classes('w-full max-w-7xl mx-auto justify-between items-center text-xs text-slate-500'):
                self.stats = ui.label('Connecting...')
                ui.label('PROTEUS Gateway v2.0')

        ui.timer(0.5, self.update)

    def _chart_opts(self, name, ymin=None, ymax=None, health=False):
        opts = {
            'backgroundColor': 'transparent',
            'grid': {'left': 45, 'right': 15, 'top': 15, 'bottom': 25},
            'xAxis': {
                'type': 'category', 'data': [],
                'axisLine': {'lineStyle': {'color': '#334155'}},
                'axisLabel': {'color': '#64748b', 'fontSize': 10},
                'axisTick': {'show': False}
            },
            'yAxis': {
                'type': 'value',
                'axisLine': {'show': False},
                'axisLabel': {'color': '#64748b', 'fontSize': 10},
                'splitLine': {'lineStyle': {'color': '#1e293b'}},
            },
            'series': [{
                'type': 'line', 'data': [], 'smooth': True, 'symbol': 'none',
                'lineStyle': {'width': 2, 'color': '#8b5cf6'},
                'areaStyle': {'color': {'type': 'linear', 'x': 0, 'y': 0, 'x2': 0, 'y2': 1,
                    'colorStops': [{'offset': 0, 'color': 'rgba(139,92,246,0.25)'}, {'offset': 1, 'color': 'rgba(139,92,246,0)'}]}}
            }],
            'tooltip': {'trigger': 'axis', 'backgroundColor': '#1e293b', 'borderColor': '#334155',
                       'textStyle': {'color': '#f8fafc'}}
        }
        if ymin is not None:
            opts['yAxis']['min'] = ymin
        if ymax is not None:
            opts['yAxis']['max'] = ymax
        if health:
            opts['series'][0]['lineStyle']['color'] = '#10b981'
            opts['series'][0]['areaStyle']['color']['colorStops'] = [
                {'offset': 0, 'color': 'rgba(16,185,129,0.25)'}, {'offset': 1, 'color': 'rgba(16,185,129,0)'}]
        return opts

    def _fft_chart_opts(self):
        """FFT spectrum bar chart options."""
        return {
            'backgroundColor': 'transparent',
            'grid': {'left': 50, 'right': 20, 'top': 20, 'bottom': 35},
            'xAxis': {
                'type': 'category',
                'data': [f'{i*10}' for i in range(64)],  # Frequency bins (0-640 Hz approx)
                'name': 'Frequency (Hz)',
                'nameLocation': 'middle',
                'nameGap': 25,
                'nameTextStyle': {'color': '#64748b', 'fontSize': 10},
                'axisLine': {'lineStyle': {'color': '#334155'}},
                'axisLabel': {'color': '#64748b', 'fontSize': 9, 'interval': 7},
                'axisTick': {'show': False}
            },
            'yAxis': {
                'type': 'value',
                'name': 'Amplitude',
                'nameTextStyle': {'color': '#64748b', 'fontSize': 10},
                'axisLine': {'show': False},
                'axisLabel': {'color': '#64748b', 'fontSize': 10},
                'splitLine': {'lineStyle': {'color': '#1e293b'}},
            },
            'series': [{
                'type': 'bar',
                'data': [0] * 64,
                'itemStyle': {
                    'color': {
                        'type': 'linear', 'x': 0, 'y': 0, 'x2': 0, 'y2': 1,
                        'colorStops': [
                            {'offset': 0, 'color': '#f472b6'},  # Pink top
                            {'offset': 1, 'color': '#8b5cf6'}   # Purple bottom
                        ]
                    }
                },
                'barWidth': '60%'
            }],
            'tooltip': {
                'trigger': 'axis',
                'backgroundColor': '#1e293b',
                'borderColor': '#334155',
                'textStyle': {'color': '#f8fafc'},
                'formatter': '{b} Hz: {c}'
            }
        }

    async def update(self):
        s = self.gw.sensor_state
        now = datetime.now(timezone.utc)  # Use UTC to match gateway timestamps

        # Status
        last = s.get('last_update')
        if self.gw.connected and last:
            # Ensure both are UTC for comparison
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            age = (now - last).total_seconds()
            if age < 5:
                self.status_dot.classes(replace='status-dot live')
                self.status_text.text = 'Live'
                self.status_text.classes(replace='text-sm font-medium text-emerald-400 hidden sm:block')
                self.device_status.text = f'Connected to PROTEUS sensor • Streaming data'
            else:
                self.status_dot.classes(replace='status-dot stale')
                self.status_text.text = f'Stale ({int(age)}s)'
                self.status_text.classes(replace='text-sm font-medium text-amber-400 hidden sm:block')
                self.device_status.text = f'Connected but no new data for {int(age)}s'
        else:
            self.status_dot.classes(replace='status-dot offline')
            self.status_text.text = 'Offline'
            self.status_text.classes(replace='text-sm font-medium text-red-400 hidden sm:block')
            self.device_status.text = 'Searching for PROTEUS sensor...'

        # Data Stream Indicators
        def stream_status(stream_name):
            ts = s.get(f'{stream_name}_ts')
            if ts:
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                age = (now - ts).total_seconds()
                if age < 10:
                    return 'active'
                elif age < 30:
                    return 'stale'
            return 'inactive'

        # Check individual streams from gateway sensor_state timestamps
        temp_active = s.get('temperature', 0) != 0 and self.gw.connected
        pres_active = s.get('pressure', 0) != 0 and self.gw.connected
        vib_active = s.get('velocity_rms', 0) != 0 and self.gw.connected
        accel_active = len(self.gw.accel_buffer) > 0 and self.gw.connected

        self.stream_temp.classes(replace=f'stream-dot {"active" if temp_active else "inactive"}')
        self.stream_pres.classes(replace=f'stream-dot {"active" if pres_active else "inactive"}')
        self.stream_vib.classes(replace=f'stream-dot {"active" if vib_active else "inactive"}')
        self.stream_accel.classes(replace=f'stream-dot {"active" if accel_active else "inactive"}')

        # Values - use display values which have threshold correction for stationary sensor
        health = s.get('bearing_health_index', 100) or 100
        vel = s.get('velocity_rms', 0) or 0
        temp = s.get('temperature', 0) or 0
        pres = s.get('pressure', 0) or 0

        # Use display values (corrected for low-velocity noise) if available
        cf = s.get('crest_factor_display') or s.get('crest_factor', 0) or 0
        kurt = s.get('kurtosis_display') or s.get('kurtosis', 3) or 3
        temp_d = s.get('temperature_delta', 0) or 0

        # Health
        self.health_value.text = f'{health:.0f}'
        if health >= 80:
            self.health_value.classes(replace='metric-lg text-emerald-400')
            self.health_status.text = 'Healthy - No action needed'
        elif health >= 50:
            self.health_value.classes(replace='metric-lg text-amber-400')
            self.health_status.text = 'Attention - Schedule maintenance'
        else:
            self.health_value.classes(replace='metric-lg text-red-400')
            self.health_status.text = 'Critical - Immediate inspection'

        # Vibration
        self.vib_value.text = f'{vel:.2f}'
        zone = classify_iso_zone(vel)
        colors = {'A': 'text-emerald-400', 'B': 'text-lime-400', 'C': 'text-amber-400', 'D': 'text-red-400'}
        self.vib_value.classes(replace=f'metric-lg {colors[zone]}')
        self.vib_zone.text = f'Zone {zone} - {"Excellent" if zone=="A" else "Good" if zone=="B" else "Alert" if zone=="C" else "Danger"}'

        # Temp & Pressure
        self.temp_value.text = f'{temp:.1f}'
        self.temp_delta.text = f'+{temp_d:.1f}°C from baseline' if temp_d >= 0 else f'{temp_d:.1f}°C from baseline'
        self.pres_value.text = f'{pres:.0f}'

        # Crest Factor
        self.cf_value.text = f'{cf:.2f}'
        self.cf_bar.value = min(cf / 8, 1)
        if cf < 3:
            self.cf_status.text = 'Normal'
            self.cf_status.classes(replace='px-3 py-1 rounded-full text-sm font-medium bg-emerald-500/20 text-emerald-400')
            self.cf_bar.props('color=green')
        elif cf < 4.5:
            self.cf_status.text = 'Monitor'
            self.cf_status.classes(replace='px-3 py-1 rounded-full text-sm font-medium bg-amber-500/20 text-amber-400')
            self.cf_bar.props('color=amber')
        else:
            self.cf_status.text = 'Alert'
            self.cf_status.classes(replace='px-3 py-1 rounded-full text-sm font-medium bg-red-500/20 text-red-400')
            self.cf_bar.props('color=red')

        # Kurtosis
        self.kurt_value.text = f'{kurt:.2f}'
        self.kurt_bar.value = min((kurt - 2) / 8, 1) if kurt > 2 else 0
        if kurt < 4:
            self.kurt_status.text = 'Normal'
            self.kurt_status.classes(replace='px-3 py-1 rounded-full text-sm font-medium bg-emerald-500/20 text-emerald-400')
            self.kurt_bar.props('color=green')
        elif kurt < 6:
            self.kurt_status.text = 'Monitor'
            self.kurt_status.classes(replace='px-3 py-1 rounded-full text-sm font-medium bg-amber-500/20 text-amber-400')
            self.kurt_bar.props('color=amber')
        else:
            self.kurt_status.text = 'Damage'
            self.kurt_status.classes(replace='px-3 py-1 rounded-full text-sm font-medium bg-red-500/20 text-red-400')
            self.kurt_bar.props('color=red')

        # History
        ts = now.strftime('%H:%M:%S')
        self.history['time'].append(ts)
        self.history['velocity'].append(vel)
        self.history['health'].append(health)

        # Charts
        times = list(self.history['time'])
        self.vib_chart.options['xAxis']['data'] = times
        self.vib_chart.options['series'][0]['data'] = list(self.history['velocity'])
        vels = list(self.history['velocity'])
        if vels and max(vels) > 0:
            self.vib_chart.options['yAxis']['max'] = round(max(vels) * 1.3, 1)
        self.vib_chart.update()

        self.health_chart.options['xAxis']['data'] = times
        self.health_chart.options['series'][0]['data'] = list(self.history['health'])
        self.health_chart.update()

        # FFT Spectrum
        fft_data = s.get('fft_amplitudes', [])
        if fft_data:
            # Pad or trim to 64 bins
            fft_display = (fft_data + [0] * 64)[:64]
            self.fft_chart.options['series'][0]['data'] = fft_display
            self.fft_chart.update()

        # Recording
        if self.recording:
            self.data.append({
                'timestamp': now.isoformat(),
                'temperature': temp,
                'pressure': pres,
                'velocity_rms': vel,
                'crest_factor': cf,
                'kurtosis': kurt,
                'health_index': health,
                'iso_zone': zone,
                'fft_amplitudes': fft_data if fft_data else []
            })

        # Stats
        buf = len(self.gw.accel_buffer)
        sent = self.gw.stats.get('messages_sent', 0)
        rec = f' | Recording: {len(self.data)}' if self.recording else ''
        self.stats.text = f'Messages: {sent} | Buffer: {buf}/32 | Samples: {len(times)}{rec}'


def run_dashboard(gateway: ProteusGateway, port: int = 8080):
    import threading
    dash = Dashboard(gateway)

    @ui.page('/')
    def index():
        dash.build()

    def run_gw():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(gateway.run())
        except Exception as e:
            print(f"Gateway error: {e}")

    threading.Thread(target=run_gw, daemon=True).start()
    ui.run(port=port, title='PROTEUS Dashboard', dark=True, reload=False, favicon='🔧')


# Alias for backwards compatibility
ProteusDashboard = Dashboard

if __name__ == "__main__":
    gw = ProteusGateway()
    run_dashboard(gw, 8080)
