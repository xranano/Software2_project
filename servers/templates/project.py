from .base import render_template

_CONTENT = '''
    <div class="container">
        <div class="video-section">
            <img src="/video" class="stream" id="videoStream">
        </div>

        <div class="controls-section">

            <div class="card">
                <div class="card-header">
                    Status
                    <span id="statusDot" style="width:8px;height:8px;border-radius:50%;
                        background:var(--accent-green);display:inline-block;"></span>
                </div>
                <div id="statusTable" style="font-size:12px;">
                    <div style="color:var(--text-muted);text-align:center;padding:12px 0;">
                        Waiting for data...
                    </div>
                </div>
            </div>

            <div class="card">
                <div class="card-header">Motion Control</div>
                <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px;">
                    <button class="button" onclick="startMotion()">Start</button>
                    <button class="button secondary" onclick="stopMotion()">Stop</button>
                    <button class="button secondary" onclick="resetSim()">Reset</button>
                </div>
                <div style="display:flex;gap:8px;align-items:center;">
                    <label for="trajectorySelect" style="font-size:13px;">Turn:</label>
                    <select id="trajectorySelect"
                        style="flex:1;padding:6px 8px;background:var(--bg-sidebar);
                               border:1px solid var(--border-color);border-radius:4px;
                               color:var(--text-primary);font-size:13px;">
                        <option value="straight">Straight</option>
                        <option value="left">Left</option>
                        <option value="right">Right</option>
                    </select>
                    <button class="button" onclick="setTrajectory()">Apply</button>
                </div>
                <div id="motionStatus" class="status"></div>
            </div>

            <div class="card">
                <div class="card-header">Send Command</div>
                <div style="display:flex;flex-direction:column;gap:8px;">
                    <div style="display:flex;gap:6px;">
                        <input id="cmdKey" type="text" placeholder="key"
                            style="flex:1;padding:6px 8px;background:var(--bg-sidebar);
                                   border:1px solid var(--border-color);border-radius:4px;
                                   color:var(--text-primary);font-size:13px;">
                        <input id="cmdValue" type="text" placeholder="value"
                            style="flex:2;padding:6px 8px;background:var(--bg-sidebar);
                                   border:1px solid var(--border-color);border-radius:4px;
                                   color:var(--text-primary);font-size:13px;">
                    </div>
                    <button class="button" onclick="sendCommand()">Send</button>
                    <div id="cmdStatus" class="status"></div>
                </div>
            </div>

        </div>
    </div>
'''

_EXTRA_CSS = '''
#statusTable .row {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid var(--border-color);
    align-items: baseline;
}
#statusTable .row:last-child { border-bottom: none; }
#statusTable .key  { color: var(--text-secondary); font-size: 12px; }
#statusTable .val  { color: var(--text-primary);   font-weight: 500; font-size: 13px; font-family: monospace; }
'''

_EXTRA_JS = '''
function refreshStatus() {
    fetch('/status')
        .then(r => r.json())
        .then(data => {
            const table = document.getElementById('statusTable');
            const keys = Object.keys(data);
            if (keys.length === 0) {
                table.innerHTML = '<div style="color:var(--text-muted);text-align:center;padding:12px 0;">No status yet</div>';
                return;
            }
            table.innerHTML = keys.map(k =>
                `<div class="row">
                    <span class="key">${k}</span>
                    <span class="val">${JSON.stringify(data[k])}</span>
                </div>`
            ).join('');
            document.getElementById('statusDot').style.background =
                data.game_over ? 'var(--accent-red)' : 'var(--accent-green)';
            if (data.trajectory) {
                document.getElementById('trajectorySelect').value = data.trajectory;
            }
        })
        .catch(() => {
            document.getElementById('statusDot').style.background = 'var(--accent-red)';
        });
}

function startMotion() {
    postJSON('/start', {})
        .then(r => showStatus('motionStatus', r.running ? 'Running' : 'Start failed', r.status === 'ok' ? 'success' : 'error'))
        .catch(e => showStatus('motionStatus', 'Error: ' + e, 'error'));
}

function stopMotion() {
    postJSON('/stop', {})
        .then(r => showStatus('motionStatus', 'Stopped', 'success'))
        .catch(e => showStatus('motionStatus', 'Error: ' + e, 'error'));
}

function resetSim() {
    const trajectory = document.getElementById('trajectorySelect').value;
    postJSON('/reset', {trajectory})
        .then(r => showStatus('motionStatus', 'Reset (' + trajectory + ')', 'success'))
        .catch(e => showStatus('motionStatus', 'Error: ' + e, 'error'));
}

function setTrajectory() {
    const value = document.getElementById('trajectorySelect').value;
    postJSON('/command', {key: 'trajectory', value})
        .then(r => showStatus('motionStatus', 'Trajectory: ' + value, r.status === 'ok' ? 'success' : 'error'))
        .catch(e => showStatus('motionStatus', 'Error: ' + e, 'error'));
}

function sendCommand() {
    const key   = document.getElementById('cmdKey').value.trim();
    const value = document.getElementById('cmdValue').value.trim();
    if (!key) {
        showStatus('cmdStatus', 'Key cannot be empty', 'error');
        return;
    }
    postJSON('/command', {key, value})
        .then(r => showStatus('cmdStatus', r.status === 'ok' ? 'Sent' : r.message, r.status === 'ok' ? 'success' : 'error'))
        .catch(e => showStatus('cmdStatus', 'Error: ' + e, 'error'));
}

document.getElementById('cmdValue').addEventListener('keydown', e => {
    if (e.key === 'Enter') sendCommand();
});

refreshStatus();
setInterval(refreshStatus, 500);
'''


def get_template(title='Project', subtitle='Real Duckiebot'):
    return render_template(
        title=title,
        subtitle=subtitle,
        content_html=_CONTENT,
        extra_css=_EXTRA_CSS,
        extra_js=_EXTRA_JS,
    )
