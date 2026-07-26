const screens = {
    landing: document.getElementById('screen-landing'),
    game: document.getElementById('screen-game'),
    result: document.getElementById('screen-result'),
};

const els = {
    videoInput: document.getElementById('video-input'),
    uploadZone: document.getElementById('upload-zone'),
    uploadPrompt: document.querySelector('.upload-prompt'),
    uploadLoading: document.querySelector('.upload-loading'),
    uploadDone: document.querySelector('.upload-done'),
    btnStart: document.getElementById('btn-start'),
    btnStop: document.getElementById('btn-stop'),
    btnRestart: document.getElementById('btn-restart'),
    errorLanding: document.getElementById('error-landing'),
    errorGame: document.getElementById('error-game'),
    videoFeed: document.getElementById('video-feed'),
    timer: document.getElementById('timer'),
    freezeOverlay: document.getElementById('freeze-overlay'),
    wallBadge: document.getElementById('wall-badge'),
    wallMatch: document.getElementById('wall-match'),
    feedbackFlash: document.getElementById('feedback-flash'),
    hudDance: document.getElementById('hud-dance'),
    hudBonus: document.getElementById('hud-bonus'),
    hudCombo: document.getElementById('hud-combo'),
    hudFeedback: document.getElementById('hud-feedback'),
    hudJoints: document.getElementById('hud-joints'),
    hudFps: document.getElementById('hud-fps'),
    resultGrade: document.getElementById('result-grade'),
    resultTotal: document.getElementById('result-total'),
    resultDance: document.getElementById('result-dance'),
    resultBonus: document.getElementById('result-bonus'),
    resultTotalRow: document.getElementById('result-total-row'),
};

let eventSource = null;
let referenceLoaded = false;
let readyCheckInterval = null;

// Poll server to see if the default (or uploaded) reference is ready.
function startReadyCheck() {
    if (readyCheckInterval) clearInterval(readyCheckInterval);
    readyCheckInterval = setInterval(async () => {
        try {
            const res = await fetch('/api/ready');
            const data = await res.json();
            if (data.error) {
                showError(els.errorLanding, data.error);
                clearInterval(readyCheckInterval);
                readyCheckInterval = null;
                return;
            }
            if (data.ready) {
                referenceLoaded = true;
                els.btnStart.disabled = false;
                els.uploadPrompt.classList.add('hidden');
                els.uploadLoading.classList.add('hidden');
                els.uploadDone.classList.remove('hidden');
                clearInterval(readyCheckInterval);
                readyCheckInterval = null;
            }
        } catch (err) {
            // server may not be up yet; keep polling
        }
    }, 800);
}

startReadyCheck();

function showScreen(name) {
    Object.values(screens).forEach(s => s.classList.remove('active'));
    screens[name].classList.add('active');
}

function showError(el, message) {
    if (!message) {
        el.classList.add('hidden');
        return;
    }
    el.textContent = message;
    el.classList.remove('hidden');
}

function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

// Upload handling
els.uploadZone.addEventListener('click', () => els.videoInput.click());

els.uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    els.uploadZone.style.borderColor = 'var(--orange)';
});

els.uploadZone.addEventListener('dragleave', () => {
    els.uploadZone.style.borderColor = '';
});

els.uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    els.uploadZone.style.borderColor = '';
    const files = e.dataTransfer.files;
    if (files.length) uploadVideo(files[0]);
});

els.videoInput.addEventListener('change', () => {
    if (els.videoInput.files.length) uploadVideo(els.videoInput.files[0]);
});

async function uploadVideo(file) {
    showError(els.errorLanding, '');
    els.uploadPrompt.classList.add('hidden');
    els.uploadLoading.classList.remove('hidden');
    els.uploadDone.classList.add('hidden');
    els.btnStart.disabled = true;
    referenceLoaded = false;

    const formData = new FormData();
    formData.append('video', file);

    try {
        const res = await fetch('/api/upload', { method: 'POST', body: formData });
        const data = await res.json();
        if (!data.ok) {
            throw new Error(data.error || 'Upload failed');
        }
        // Wait for background analysis to finish.
        startReadyCheck();
    } catch (err) {
        showError(els.errorLanding, err.message);
        els.uploadPrompt.classList.remove('hidden');
        els.uploadLoading.classList.add('hidden');
    }
}

// Start game
els.btnStart.addEventListener('click', async () => {
    showError(els.errorLanding, '');
    try {
        const res = await fetch('/api/start', { method: 'POST' });
        const data = await res.json();
        if (!data.ok) {
            throw new Error(data.error || 'Failed to start game');
        }
        startGame();
    } catch (err) {
        showError(els.errorLanding, err.message);
    }
});

function startGame() {
    showScreen('game');
    showError(els.errorGame, '');
    els.videoFeed.src = '/api/video?' + Date.now();
    connectStateStream();
}

function connectStateStream() {
    if (eventSource) eventSource.close();
    eventSource = new EventSource('/api/state');

    eventSource.onmessage = (e) => {
        const state = JSON.parse(e.data);
        updateHud(state);

        if (state.camera_error || state.model_error) {
            showError(els.errorGame, state.camera_error || state.model_error);
            stopGame();
        }

        if (state.finished) {
            eventSource.close();
            eventSource = null;
            showResult();
        }
    };

    eventSource.onerror = () => {
        if (eventSource) eventSource.close();
        eventSource = null;
    };
}

function updateHud(state) {
    els.timer.textContent = `${formatTime(state.elapsed)} / ${formatTime(state.duration)}`;
    els.hudDance.textContent = Math.round(state.dance_score);
    els.hudBonus.textContent = state.bonus_score;
    els.hudCombo.textContent = `x${state.combo}`;
    els.hudFeedback.textContent = state.feedback;
    els.hudFeedback.style.color = `rgb(${state.feedback_color[2]}, ${state.feedback_color[1]}, ${state.feedback_color[0]})`;
    els.hudJoints.textContent = `${state.pose_visible}/12`;
    els.hudFps.textContent = state.fps;

    if (state.wall_state) {
        els.wallBadge.classList.remove('hidden');
        els.wallMatch.textContent = `${Math.round(state.wall_match * 100)}%`;
    } else {
        els.wallBadge.classList.add('hidden');
    }

    if (state.freeze_active) {
        els.freezeOverlay.classList.remove('hidden');
    } else {
        els.freezeOverlay.classList.add('hidden');
    }

    flashFeedback(state.feedback);
}

let lastFeedback = '';
function flashFeedback(feedback) {
    if (!feedback || feedback === lastFeedback) return;
    lastFeedback = feedback;
    els.feedbackFlash.textContent = feedback;
    const color = feedback === 'MISS' ? 'var(--red)' : 'var(--green)';
    els.feedbackFlash.style.color = color;
    els.feedbackFlash.classList.remove('show');
    void els.feedbackFlash.offsetWidth;
    els.feedbackFlash.classList.add('show');
    setTimeout(() => els.feedbackFlash.classList.remove('show'), 800);
}

async function stopGame() {
    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }
    try {
        await fetch('/api/stop', { method: 'POST' });
    } catch (err) {
        console.error('Stop failed:', err);
    }
    els.videoFeed.src = '';
}

els.btnStop.addEventListener('click', () => {
    stopGame().then(showResult);
});

async function showResult() {
    await stopGame();
    try {
        const res = await fetch('/api/final');
        const data = await res.json();
        els.resultGrade.textContent = data.grade;
        els.resultTotal.textContent = data.total_score;
        els.resultDance.textContent = Math.round(data.dance_score);
        els.resultBonus.textContent = data.bonus_score;
        els.resultTotalRow.textContent = data.total_score;
    } catch (err) {
        console.error('Final state failed:', err);
    }
    showScreen('result');
}

els.btnRestart.addEventListener('click', () => {
    lastFeedback = '';
    els.uploadPrompt.classList.remove('hidden');
    els.uploadDone.classList.add('hidden');
    els.btnStart.disabled = !referenceLoaded;
    showScreen('landing');
    if (!referenceLoaded) startReadyCheck();
});
