/**
 * app.js — ARIA v3.0
 * Modernized for LangGraph 0.4 Command(resume) HITL pattern.
 * Features: pipeline step tracking, output files panel, improved markdown,
 * session management, voice recording, SSE streaming.
 */

document.addEventListener('DOMContentLoaded', () => {
    const recorder = new AudioRecorder();

    // ── State ──────────────────────────────────────────────────────────────────
    let currentThreadId  = localStorage.getItem('aria_thread_id') || null;
    let chatHistory      = [];
    let isProcessing     = false;

    const DEFAULT_PATH   = './output';
    let recentPaths      = JSON.parse(localStorage.getItem('aria_recent_paths') || `["${DEFAULT_PATH}"]`);
    let currentOutputPath = localStorage.getItem('aria_current_path') || DEFAULT_PATH;

    // ── DOM refs ───────────────────────────────────────────────────────────────
    const chatMessages      = document.getElementById('chat-messages');
    const chatInput         = document.getElementById('chat-input');
    const micBtn            = document.getElementById('mic-btn');
    const uploadBtn         = document.getElementById('upload-btn');
    const fileInput         = document.getElementById('file-input');
    const sendBtn           = document.getElementById('send-btn');
    const newChatBtn        = document.getElementById('new-chat-btn');
    const sessionList       = document.getElementById('session-list');
    const fileList          = document.getElementById('file-list');
    const pathDropdown      = document.getElementById('path-dropdown');
    const addPathBtn        = document.getElementById('add-path-btn');
    const newPathWrapper    = document.getElementById('new-path-input-wrapper');
    const newPathInput      = document.getElementById('new-path-input');
    const savePathBtn       = document.getElementById('save-path-btn');
    const recorderViz       = document.getElementById('recorder-visualizer');
    const statusDot         = document.getElementById('status-dot');
    const statusText        = document.getElementById('status-text');
    const hitlOverlay       = document.getElementById('hitl-overlay');
    const hitlMessage       = document.getElementById('hitl-message');
    const confirmBtn        = document.getElementById('confirm-btn');
    const cancelBtn         = document.getElementById('cancel-btn');
    const pipelineBar       = document.getElementById('pipeline-bar');
    const pipelineIntent    = document.getElementById('pipeline-intent');
    const refreshFilesBtn   = document.getElementById('refresh-files-btn');
    const resetBtn          = document.getElementById('reset-session');

    // ── Init ───────────────────────────────────────────────────────────────────
    async function init() {
        renderPathDropdown();
        await loadSessions();
        await loadOutputFiles();
        if (currentThreadId) {
            await selectSession(currentThreadId);
        }
    }
    init();

    // ── Path management ────────────────────────────────────────────────────────
    function renderPathDropdown() {
        pathDropdown.innerHTML = '';
        recentPaths.forEach(p => {
            const opt = document.createElement('option');
            opt.value = p;
            opt.textContent = p === DEFAULT_PATH ? `${p} (default)` : p;
            if (p === currentOutputPath) opt.selected = true;
            pathDropdown.appendChild(opt);
        });
    }

    pathDropdown.addEventListener('change', e => {
        currentOutputPath = e.target.value;
        localStorage.setItem('aria_current_path', currentOutputPath);
        loadOutputFiles();
    });

    addPathBtn.addEventListener('click', () => {
        newPathWrapper.classList.toggle('hidden');
        addPathBtn.classList.toggle('active');
        if (!newPathWrapper.classList.contains('hidden')) newPathInput.focus();
    });

    savePathBtn.addEventListener('click', () => {
        const np = newPathInput.value.trim();
        if (!np) return;
        if (!recentPaths.includes(np)) {
            recentPaths.unshift(np);
            localStorage.setItem('aria_recent_paths', JSON.stringify(recentPaths));
        }
        currentOutputPath = np;
        localStorage.setItem('aria_current_path', currentOutputPath);
        renderPathDropdown();
        newPathInput.value = '';
        newPathWrapper.classList.add('hidden');
        addPathBtn.classList.remove('active');
        loadOutputFiles();
    });

    newPathInput.addEventListener('keydown', e => {
        if (e.key === 'Enter') savePathBtn.click();
        if (e.key === 'Escape') { newPathWrapper.classList.add('hidden'); addPathBtn.classList.remove('active'); }
    });

    // ── Output files ───────────────────────────────────────────────────────────
    async function loadOutputFiles() {
        try {
            const res = await fetch(`/api/output_files?folder=${encodeURIComponent(currentOutputPath)}`);
            if (!res.ok) { fileList.innerHTML = '<div class="empty-state">Could not load.</div>'; return; }
            const files = await res.json();
            if (!files.length) { fileList.innerHTML = '<div class="empty-state">No files yet.</div>'; return; }
            fileList.innerHTML = '';
            files.forEach(f => {
                const item = document.createElement('div');
                item.className = 'file-item';
                const icon = getFileIcon(f.name);
                const size = formatBytes(f.size);
                item.innerHTML = `<span class="file-icon">${icon}</span><span class="file-name" title="${f.name}">${f.name}</span><span class="file-size">${size}</span>`;
                fileList.appendChild(item);
            });
        } catch {
            fileList.innerHTML = '<div class="empty-state">Could not load.</div>';
        }
    }

    refreshFilesBtn.addEventListener('click', loadOutputFiles);

    function getFileIcon(name) {
        const ext = name.split('.').pop().toLowerCase();
        const map = { py:'🐍', js:'📜', ts:'📜', html:'🌐', css:'🎨', json:'📋', md:'📝', txt:'📄', sh:'⚡', yaml:'⚙️', yml:'⚙️' };
        return map[ext] || '📄';
    }

    function formatBytes(b) {
        if (b < 1024) return `${b}B`;
        if (b < 1024*1024) return `${(b/1024).toFixed(1)}KB`;
        return `${(b/1024/1024).toFixed(1)}MB`;
    }

    // ── Sessions ────────────────────────────────────────────────────────────────
    async function loadSessions() {
        try {
            const res = await fetch('/api/sessions');
            const sessions = await res.json();
            if (!sessions.length) { sessionList.innerHTML = '<div class="empty-state">No sessions yet.</div>'; return; }
            sessionList.innerHTML = '';
            sessions.forEach((s) => {
                const item = document.createElement('div');
                item.className = 'session-item';
                if (s.thread_id === currentThreadId) item.classList.add('active');

                const label = document.createElement('span');
                label.className = 'session-label';
                label.textContent = `Session ${s.thread_id.substring(0, 8)}…`;
                label.title = s.thread_id;
                label.onclick = () => selectSession(s.thread_id);

                const delBtn = document.createElement('button');
                delBtn.className = 'session-delete-btn';
                delBtn.title = 'Delete session';
                delBtn.innerHTML = '🗑';
                delBtn.onclick = async (e) => {
                    e.stopPropagation();
                    if (!confirm(`Delete session ${s.thread_id.substring(0, 8)}?`)) return;
                    try {
                        await fetch(`/api/sessions/${s.thread_id}`, { method: 'DELETE' });
                        if (s.thread_id === currentThreadId) {
                            currentThreadId = null;
                            localStorage.removeItem('aria_thread_id');
                            chatHistory = [];
                            showWelcome();
                        }
                        await loadSessions();
                    } catch (err) {
                        console.error('Delete failed', err);
                    }
                };

                item.appendChild(label);
                item.appendChild(delBtn);
                sessionList.appendChild(item);
            });
        } catch (e) { console.error('Failed to load sessions', e); }
    }

    async function selectSession(threadId) {
        currentThreadId = threadId;
        localStorage.setItem('aria_thread_id', threadId);
        document.querySelectorAll('.session-item').forEach(el => el.classList.toggle('active', el.title === threadId));
        setStatus('Loading…', 'busy');
        try {
            const res = await fetch(`/api/sessions/${threadId}`);
            if (!res.ok) throw new Error('Not found');
            const data = await res.json();
            chatHistory = data.messages || [];
            if (data.output_path && data.output_path !== currentOutputPath) {
                currentOutputPath = data.output_path;
                if (!recentPaths.includes(currentOutputPath)) { recentPaths.push(currentOutputPath); localStorage.setItem('aria_recent_paths', JSON.stringify(recentPaths)); }
                renderPathDropdown();
                await loadOutputFiles();
            }
            renderChat();
            if (data.is_interrupted && data.interrupt_data) showHITL(data.interrupt_data);
        } catch {
            addBotMsg('❌ Failed to load session.');
            currentThreadId = null;
            localStorage.removeItem('aria_thread_id');
        } finally { setStatus('Ready', 'ready'); }
    }

    newChatBtn.addEventListener('click', () => {
        currentThreadId = null;
        localStorage.removeItem('aria_thread_id');
        chatHistory = [];
        document.querySelectorAll('.session-item').forEach(el => el.classList.remove('active'));
        showWelcome();
    });

    function showWelcome() {
        chatMessages.innerHTML = `
            <div class="welcome-card">
                <div class="welcome-orb"></div>
                <h2 class="welcome-title">Hello, I'm ARIA</h2>
                <p class="welcome-subtitle">Your voice-controlled local AI agent</p>
                <div class="welcome-capabilities">
                    <div class="cap-item"><span>📄</span> Create &amp; write files</div>
                    <div class="cap-item"><span>💻</span> Generate code</div>
                    <div class="cap-item"><span>📝</span> Summarize text</div>
                    <div class="cap-item"><span>⚡</span> Run commands</div>
                    <div class="cap-item"><span>💬</span> General chat</div>
                </div>
                <p class="welcome-hint">Type a command below or press 🎙️ to speak</p>
            </div>`;
    }

    // ── Input ──────────────────────────────────────────────────────────────────
    function refreshInputState() {
        chatInput.style.height = 'auto';
        chatInput.style.height = Math.min(chatInput.scrollHeight, 180) + 'px';
        sendBtn.disabled = !chatInput.value.trim() || isProcessing;
    }

    chatInput.addEventListener('input', refreshInputState);
    chatInput.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); } });
    sendBtn.addEventListener('click', handleSend);

    // ── Mic / Upload ───────────────────────────────────────────────────────────
    micBtn.addEventListener('click', async () => {
        if (!recorder.isRecording) {
            setStatus('Requesting mic…', 'busy');
            const ok = await recorder.start();
            if (ok) {
                micBtn.classList.add('recording');
                recorderViz.classList.remove('hidden');
                setStatus('Recording…', 'busy');
            } else {
                alert('Mic access denied. Please allow microphone permissions.');
                setStatus('Ready', 'ready');
            }
        } else {
            const result = await recorder.stop();
            micBtn.classList.remove('recording');
            recorderViz.classList.add('hidden');
            if (result?.file) await processTranscription(result.file);
            else setStatus('Ready', 'ready');
        }
    });

    uploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', async () => {
        if (fileInput.files.length) {
            await processTranscription(fileInput.files[0]);
            fileInput.value = '';
        }
    });

    async function processTranscription(file) {
        setStatus('Transcribing…', 'busy');
        const orig = chatInput.placeholder;
        chatInput.placeholder = 'Transcribing voice… please wait';
        chatInput.disabled = true;
        try {
            const fd = new FormData();
            fd.append('audio', file);
            const res = await fetch('/api/transcribe', { method: 'POST', body: fd });
            if (!res.ok) throw new Error(await res.text());
            const data = await res.json();
            chatInput.value = data.transcript;
        } catch (e) {
            addBotMsg(`❌ Transcription error: ${e.message}`);
        } finally {
            chatInput.placeholder = orig;
            chatInput.disabled = false;
            refreshInputState();
            chatInput.focus();
            setStatus('Ready', 'ready');
        }
    }

    async function handleSend() {
        const text = chatInput.value.trim();
        if (!text || isProcessing) return;
        addUserMsg(text, 'Detecting…');
        chatInput.value = '';
        chatInput.style.height = 'auto';
        sendBtn.disabled = true;
        await processCommand(text);
    }

    // ── Core streaming ─────────────────────────────────────────────────────────
    async function processCommand(text) {
        isProcessing = true;
        setStatus('Thinking…', 'busy');
        showPipelineBar();
        setPipelineStep('intent');
        addLoadingMsg();

        try {
            const res = await fetch('/api/process_text_stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text, chat_history: chatHistory, thread_id: currentThreadId, output_path: currentOutputPath })
            });
            if (!res.ok) throw new Error(await res.text());

            const reader = res.body.getReader();
            const dec = new TextDecoder();
            let buf = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buf += dec.decode(value, { stream: true });
                const lines = buf.split('\n');
                buf = lines.pop();
                for (const line of lines) {
                    if (line.trim().startsWith('data: ')) {
                        try { handleUpdate(JSON.parse(line.trim().slice(6))); } catch {}
                    }
                }
            }
        } catch (e) {
            removeLoadingMsg();
            addBotMsg(`❌ Pipeline error: ${e.message}`);
        } finally {
            isProcessing = false;
            setStatus('Ready', 'ready');
            refreshInputState();
            setTimeout(hidePipelineBar, 1800);
        }
    }

    function handleUpdate(data) {
        const isNew = !currentThreadId;
        if (data.thread_id) {
            currentThreadId = data.thread_id;
            localStorage.setItem('aria_thread_id', currentThreadId);
        }
        if (data.messages) chatHistory = data.messages;

        // Pipeline step advancement
        if (data.transcript)      setPipelineStep('intent');
        if (data.detected_intent) { setPipelineStep('tool'); pipelineIntent.textContent = data.detected_intent; }
        if (data.action_taken)    setPipelineStep('done');

        // HITL interrupt
        if (data.is_interrupted && data.interrupt_data) {
            removeLoadingMsg();
            showHITL(data.interrupt_data);
            setStatus('Awaiting confirmation', 'busy');
            setPipelineStep('tool');
            return;
        }

        removeLoadingMsg();
        renderChat();
        if (isNew) loadSessions();

        // Mark done when we have a final AI response
        const lastMsg = (data.messages || []).at(-1);
        if (lastMsg?.role === 'assistant') {
            setPipelineStep('done');
            loadOutputFiles();
        }
    }

    // ── HITL ────────────────────────────────────────────────────────────────────
    function showHITL(data) {
        const summary = (data.actions_summary || '').replace(/\n/g, '<br>');
        hitlMessage.innerHTML = `
            <p>${data.message || '⚠️ Confirm these actions?'}</p>
            <pre>${summary}</pre>`;
        hitlOverlay.classList.remove('hidden');
    }

    async function handleConfirm(confirmed) {
        hitlOverlay.classList.add('hidden');
        setStatus('Executing…', 'busy');
        addLoadingMsg();
        try {
            const res = await fetch('/api/confirm', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ thread_id: currentThreadId, confirmed })
            });
            if (!res.ok) throw new Error(await res.text());
            const data = await res.json();
            removeLoadingMsg();
            if (data.messages) chatHistory = data.messages;
            renderChat();
            loadOutputFiles();
            setPipelineStep('done');
        } catch (e) {
            removeLoadingMsg();
            addBotMsg(`❌ Confirm error: ${e.message}`);
            setStatus('Error', 'error');
        } finally {
            setStatus('Ready', 'ready');
            setTimeout(hidePipelineBar, 1800);
        }
    }

    confirmBtn.addEventListener('click', () => handleConfirm(true));
    cancelBtn.addEventListener('click',  () => handleConfirm(false));

    // ── Pipeline bar ───────────────────────────────────────────────────────────
    const STEPS = ['stt', 'intent', 'tool', 'done'];

    function showPipelineBar() {
        pipelineBar.style.display = 'flex';
        STEPS.forEach(s => {
            const el = document.getElementById(`step-${s}`);
            if (el) { el.classList.remove('active', 'done'); }
        });
        pipelineIntent.textContent = '';
    }

    function hidePipelineBar() { pipelineBar.style.display = 'none'; }

    function setPipelineStep(active) {
        const idx = STEPS.indexOf(active);
        STEPS.forEach((s, i) => {
            const el = document.getElementById(`step-${s}`);
            if (!el) return;
            el.classList.remove('active', 'done');
            if (i < idx) el.classList.add('done');
            else if (i === idx) el.classList.add('active');
        });
    }

    // ── Rendering ──────────────────────────────────────────────────────────────
    function renderChat() {
        chatMessages.innerHTML = '';
        if (!chatHistory.length) { showWelcome(); return; }

        chatHistory.forEach(msg => {
            if (msg.role === 'user') {
                const div = document.createElement('div');
                div.className = 'chat-msg user';
                const badge = document.createElement('div');
                badge.className = 'msg-intent-badge';
                badge.textContent = msg.intent || '💬 General Chat';
                div.appendChild(badge);
                const content = document.createElement('div');
                content.textContent = msg.content;
                div.appendChild(content);
                chatMessages.appendChild(div);
            } else if (msg.role === 'assistant') {
                const div = document.createElement('div');
                div.className = 'chat-msg bot';
                div.innerHTML = renderMarkdown(msg.content);
                chatMessages.appendChild(div);
            }
        });
        scrollBottom();
    }

    function addUserMsg(text, intent) {
        const div = document.createElement('div');
        div.className = 'chat-msg user';
        const badge = document.createElement('div');
        badge.className = 'msg-intent-badge';
        badge.textContent = intent || '💬 General Chat';
        div.appendChild(badge);
        const c = document.createElement('div');
        c.textContent = text;
        div.appendChild(c);
        chatMessages.appendChild(div);
        scrollBottom();
    }

    function addBotMsg(text) {
        const div = document.createElement('div');
        div.className = 'chat-msg bot';
        div.innerHTML = renderMarkdown(text);
        chatMessages.appendChild(div);
        scrollBottom();
    }

    function addLoadingMsg() {
        removeLoadingMsg();
        const div = document.createElement('div');
        div.className = 'chat-msg bot loading';
        div.id = 'loading-msg';
        div.innerHTML = '<div class="typing-dots"><span></span><span></span><span></span></div>';
        chatMessages.appendChild(div);
        scrollBottom();
    }

    function removeLoadingMsg() {
        const el = document.getElementById('loading-msg');
        if (el) el.remove();
    }

    function scrollBottom() { chatMessages.scrollTop = chatMessages.scrollHeight; }

    // ── Markdown renderer (no deps) ────────────────────────────────────────────
    function renderMarkdown(text) {
        if (!text) return '';
        // Escape HTML first
        text = text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
        // Code blocks (must come before inline code)
        text = text.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) =>
            `<pre><code class="${lang}">${code.trim()}</code></pre>`);
        // Inline code
        text = text.replace(/`([^`\n]+)`/g, '<code>$1</code>');
        // Bold
        text = text.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
        // Italic
        text = text.replace(/\*([^*\n]+)\*/g, '<em>$1</em>');
        // HR
        text = text.replace(/^---$/gm, '<hr>');
        // Headers
        text = text.replace(/^### (.+)$/gm, '<strong>$1</strong>');
        text = text.replace(/^## (.+)$/gm, '<strong>$1</strong>');
        // Bullet lists
        text = text.replace(/^[\s]*[•\-\*] (.+)$/gm, '<li>$1</li>');
        text = text.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
        // Numbered lists
        text = text.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');
        // Newlines (not inside pre blocks)
        text = text.replace(/\n/g, '<br>');
        // Clean up <br> inside pre/ul
        text = text.replace(/<pre>(.*?)<\/pre>/gs, (_, inner) =>
            `<pre>${inner.replace(/<br>/g, '\n')}</pre>`);
        text = text.replace(/<ul>(.*?)<\/ul>/gs, (_, inner) =>
            `<ul>${inner.replace(/<br>/g, '')}</ul>`);
        return text;
    }

    // ── Status ─────────────────────────────────────────────────────────────────
    function setStatus(text, type) {
        statusText.textContent = text;
        statusDot.className = 'status-dot';
        if (type === 'busy')  statusDot.classList.add('busy');
        if (type === 'error') statusDot.classList.add('error');
    }

    // ── Reset ──────────────────────────────────────────────────────────────────
    resetBtn.addEventListener('click', () => {
        if (confirm('Clear local cache? (SQLite data is preserved)')) {
            localStorage.clear();
            location.reload();
        }
    });
});
