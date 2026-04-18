/**
 * app.js
 * Redesigned for Chat-Centric ARIA with Session Persistence, Integrated Custom Output Paths, and Voice Output.
 */

document.addEventListener('DOMContentLoaded', () => {
    const recorder = new AudioRecorder();
    let currentThreadId = localStorage.getItem('aria_thread_id');
    let chatHistory = [];
    let isProcessing = false;

    // Output Path Management
    const DEFAULT_PATH = './output';
    let recentPaths = JSON.parse(localStorage.getItem('aria_recent_paths') || `["${DEFAULT_PATH}"]`);
    let currentOutputPath = localStorage.getItem('aria_current_path') || DEFAULT_PATH;

    // UI Elements
    const chatMessages = document.getElementById('chat-messages');
    const chatInput = document.getElementById('chat-input');
    const micBtn = document.getElementById('mic-btn');
    const uploadBtn = document.getElementById('upload-btn');
    const fileInput = document.getElementById('file-input');
    const sendBtn = document.getElementById('send-btn');
    const resetBtn = document.getElementById('reset-session');
    const newChatBtn = document.getElementById('new-chat-btn');
    const sessionList = document.getElementById('session-list');
    
    const pathDropdown = document.getElementById('path-dropdown');
    const addPathBtn = document.getElementById('add-path-btn');
    const newPathWrapper = document.getElementById('new-path-input-wrapper');
    const newPathInput = document.getElementById('new-path-input');
    const savePathBtn = document.getElementById('save-path-btn');
    const ttsToggle = document.getElementById('tts-toggle');
    
    const recorderVisualizer = document.getElementById('recorder-visualizer');
    const appStatusText = document.querySelector('.status-text');
    const appStatusDot = document.querySelector('.status-badge .status-dot');

    const hitlOverlay = document.getElementById('hitl-overlay');
    const hitlMessage = document.getElementById('hitl-message');
    const confirmBtn = document.getElementById('confirm-btn');
    const cancelBtn = document.getElementById('cancel-btn');

    const stepStt = document.getElementById('step-stt');
    const stepIntent = document.getElementById('step-intent');
    const stepTools = document.getElementById('step-tools');

    // --- Initialization ---

    async function init() {
        renderPathDropdown();
        await loadSessions();
        if (currentThreadId) {
            await selectSession(currentThreadId);
        } else {
            showWelcome();
        }
        
        // Restore TTS setting
        const savedTTS = localStorage.getItem('aria_tts_enabled');
        if (savedTTS !== null) {
            ttsToggle.checked = savedTTS === 'true';
        }
    }

    init();

    ttsToggle.addEventListener('change', () => {
        localStorage.setItem('aria_tts_enabled', ttsToggle.checked);
    });

    // --- Path Management ---

    function renderPathDropdown() {
        pathDropdown.innerHTML = '';
        recentPaths.forEach(path => {
            const opt = document.createElement('option');
            opt.value = path;
            opt.textContent = path === DEFAULT_PATH ? `${path} (default)` : path;
            if (path === currentOutputPath) opt.selected = true;
            pathDropdown.appendChild(opt);
        });
    }

    pathDropdown.addEventListener('change', (e) => {
        currentOutputPath = e.target.value;
        localStorage.setItem('aria_current_path', currentOutputPath);
    });

    addPathBtn.addEventListener('click', () => {
        newPathWrapper.classList.toggle('hidden');
        addPathBtn.classList.toggle('active');
        if (!newPathWrapper.classList.contains('hidden')) {
            newPathInput.focus();
        }
    });

    savePathBtn.addEventListener('click', () => {
        const newPath = newPathInput.value.trim();
        if (newPath) {
            if (!recentPaths.includes(newPath)) {
                recentPaths.unshift(newPath);
                localStorage.setItem('aria_recent_paths', JSON.stringify(recentPaths));
            }
            currentOutputPath = newPath;
            localStorage.setItem('aria_current_path', currentOutputPath);
            renderPathDropdown();
            newPathInput.value = '';
            newPathWrapper.classList.add('hidden');
            addPathBtn.classList.remove('active');
        }
    });

    newPathInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            savePathBtn.click();
        }
        if (e.key === 'Escape') {
            newPathWrapper.classList.add('hidden');
            addPathBtn.classList.remove('active');
        }
    });

    // --- Session Management ---

    async function loadSessions() {
        try {
            const response = await fetch('/api/sessions');
            const sessions = await response.json();
            
            if (sessions.length === 0) {
                sessionList.innerHTML = '<div class="empty-state">No sessions found.</div>';
                return;
            }

            sessionList.innerHTML = '';
            sessions.forEach(s => {
                const item = document.createElement('div');
                item.className = 'session-item';
                if (s.thread_id === currentThreadId) item.classList.add('active');
                
                item.textContent = `Session ${s.thread_id.substring(0, 8)}`;
                item.title = s.thread_id;
                
                item.onclick = () => selectSession(s.thread_id);
                sessionList.appendChild(item);
            });
        } catch (err) {
            console.error("Failed to load sessions", err);
        }
    }

    async function selectSession(threadId) {
        currentThreadId = threadId;
        localStorage.setItem('aria_thread_id', threadId);
        
        document.querySelectorAll('.session-item').forEach(el => {
            el.classList.toggle('active', el.title === threadId);
        });

        setAppStatus("Loading session...", "busy");
        try {
            const response = await fetch(`/api/sessions/${threadId}`);
            if (!response.ok) throw new Error("Session not found");
            const data = await response.json();
            
            chatHistory = data.messages;
            if (data.output_path) {
                currentOutputPath = data.output_path;
                if (!recentPaths.includes(currentOutputPath)) {
                    recentPaths.push(currentOutputPath);
                    localStorage.setItem('aria_recent_paths', JSON.stringify(recentPaths));
                }
                renderPathDropdown();
            }
            
            updateChatDisplay();
            
            if (data.is_interrupted) {
                showHITL(data.interrupt_data);
            }
        } catch (err) {
            console.error(err);
            addBotMessage("❌ Failed to load session history.");
            currentThreadId = null;
            localStorage.removeItem('aria_thread_id');
        } finally {
            setAppStatus("Ready", "ready");
        }
    }

    function showWelcome() {
        chatMessages.innerHTML = `
            <div class="chat-msg bot">
                Hello! I am ARIA. How can I help you today?
            </div>
        `;
    }

    newChatBtn.addEventListener('click', () => {
        currentThreadId = null;
        localStorage.removeItem('aria_thread_id');
        chatHistory = [];
        updateChatDisplay();
        showWelcome();
        document.querySelectorAll('.session-item').forEach(el => el.classList.remove('active'));
    });

    // --- Input Handling ---

    function refreshInputState() {
        chatInput.style.height = 'auto';
        chatInput.style.height = (chatInput.scrollHeight) + 'px';
        sendBtn.disabled = !chatInput.value.trim() || isProcessing;
    }

    chatInput.addEventListener('input', refreshInputState);

    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });

    micBtn.addEventListener('click', async () => {
        if (!recorder.isRecording) {
            const success = await recorder.start();
            if (success) {
                micBtn.classList.add('recording');
                recorderVisualizer.classList.remove('hidden');
                setAppStatus("Recording...", "busy");
            }
        } else {
            const result = await recorder.stop();
            micBtn.classList.remove('recording');
            recorderVisualizer.classList.add('hidden');
            
            if (result) {
                await processTranscription(result.file);
            } else {
                setAppStatus("Ready", "ready");
            }
        }
    });

    uploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', async () => {
        if (fileInput.files.length) {
            const file = fileInput.files[0];
            await processTranscription(file);
            fileInput.value = '';
        }
    });

    async function processTranscription(file) {
        setAppStatus("Transcribing...", "busy");
        updateStep(stepStt, 'active');
        
        const originalPlaceholder = chatInput.placeholder;
        chatInput.placeholder = "Transcribing voice... please wait";
        chatInput.disabled = true;

        try {
            const transcript = await transcribeAudio(file);
            chatInput.value = transcript;
            updateStep(stepStt, 'complete');
        } catch (err) {
            addBotMessage(`❌ Transcription error: ${err.message}`);
            updateStep(stepStt, 'error');
        } finally {
            chatInput.placeholder = originalPlaceholder;
            chatInput.disabled = false;
            refreshInputState();
            chatInput.focus();
            setAppStatus("Ready", "ready");
        }
    }

    async function transcribeAudio(file) {
        const formData = new FormData();
        formData.append('audio', file);
        const response = await fetch('/api/transcribe', {
            method: 'POST',
            body: formData
        });
        if (!response.ok) throw new Error(await response.text());
        const data = await response.json();
        return data.transcript;
    }

    async function handleSend() {
        const text = chatInput.value.trim();
        if (!text || isProcessing) return;

        addUserMessage(text);
        chatInput.value = '';
        chatInput.style.height = 'auto';
        sendBtn.disabled = true;

        await processCommand(text);
    }

    async function processCommand(text) {
        isProcessing = true;
        setAppStatus("Processing...", "busy");
        updatePipelineUI('complete', 'active', '');

        try {
            const response = await fetch('/api/process_text', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: text,
                    chat_history: chatHistory,
                    thread_id: currentThreadId,
                    output_path: currentOutputPath
                })
            });

            if (!response.ok) throw new Error(await response.text());

            const result = await response.json();
            handlePipelineResult(result);
        } catch (err) {
            console.error(err);
            addBotMessage(`❌ Pipeline error: ${err.message}`);
            updatePipelineUI('complete', 'error', 'error');
        } finally {
            isProcessing = false;
            setAppStatus("Ready", "ready");
            refreshInputState();
        }
    }

    function handlePipelineResult(result) {
        const isNewSession = !currentThreadId;
        currentThreadId = result.thread_id;
        localStorage.setItem('aria_thread_id', result.thread_id);
        chatHistory = result.messages;
        
        if (result.is_interrupted) {
            updatePipelineUI('complete', 'complete', 'active');
            showHITL(result.interrupt_data);
        } else {
            updatePipelineUI('complete', 'complete', 'complete');
        }

        if (result.error) {
            addBotMessage(`⚠️ ${result.error}`);
        }

        const lastMessage = chatHistory[chatHistory.length - 1];
        if (lastMessage && lastMessage.role === 'assistant') {
            speak(lastMessage.content);
        }

        updateChatDisplay();
        if (isNewSession) loadSessions();
    }

    // --- Voice Output (TTS) ---

    async function speak(text) {
        if (!ttsToggle.checked || !text) return;
        
        try {
            const response = await fetch('/api/tts', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            });
            
            if (!response.ok) throw new Error("TTS failed");
            
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const audio = new Audio(url);
            audio.play();
        } catch (err) {
            console.error("TTS Error:", err);
        }
    }

    // --- HITL Handling ---

    function showHITL(data) {
        hitlMessage.innerHTML = `<p>${data.message}</p><pre style="font-size:0.7rem; margin-top:1rem; opacity:0.7;">${data.actions_summary}</pre>`;
        hitlOverlay.classList.remove('hidden');
    }

    confirmBtn.addEventListener('click', () => handleConfirm(true));
    cancelBtn.addEventListener('click', () => handleConfirm(false));

    async function handleConfirm(confirmed) {
        hitlOverlay.classList.add('hidden');
        setAppStatus("Executing...", "busy");
        updateStep(stepTools, 'active');

        try {
            const response = await fetch('/api/confirm', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ thread_id: currentThreadId, confirmed })
            });

            if (!response.ok) throw new Error(await response.text());

            const result = await response.json();
            handlePipelineResult(result);
        } catch (err) {
            addBotMessage(`❌ Execution error: ${err.message}`);
            updateStep(stepTools, 'error');
        } finally {
            setAppStatus("Ready", "ready");
        }
    }

    // --- UI Helpers ---

    function addUserMessage(text) {
        const div = document.createElement('div');
        div.className = 'chat-msg user';
        div.textContent = text;
        chatMessages.appendChild(div);
        scrollToBottom();
    }

    function addBotMessage(text) {
        const div = document.createElement('div');
        div.className = 'chat-msg bot';
        div.innerHTML = formatResult(text);
        chatMessages.appendChild(div);
        scrollToBottom();
    }

    function updateChatDisplay() {
        chatMessages.innerHTML = '';
        chatHistory.forEach(msg => {
            if (msg.role === 'user' || msg.role === 'assistant') {
                const div = document.createElement('div');
                div.className = `chat-msg ${msg.role === 'user' ? 'user' : 'bot'}`;
                div.innerHTML = formatResult(msg.content);
                chatMessages.appendChild(div);
            }
        });
        scrollToBottom();
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function setAppStatus(text, type) {
        appStatusText.textContent = text;
        appStatusDot.className = 'status-dot';
        if (type === 'busy') appStatusDot.classList.add('busy');
        if (type === 'error') appStatusDot.classList.add('error');
    }

    function updatePipelineUI(stt, intent, tools) {
        updateStep(stepStt, stt);
        updateStep(stepIntent, intent);
        updateStep(stepTools, tools);
    }

    function updateStep(el, status) {
        el.className = 'step-status';
        if (status) el.classList.add(status);
    }

    function formatResult(text) {
        if (!text) return '';
        
        // Code Blocks
        text = text.replace(/```([\w]*)\n([\s\S]*?)```/g, '<pre><code class="$1">$2</code></pre>');
        
        // Inline Code
        text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
        
        // Bold
        text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        
        // Bullet Points
        text = text.replace(/^\s*[\-\*]\s+(.*)/gm, '<li>$1</li>');
        
        // Group list items
        text = text.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
        
        // Lines
        text = text.replace(/\n/g, '<br>');
        
        return text;
    }

    resetBtn.addEventListener('click', () => {
        if (confirm("Clear local cache and restart? This will not delete SQLite data.")) {
            localStorage.clear();
            location.reload();
        }
    });
});
