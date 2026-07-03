// AI学习助手前端JavaScript代码

class AILearningAssistant {
    constructor() {
        this.messagesContainer = document.getElementById('messagesContainer');
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.loading = document.getElementById('loading');
        this.settingsPanel = document.getElementById('settingsPanel');
        this.settingsToggle = document.getElementById('settingsToggle');
        this.temperatureSlider = document.getElementById('temperature');
        this.temperatureValue = document.getElementById('temperatureValue');
        
        this.init();
    }
    
    init() {
        this.bindEvents();
        this.setupAutoResize();
        this.updateTemperatureValue();
    }
    
    bindEvents() {
        // 发送消息事件
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // 设置面板切换
        this.settingsToggle.addEventListener('click', () => {
            this.settingsPanel.classList.toggle('active');
        });
        
        // 温度滑块事件
        this.temperatureSlider.addEventListener('input', () => {
            this.updateTemperatureValue();
        });

        // 意图预测事件
        const intentBtn = document.getElementById('intentButton');
        const intentInput = document.getElementById('intentInput');
        const intentResult = document.getElementById('intentResult');
        intentBtn.addEventListener('click', async () => {
            const text = intentInput.value.trim();
            if (!text) return;
            intentBtn.disabled = true;
            intentResult.textContent = '预测中...';
            try {
                const resp = await fetch('/api/intent/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text })
                });
                const data = await resp.json();
                if (!resp.ok) throw new Error(data.error || '请求失败');
                let html = `<strong>预测类别：</strong> ${data.label}`;
                if (data.topk) {
                    html += '<br><strong>Top-5概率：</strong> ';
                    html += '<ul style="margin-top: 0.5rem">' + data.topk.map(it => `<li>${it.label}: ${(it.prob * 100).toFixed(1)}%</li>`).join('') + '</ul>';
                }
                intentResult.innerHTML = html;
            } catch (err) {
                intentResult.textContent = `错误：${err.message}`;
            } finally {
                intentBtn.disabled = false;
            }
        });
    }
    
    setupAutoResize() {
        this.messageInput.addEventListener('input', () => {
            this.messageInput.style.height = 'auto';
            this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 120) + 'px';
        });
    }
    
    updateTemperatureValue() {
        this.temperatureValue.textContent = this.temperatureSlider.value;
    }
    
    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message) return;
        
        // 清空输入框
        this.messageInput.value = '';
        this.messageInput.style.height = 'auto';
        
        // 添加用户消息
        this.addMessage(message, 'user');
        
        // 显示加载动画
        this.showLoading();
        
        // 预创建一个助手消息用于流式追加
        const assistantDiv = this.createAssistantMessageContainer();
        
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 60000);

            const response = await fetch('/api/chat/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: message,
                    max_tokens: document.getElementById('maxTokens').value,
                    temperature: this.temperatureSlider.value
                }),
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            if (!response.ok || !response.body) {
                const data = await response.json().catch(() => ({}));
                throw new Error(data.error || 'Request failed');
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let buffer = '';

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                let idx;
                while ((idx = buffer.indexOf('\n\n')) !== -1) {
                    const chunk = buffer.slice(0, idx);
                    buffer = buffer.slice(idx + 2);
                    if (chunk.startsWith('data:')) {
                        const jsonStr = chunk.replace(/^data:\s*/, '');
                        try {
                            const obj = JSON.parse(jsonStr);
                            if (obj.delta) {
                                this.appendToAssistantMessage(assistantDiv, obj.delta);
                            }
                        } catch (e) { /* ignore parse errors */ }
                    }
                }
            }
        } catch (error) {
            console.error('Failed to send message:', error);
            let errorMessage = 'An error occurred while processing the request';
            
            if (error.name === 'AbortError') {
                errorMessage = 'Request timed out, please try again later';
            } else if (error.message.includes('网络')) {
                errorMessage = 'Network connection failed, please check your connection';
            } else if (error.message.includes('超时')) {
                errorMessage = 'Request timed out, please try again later';
            } else if (error.message.includes('频率')) {
                errorMessage = 'Too many requests, please try again later';
            } else {
                errorMessage = error.message;
            }
            
            this.addMessage(`抱歉，发生了错误：${errorMessage}`, 'assistant');
        } finally {
            this.hideLoading();
        }
    }
    
    addMessage(content, sender) {
        if (this.messagesContainer.querySelector('.welcome-message')) {
            this.messagesContainer.innerHTML = '';
        }
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = sender === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        const formattedContent = this.formatMessage(content);
        messageContent.innerHTML = formattedContent;
        if (sender === 'user') {
            messageDiv.appendChild(messageContent);
            messageDiv.appendChild(avatar);
        } else {
            messageDiv.appendChild(avatar);
            messageDiv.appendChild(messageContent);
        }
        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }

    createAssistantMessageContainer() {
        if (this.messagesContainer.querySelector('.welcome-message')) {
            this.messagesContainer.innerHTML = '';
        }
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant';
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = '<i class="fas fa-robot"></i>';
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageContent.innerHTML = '';
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(messageContent);
        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
        return messageDiv;
    }

    appendToAssistantMessage(assistantDiv, deltaText) {
        const contentDiv = assistantDiv.querySelector('.message-content');
        // 直接追加文本，再做最简清洗；可按需改成增量Markdown渲染
        contentDiv.innerHTML += this.formatMessage(deltaText);
        this.scrollToBottom();
    }
    
    formatMessage(content) {
        // 简单的Markdown格式化
        return content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')  // 粗体
            .replace(/\*(.*?)\*/g, '<em>$1</em>')              // 斜体
            .replace(/`(.*?)`/g, '<code>$1</code>')            // 行内代码
            .replace(/\n/g, '<br>')                           // 换行
            .replace(/^#\s+(.*)$/gm, '<h3>$1</h3>')           // 标题
            .replace(/^-\s+(.*)$/gm, '<li>$1</li>')           // 列表项
            .replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');        // 列表
    }
    
    scrollToBottom() {
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }
    
    showLoading() {
        this.sendButton.disabled = true;
        this.loading.classList.add('active');
    }
    
    hideLoading() {
        this.sendButton.disabled = false;
        this.loading.classList.remove('active');
    }
}

// 设置建议问题
function setSuggestion(question) {
    document.getElementById('messageInput').value = question;
    document.getElementById('messageInput').focus();
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    new AILearningAssistant();
});

// 健康检查
async function checkHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        
        if (!data.deepseek_configured) {
            console.warn('DeepSeek API not configured');
        }
    } catch (error) {
        console.error('Health check failed:', error);
    }
}

// 页面加载时执行健康检查
window.addEventListener('load', checkHealth);