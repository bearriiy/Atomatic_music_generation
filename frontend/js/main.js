document.addEventListener('DOMContentLoaded', function() {
    // DOM元素
    const emotionSelect = document.getElementById('emotion-select');
    const numNotesSlider = document.getElementById('num-notes');
    const numNotesValue = document.getElementById('num-notes-value');
    const generateBtn = document.getElementById('generate-btn');
    const midiUpload = document.getElementById('midi-upload');
    const completeBtn = document.getElementById('complete-btn');
    const resultSection = document.getElementById('result-section');
    const resultMessage = document.getElementById('result-message');
    const resultContainer = document.getElementById('result-container');
    const fileInfo = document.getElementById('file-info');
    const downloadBtn = document.getElementById('download-btn');
    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingMessage = document.getElementById('loading-message');

    let currentFileUrl = '';
    let currentFilename = '';

    // 初始化页面
    init();

    function init() {
        // 加载可用的情绪类别
        loadEmotions();
        
        // 设置事件监听器
        setupEventListeners();
    }

    function setupEventListeners() {
        // 音符数量滑块事件
        numNotesSlider.addEventListener('input', function() {
            numNotesValue.textContent = this.value;
        });

        // 生成音乐按钮事件
        generateBtn.addEventListener('click', generateMusic);

        // MIDI文件上传事件
        midiUpload.addEventListener('change', function() {
            if (this.files && this.files[0]) {
                const fileName = this.files[0].name;
                // 更新提示文本显示文件名
                const selectedFileElement = document.getElementById('selected-file');
                selectedFileElement.textContent = `已选择: ${fileName}`;
                // 启用补全按钮
                completeBtn.disabled = false;
            } else {
                // 恢复默认提示文本
                const selectedFileElement = document.getElementById('selected-file');
                selectedFileElement.textContent = '请上传包含初始旋律的.mid文件';
                completeBtn.disabled = true;
            }
        });

        // 补全旋律按钮事件
        completeBtn.addEventListener('click', completeMelody);

        // 下载按钮事件
        downloadBtn.addEventListener('click', downloadFile);
    }

    // 加载可用的情绪类别
    async function loadEmotions() {
        try {
            const response = await fetch('/api/emotions');
            const data = await response.json();
            
            if (data.emotions && data.emotions.length > 0) {
                // 清空现有选项
                emotionSelect.innerHTML = '';
                
                // 添加新选项
                data.emotions.forEach(emotion => {
                    const option = document.createElement('option');
                    option.value = emotion;
                    option.textContent = getEmotionLabel(emotion);
                    emotionSelect.appendChild(option);
                });
            }
        } catch (error) {
            console.error('加载情绪类别失败:', error);
            // 使用默认情绪选项
        }
    }

    // 获取情绪标签
    function getEmotionLabel(emotion) {
        const labels = {
            'Q1': 'Q1 (欢快/兴奋)',
            'Q2': 'Q2 (活泼/动感)',
            'Q3': 'Q3 (悲伤/忧郁)',
            'Q4': 'Q4 (平静/舒缓)',
            'GiantMIDI-Piano': 'GiantMIDI-Piano (通用)'
        };
        return labels[emotion] || emotion;
    }

    // 显示加载覆盖层
    function showLoading(message = '正在处理，请稍候...') {
        loadingMessage.textContent = message;
        loadingOverlay.classList.remove('hidden');
    }

    // 隐藏加载覆盖层
    function hideLoading() {
        loadingOverlay.classList.add('hidden');
    }

    // 显示消息
    function showMessage(message, type = 'info') {
        resultMessage.textContent = message;
        resultMessage.className = `message ${type}`;
        resultMessage.classList.remove('hidden');
    }

    // 隐藏消息
    function hideMessage() {
        resultMessage.classList.add('hidden');
    }

    // 显示结果
    function showResult(filename) {
        fileInfo.textContent = filename;
        resultContainer.classList.remove('hidden');
    }

    // 隐藏结果
    function hideResult() {
        resultContainer.classList.add('hidden');
    }

    // 生成音乐
    async function generateMusic() {
        const emotion = emotionSelect.value;
        const numNotes = parseInt(numNotesSlider.value);

        try {
            showLoading(`正在生成${getEmotionLabel(emotion)}情绪的音乐...`);
            hideMessage();
            hideResult();

            const response = await fetch('/api/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ emotion, num_notes: numNotes })
            });

            const data = await response.json();
            hideLoading();

            if (data.success) {
                currentFileUrl = data.file_url;
                currentFilename = data.filename;
                showMessage('音乐生成成功！', 'success');
                showResult(currentFilename);
            } else {
                showMessage(`生成失败: ${data.error || '未知错误'}`, 'error');
            }
        } catch (error) {
            hideLoading();
            showMessage(`生成过程中出错: ${error.message}`, 'error');
            console.error('生成音乐失败:', error);
        }
    }

    // 补全旋律
    async function completeMelody() {
        const file = midiUpload.files[0];
        
        if (!file) {
            showMessage('请先选择一个MIDI文件', 'error');
            return;
        }

        try {
            showLoading('正在分析和补全旋律...');
            hideMessage();
            hideResult();

            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch('/api/complete', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            hideLoading();

            if (data.success) {
                currentFileUrl = data.file_url;
                currentFilename = data.filename;
                showMessage('旋律补全成功！', 'success');
                showResult(currentFilename);
            } else {
                showMessage(`补全失败: ${data.error || '未知错误'}`, 'error');
            }
        } catch (error) {
            hideLoading();
            showMessage(`补全过程中出错: ${error.message}`, 'error');
            console.error('补全旋律失败:', error);
        }
    }

    // 下载文件
    function downloadFile() {
        if (currentFileUrl) {
            // 创建临时链接进行下载
            const link = document.createElement('a');
            link.href = currentFileUrl;
            link.download = currentFilename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    }

    // 添加键盘快捷键
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + G 生成音乐
        if ((e.ctrlKey || e.metaKey) && e.key === 'g') {
            e.preventDefault();
            generateBtn.click();
        }
        // Ctrl/Cmd + C 补全旋律
        else if ((e.ctrlKey || e.metaKey) && e.key === 'c') {
            e.preventDefault();
            if (!completeBtn.disabled) {
                completeBtn.click();
            }
        }
        // Ctrl/Cmd + D 下载文件
        else if ((e.ctrlKey || e.metaKey) && e.key === 'd') {
            e.preventDefault();
            if (!downloadBtn.disabled && currentFileUrl) {
                downloadFile();
            }
        }
    });
});