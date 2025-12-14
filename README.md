# 自动音乐生成系统

## 项目简介

本项目是一个基于深度学习的自动音乐生成系统，支持两种核心功能：
- 情绪驱动的音乐生成：根据用户选择的情绪类别（Q1-Q4）生成相应风格的音乐
- 旋律补全创作：基于用户上传的MIDI种子文件，自动补全并生成完整旋律

## 功能特性

### 情绪驱动音乐生成
- 支持4种情绪类别：Q1、Q2、Q3、Q4
- 基于Transformer神经网络模型，根据情绪类别生成相应风格的MIDI音乐
- 提供生成结果的下载功能

### 旋律补全创作
- 支持上传MIDI种子文件(.mid格式)
- 自动补全并生成完整的音乐作品
- 提供补全结果的下载功能
- 具备智能填充和未知音符替换功能

## 技术栈

### 后端
- Python 3.12
- Flask：Web框架
- PyTorch：深度学习框架
- pretty_midi：MIDI文件处理库
- music21：音乐理论和MIDI处理
- NumPy：数据处理

### 前端
- HTML5/CSS3：页面结构和样式
- JavaScript：交互逻辑
- 响应式设计：适配不同设备屏幕

## 项目结构

```
Automatic_music_generation/
├── backend/              # 后端Flask服务
│   └── app.py           # 主应用入口和API实现
├── data/                # 数据处理和存储
│   ├── emotion_to_id.pkl    # 情绪类别映射
│   ├── note_to_int.pkl      # 音符到整数的映射
│   └── notes_with_emotion.pkl  # 带情绪标签的音符数据
├── fill_data/           # 旋律补全模式的数据
│   ├── emotion_to_id.pkl
│   ├── note_to_int.pkl
│   └── notes_with_emotion.pkl
├── fill_songs/          # 旋律补全训练用的MIDI文件
│   └── GiantMIDI-Piano/    # 通用MIDI数据集
├── frontend/            # 前端界面
│   ├── css/            # 样式文件
│   │   └── style.css
│   ├── js/             # JavaScript逻辑
│   │   └── main.js
│   └── index.html      # 主页面
├── midi_songs/          # 情绪分类的MIDI训练数据
│   ├── Q1/             # 欢快/兴奋情绪
│   ├── Q2/             # 活泼/动感情绪
│   ├── Q3/             # 悲伤/忧郁情绪
│   └── Q4/             # 平静/舒缓情绪
├── models/              # 训练好的模型
│   └── inpainting_music_model.pth  # 旋律补全模型
├── output/              # 生成结果输出目录
├── main.py              # 主程序和音乐生成核心逻辑
├── model.py             # Transformer模型定义
├── utils.py             # 工具函数
├── evaluate.py          # 模型评估脚本
├── requirements.txt     # Python依赖
└── README.md            # 项目说明文档
```

## 安装说明

### 1. 克隆项目

```bash
git clone <项目仓库URL>
cd Automatic_music_generation
```

### 2. 安装依赖

使用pip安装所需的Python依赖：

```bash
pip install -r requirements.txt
```

### 3. 准备模型文件

确保在`models/`目录下有预训练的模型文件：
- 旋律补全模型：`inpainting_music_model.pth`（必须）
- 情绪生成模型：`conditional_music_model.pth`（如果需要情绪生成功能）

## 使用方法

### 启动服务器

在后端目录下运行Flask应用：

```bash
cd backend
python app.py
```

服务将在`http://127.0.0.1:5000`启动。

### 访问系统

打开浏览器，访问`http://127.0.0.1:5000`即可使用系统。

### 情绪驱动音乐生成
1. 在情绪驱动音乐生成区域，选择一个情绪类别：
   - Q1：欢快/兴奋
   - Q2：活泼/动感
   - Q3：悲伤/忧郁
   - Q4：平静/舒缓
2. 可以调整生成的音符数量（滑块控制）
3. 点击"生成音乐"按钮
4. 等待生成完成后，点击"下载结果"获取MIDI文件

### 旋律补全创作
1. 在旋律补全创作区域，点击上传图标按钮选择MIDI种子文件
2. 点击"补全旋律"按钮
3. 等待补全完成后，点击"下载结果"获取补全后的MIDI文件

## 支持的MIDI格式

系统支持标准的MIDI文件格式(.mid)。对于旋律补全功能，建议上传的种子文件包含清晰的旋律线条，以获得更好的补全效果。系统会自动处理未知音符和不足长度的输入。

## 模型架构

系统使用基于Transformer的神经网络模型，主要特点：
- 包含多头注意力机制，能够捕捉音符间的长距离依赖关系
- 使用位置编码处理序列数据，保持时间顺序信息
- 编码器-解码器结构，支持序列到序列的生成
- 情绪嵌入机制，用于条件生成模式
- 可配置的温度参数控制生成的随机性
- 智能填充和未知音符替换功能，提高模型鲁棒性

## 注意事项

1. 生成音乐可能需要几秒钟到几分钟的时间，取决于服务器性能
2. 确保上传的MIDI文件格式正确，否则可能导致处理失败
3. 生成的音乐文件将保存在项目根目录的`output/`目录中
4. 情绪类别Q1-Q4对应不同的音乐风格，生成时会应用相应的温度、速度和音高范围参数

## 许可证

[MIT License](LICENSE)

## 联系方式

如有问题或建议，请联系项目维护者。