from flask import Flask, request, jsonify, send_from_directory, render_template_string
import os
import sys
import tempfile
import uuid

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入现有的音乐生成功能
from main import MusicGenerator
import utils
import model

app = Flask(__name__, static_folder='../frontend')

# 配置
UPLOAD_FOLDER = tempfile.gettempdir()
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 初始化音乐生成器实例
conditional_generator = None
inpainting_generator = None

# 创建条件生成模式的生成器
def create_conditional_generator():
    try:
        import main
        original_mode = main.mode
        main.mode = "conditional"
        
        # 设置条件生成模式的配置
        main.midi_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "midi_songs")
        main.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        main.model_name = "conditional_music_model"
        main.models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        
        # 确保输出目录存在
        main.output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
        os.makedirs(main.output_dir, exist_ok=True)
        
        generator = MusicGenerator()
        
        # 恢复原始配置
        main.mode = original_mode
        
        return generator
    except Exception as e:
        print(f"创建条件生成器时出错: {e}")
        return None

# 创建旋律补全模式的生成器
def create_inpainting_generator():
    try:
        import main
        original_mode = main.mode
        main.mode = "inpainting"
        
        # 设置旋律补全模式的配置
        main.midi_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fill_songs")
        main.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fill_data")
        main.model_name = "inpainting_music_model"
        main.models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        
        # 确保输出目录存在
        main.output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
        os.makedirs(main.output_dir, exist_ok=True)
        
        generator = MusicGenerator()
        
        # 恢复原始配置
        main.mode = original_mode
        
        return generator
    except Exception as e:
        print(f"创建旋律补全生成器时出错: {e}")
        return None

# 在应用启动时初始化生成器
try:
    conditional_generator = create_conditional_generator()
    inpainting_generator = create_inpainting_generator()
except Exception as e:
    print(f"初始化生成器时出错: {e}")

@app.route('/')
def index():
    # 提供前端页面
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/emotions', methods=['GET'])
def get_emotions():
    """获取可用的情绪类别"""
    try:
        # 返回默认情绪类别（根据项目结构中的midi_songs目录包含Q1-Q4）
        return jsonify({"emotions": ["Q1", "Q2", "Q3", "Q4"]})
    except Exception as e:
        print(f"获取情绪类别时出错: {e}")
        return jsonify({"emotions": ["Q1", "Q2", "Q3", "Q4"]})

@app.route('/api/generate', methods=['POST'])
def generate_music():
    """根据情绪生成音乐"""
    try:
        data = request.json
        emotion = data.get('emotion', 'Q1')
        num_notes = data.get('num_notes', 100)
        
        # 验证情绪 - 使用固定的Q1-Q4情绪类别
        valid_emotions = ["Q1", "Q2", "Q3", "Q4"]
        if emotion not in valid_emotions:
            return jsonify({"error": f"无效的情绪。可用情绪: {valid_emotions}"}), 400
        
        # 生成唯一的文件名
        output_filename = f"generated_{emotion}_{uuid.uuid4().hex[:8]}.mid"
        output_file = os.path.join(OUTPUT_DIR, output_filename)
        
        # 直接使用main.py中的方法，但需要正确设置模式
        import main
        original_mode = main.mode
        main.mode = "conditional"
        
        # 设置正确的路径
        main.midi_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "midi_songs")
        main.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        main.model_name = "conditional_music_model"
        main.models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        main.output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
        
        # 创建新的生成器实例
        generator = MusicGenerator()
        
        # 生成音乐
        result = generator.generate_conditional(
            emotion=emotion,
            num_notes=num_notes,
            output_file=output_file
        )
        
        # 恢复原始配置
        main.mode = original_mode
        
        if result:
            return jsonify({
                "success": True,
                "file_url": f"/download/{output_filename}",
                "filename": output_filename
            })
        else:
            return jsonify({"error": "音乐生成失败"}), 500
            
    except Exception as e:
        print(f"音乐生成错误: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/complete', methods=['POST'])
def complete_melody():
    """旋律补全功能"""
    try:
        # 检查是否有文件上传
        if 'file' not in request.files:
            return jsonify({"error": "没有文件上传"}), 400
        
        file = request.files['file']
        
        # 检查文件类型
        if not file.filename.endswith('.mid'):
            return jsonify({"error": "只支持.mid文件"}), 400
        
        # 保存上传的文件到临时目录
        temp_file_path = os.path.join(UPLOAD_FOLDER, f"temp_{uuid.uuid4().hex[:8]}.mid")
        file.save(temp_file_path)
        
        # 生成唯一的输出文件名
        output_filename = f"completed_{uuid.uuid4().hex[:8]}.mid"
        output_file = os.path.join(OUTPUT_DIR, output_filename)
        
        # 直接使用main.py中的方法，但需要正确设置模式
        import main
        original_mode = main.mode
        main.mode = "inpainting"
        
        # 设置正确的路径
        main.midi_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fill_songs")
        main.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fill_data")
        main.model_name = "inpainting_music_model"
        main.models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        main.output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
        
        # 创建新的生成器实例
        generator = MusicGenerator()
        
        # 补全旋律
        result = generator.complete_melody(
            input_midi_path=temp_file_path,
            output_file=output_file,
            num_completion_notes=100
        )
        
        # 删除临时文件
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        # 恢复原始配置
        main.mode = original_mode
        
        if result:
            return jsonify({
                "success": True,
                "file_url": f"/download/{output_filename}",
                "filename": output_filename
            })
        else:
            return jsonify({"error": "旋律补全失败"}), 500
            
    except Exception as e:
        print(f"旋律补全错误: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """下载生成的MIDI文件"""
    try:
        return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 404

@app.route('/css/<path:path>')
def send_css(path):
    return send_from_directory(os.path.join(app.static_folder, 'css'), path)

@app.route('/js/<path:path>')
def send_js(path):
    return send_from_directory(os.path.join(app.static_folder, 'js'), path)

if __name__ == '__main__':
    # 设置debug=False用于生产环境
    app.run(debug=False, host='0.0.0.0', port=5000)