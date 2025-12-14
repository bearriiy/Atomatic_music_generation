
import os
import pickle
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import MusicDataset, MusicGeneratorUtils
from model import TransformerMusicModel
import sys
import os
# ==========================
# 全局配置参数（统一管理）
# ==========================

# cd backend; python app.py

# --- 运行模式选择 ---
# 可选值: "conditional" (情绪生成), "inpainting" (旋律补全)
mode = "conditional"  # <<< 修改这里切换模式

# --- 路径配置（根据 mode 动态设置）---
if mode == "conditional":
    midi_dir = "midi_songs"          # 子目录为情绪类别（如 Q1/, Q2/...）
    model_name = "conditional_music_model"
    data_dir = "data"

elif mode == "inpainting":
    midi_dir = "fill_songs"     # 单一文件夹，存放用于旋律补全训练的MIDI
    model_name = "inpainting_music_model"
    data_dir = "fill_data"
else:
    raise ValueError("mode 必须是 'conditional' 或 'inpainting'")

models_dir = "models"
output_dir = "output"

model_filename = f"{model_name}.pth"

# --- 数据处理参数 ---
sequence_length = 100

# --- 模型架构参数 ---
vocab_embed_dim = 128
num_heads = 8
num_layers = 3
dropout_rate = 0.3
feedforward_dim = 512

# --- 训练参数 ---
batch_size = 64
learning_rate = 0.001
weight_decay = 1e-5
clip_grad_norm = 1.0
epochs = 10

# --- 生成参数（仅 conditional 模式使用）---
default_num_notes = 100
generation_temperature = {
    "Q1": 1.2,
    "Q2": 1.3,
    "Q3": 0.8,
    "Q4": 0.7,
    "GiantMIDI-Piano": 1.5,  # 更高温度 → 更大随机性
}
tempo_range = {
    "Q1": (130, 160),
    "Q2": (140, 180),
    "Q3": (50, 80),
    "Q4": (70, 100),
    "GiantMIDI-Piano": (40, 200),  # 覆盖极慢到极快，增强情绪跨度
}
pitch_range = {
    "Q1": (65, 88),
    "Q2": (60, 90),
    "Q3": (40, 65),
    "Q4": (55, 75),
    "GiantMIDI-Piano": (21, 108),  # MIDI 全键盘范围（A0=21 到 C8=108）
}

# --------------------------
# 音乐生成器核心类
# --------------------------
class MusicGenerator:
    def __init__(self):
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化工具类
        if self.mode == "conditional":
            self.utils = MusicGeneratorUtils(
                midi_dir=midi_dir,
                data_dir=data_dir,
                models_dir=models_dir,
                output_dir=output_dir,
                sequence_length=sequence_length,
                generation_temperature=generation_temperature,
                tempo_range=tempo_range,
                pitch_range=pitch_range
            )
        else:  # inpainting
            self.utils = MusicGeneratorUtils(
                midi_dir=midi_dir,
                data_dir=data_dir,
                models_dir=models_dir,
                output_dir=output_dir,
                sequence_length=sequence_length,
                generation_temperature=generation_temperature,
                tempo_range=tempo_range,
                pitch_range=pitch_range
            )

        print(f"🎵 使用设备: {self.device}")
        if self.mode == "conditional":
            if self.utils.emotion_to_id:
                print(f"🎭 检测到情绪类别: {list(self.utils.emotion_to_id.keys())}")
            else:
                print("⚠️  未在 midi_songs/ 下找到任何情绪子目录！")
        else:
            print("🎹 进入旋律补全模式（inpainting）")

    def create_model(self, vocab_size, num_emotions=1):
        """创建Transformer神经网络"""
        print("🧠 正在创建Transformer神经网络...")
        model = TransformerMusicModel(
            vocab_size=vocab_size,
            num_emotions=num_emotions,
            d_model=vocab_embed_dim,
            nhead=num_heads,
            num_layers=num_layers,
            dropout=dropout_rate,
            feedforward_dim=feedforward_dim
        ).to(self.device)
        return model

    def train_model(self, src_sequences, tgt_sequences, emotion_ids, note_to_int, pitchnames):
        vocab_size = len(pitchnames)
        num_emotions = len(set(emotion_ids)) if self.mode == "conditional" else 1

        print(f"📊 训练参数: vocab_size={vocab_size}, num_emotions={num_emotions}")

        src_sequences = torch.LongTensor(src_sequences)
        tgt_sequences = torch.LongTensor(tgt_sequences)
        emotion_ids = torch.LongTensor(emotion_ids)

        dataset = MusicDataset(src_sequences, tgt_sequences, emotion_ids)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = self.create_model(vocab_size, num_emotions)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        best_loss = float('inf')
        best_model_path = os.path.join(models_dir, model_filename)

        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for src, tgt, emo in dataloader:
                src, tgt, emo = src.to(self.device), tgt.to(self.device), emo.to(self.device)
                output = model(src, tgt, emo)
                loss = criterion(output.reshape(-1, vocab_size), tgt[:, 1:].reshape(-1))

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"📊 Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "vocab_size": vocab_size,
                    "num_emotions": num_emotions,
                    "note_to_int": note_to_int,
                    "pitchnames": pitchnames,
                    "emotion_to_id": getattr(self.utils, 'emotion_to_id', None)
                }, best_model_path)
                print(f"💾 新的最佳模型已保存 (loss={avg_loss:.4f}): {best_model_path}")

        return model

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(models_dir, model_filename)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"未找到训练好的模型文件: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)
        model = TransformerMusicModel(
            vocab_size=checkpoint["vocab_size"],
            num_emotions=checkpoint["num_emotions"],
            d_model=vocab_embed_dim,
            nhead=num_heads,
            num_layers=num_layers,
            dropout=dropout_rate,
            feedforward_dim=feedforward_dim
        ).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        self.note_to_int = checkpoint["note_to_int"]
        self.int_to_note = {v: k for k, v in self.note_to_int.items()}
        self.emotion_to_id = checkpoint.get("emotion_to_id", None)
        print(f"✅ 模型加载成功: {model_path}")
        return model

    def generate_conditional(self, emotion="Q1", num_notes=default_num_notes, output_file="generated.mid"):
        """情绪控制生成"""
        try:
            model = self.load_model()
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return None

        if emotion not in self.emotion_to_id:
            print(f"❌ 不支持的情绪: {emotion}，可用情绪: {list(self.emotion_to_id.keys())}")
            return None

        emotion_id = self.emotion_to_id[emotion]

        notes_data_path = os.path.join(data_dir, "notes_with_emotion.pkl")
        if not os.path.exists(notes_data_path):
            print("❌ 找不到训练数据，请先运行训练")
            return None

        with open(notes_data_path, "rb") as f:
            notes, _, _ = pickle.load(f)

        if len(notes) < sequence_length:
            print(f"❌ 音符数量不足，需要至少 {sequence_length} 个音符")
            return None

        start_idx = np.random.randint(0, len(notes) - sequence_length)
        start_notes = notes[start_idx:start_idx + sequence_length]

        try:
            start_int = [self.note_to_int[note] for note in start_notes]
        except KeyError as e:
            print(f"❌ 起始序列包含未知音符: {e}")
            return None

        start_tensor = torch.tensor([start_int], dtype=torch.long).to(self.device)
        temperature = generation_temperature[emotion]
        generated_int = model.generate(start_tensor, emotion_id, max_len=num_notes, temperature=temperature)
        generated_notes = [self.int_to_note.get(i, "60") for i in generated_int]

        return self.utils.create_midi_from_notes(generated_notes, output_file, emotion)

    def complete_melody(self, input_midi_path, output_file="completed.mid", num_completion_notes=100):
        """旋律补全：读取输入MIDI，提取前N个音符作为上下文，补全后续"""
        try:
            model = self.load_model()
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return None

        # 读取输入 MIDI（参考utils中的方法）
        try:
            from music21 import converter, instrument, note, chord
            midi = converter.parse(input_midi_path)
            parts = instrument.partitionByInstrument(midi)
            notes_to_parse = parts.parts[0].recurse() if parts else midi.flat.notes
            
            notes = []
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
            
            if not notes:
                print("❌ 输入MIDI中没有找到有效音符")
                return None
            print(f"📥 从输入MIDI中提取了 {len(notes)} 个音符")
        except Exception as e:
            print(f"❌ 无法解析输入MIDI文件: {e}")
            return None

        # 处理音符序列长度
        if len(notes) < sequence_length:
            print(f"⚠️ 输入音符不足 {sequence_length}，使用智能填充")
            # 使用训练数据中的常见音符进行填充，而不是简单重复
            if hasattr(self, 'note_to_int'):
                common_notes = list(self.note_to_int.keys())
                if common_notes:
                    # 从常见音符中随机选择填充
                    fill_notes = np.random.choice(common_notes, sequence_length - len(notes))
                    notes.extend(fill_notes)
                else:
                    # 如果没有词汇表信息，使用C大调音阶填充
                    c_major_scale = ['60', '62', '64', '65', '67', '69', '71', '72']  # C4 to C5
                    while len(notes) < sequence_length:
                        notes.append(np.random.choice(c_major_scale))
            else:
                print("❌ 无法获取词汇表信息进行智能填充")
                return None
        else:
            notes = notes[:sequence_length]

        # 过滤未知音符，使用最接近的训练集中存在的音符
        start_notes = []
        for note in notes:
            if note in self.note_to_int:
                start_notes.append(note)
            else:
                # 找到最接近的训练集中存在的音符
                try:
                    # 处理和弦
                    if '.' in note:
                        # 对于和弦，尝试找到最接近的和弦表示
                        closest_chord = min(self.note_to_int.keys(), 
                                          key=lambda x: len(set(note.split('.')) & set(x.split('.'))))
                        start_notes.append(closest_chord)
                        print(f"🔧 将未知和弦 {note} 替换为 {closest_chord}")
                    else:
                        # 对于单音符
                        note_pitch = int(note)
                        closest_note = min(self.note_to_int.keys(), 
                                         key=lambda x: abs(int(x) - note_pitch) if '.' not in x else float('inf'))
                        start_notes.append(closest_note)
                        print(f"🔧 将未知音符 {note} 替换为 {closest_note}")
                except (ValueError, TypeError):
                    # 如果无法转换为数字，使用默认音符
                    start_notes.append("60")
                    print(f"🔧 将无效音符 {note} 替换为 60")
        
        # 参考generate_conditional方法进行旋律补全
        try:
            # 将音符转换为整数表示
            start_int = [self.note_to_int[note] for note in start_notes]
            
            # 创建输入张量
            start_tensor = torch.tensor([start_int], dtype=torch.long).to(self.device)
            
            # 确定温度参数（对于补全任务，使用适当的温度）
            temperature = generation_temperature.get("GiantMIDI-Piano", 1.0)
            
            # 使用模型生成后续音符
            generated_int = model.generate(start_tensor, 0, max_len=num_completion_notes, temperature=temperature)
            
            # 将生成的整数转换回音符
            generated_notes = [self.int_to_note.get(i, "60") for i in generated_int]
            
            # 合并处理后的种子音符和生成的音符
            complete_notes = start_notes + generated_notes
            
            # 生成MIDI文件（使用GiantMIDI-Piano作为默认情绪配置）
            return self.utils.create_midi_from_notes(complete_notes, output_file, "GiantMIDI-Piano")
        except KeyError as e:
            print(f"❌ 音符转换错误: {e}")
            return None
        except Exception as e:
            print(f"❌ 旋律生成过程中出错: {e}")
            return None

    def train_from_scratch(self):
        print("🔄 开始从头训练模型...")

        if self.mode == "conditional":
            notes, emotion_labels, emotion_to_id = self.utils.get_notes_with_emotion()
            if not notes:
                print("❌ 没有找到可用的音符数据")
                return None
            src_seq, tgt_seq, emo_ids, note_to_int, pitchnames = self.utils.prepare_sequences(notes, emotion_labels)
        else:  # inpainting
            notes, _, _ = self.utils.get_notes_with_emotion()  # 忽略情绪
            if not notes:
                print("❌ midi_inpainting/ 目录为空")
                return None
            # 构造 dummy emotion_ids（全0）
            emotion_labels = [0] * len(notes)
            src_seq, tgt_seq, emo_ids, note_to_int, pitchnames = self.utils.prepare_sequences(notes, emotion_labels)

        if src_seq is None:
            print("❌ 序列准备失败")
            return None

        model = self.train_model(src_seq, tgt_seq, emo_ids, note_to_int, pitchnames)

        # 保存映射
        os.makedirs(data_dir, exist_ok=True)
        with open(os.path.join(data_dir, "note_to_int.pkl"), "wb") as f:
            pickle.dump(note_to_int, f)
        if self.mode == "conditional":
            with open(os.path.join(data_dir, "emotion_to_id.pkl"), "wb") as f:
                pickle.dump(emotion_to_id, f)

        print("✅ 训练完成")
        return model


def main():
    generator = MusicGenerator()
    model_path = os.path.join(models_dir, model_filename)

    if os.path.exists(model_path):
        print("🎵 发现已训练的模型")
        if generator.mode == "conditional":
            emotions = list(generator.utils.emotion_to_id.keys())
            for emotion in emotions:
                output_file = os.path.join(output_dir, f"demo_output_{emotion}.mid")
                result = generator.generate_conditional(
                    emotion=emotion,
                    num_notes=default_num_notes,
                    output_file=output_file
                )
                if result:
                    print(f"✅ 成功生成 {emotion} 音乐: {result}")
                else:
                    print(f"❌ 生成 {emotion} 音乐失败")
        else:  # inpainting
            # 示例：补全一个输入MIDI（需用户指定）
            input_midi = "input_seed.mid"  # <<< 用户可修改此路径
            if os.path.exists(input_midi):
                print(f"🎹 开始旋律补全，输入文件: {input_midi}")
                result = generator.complete_melody(
                    input_midi_path=input_midi,
                    output_file="completed_output.mid",
                    num_completion_notes=100
                )
                if result:
                    print(f"✅ 旋律补全成功: {result}")
                else:
                    print("❌ 旋律补全失败")
            else:
                print(f"⚠️ 未找到输入MIDI文件: {input_midi}")
                print("💡 请创建一个包含一些音符的 input_seed.mid 文件，或修改 main() 中的 input_midi 路径")
    else:
        print("🔄 开始训练新模型...")
        model = generator.train_from_scratch()
        if model is not None and generator.mode == "conditional":
            emotions = list(generator.utils.emotion_to_id.keys())
            if emotions:
                first_emotion = emotions[0]
                output_file = os.path.join(output_dir, f"demo_output_{first_emotion}.mid")
                result = generator.generate_conditional(
                    emotion=first_emotion,
                    num_notes=default_num_notes,
                    output_file=output_file
                )
                if result:
                    print(f"✅ 成功生成演示音乐: {result}")
                else:
                    print("❌ 生成演示音乐失败")


if __name__ == "__main__":
    main()