# - 包含 MusicDataset 类，用于数据加载
# - 实现 MusicGeneratorUtils 类，提供各种工具方法：
#   - get_notes_with_emotion() - 从MIDI文件中提取带情绪标签的音符
#   - prepare_sequences() - 准备训练序列数据
#   - create_midi_from_notes() - 将生成的音符序列转换为MIDI文件


import os
import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord, stream, tempo
import warnings

warnings.filterwarnings('ignore')

# --------------------------
# 数据集类（适配Transformer输入格式）
# --------------------------
class MusicDataset:
    def __init__(self, src_sequences, tgt_sequences, emotion_ids):
        self.src_sequences = src_sequences
        self.tgt_sequences = tgt_sequences
        self.emotion_ids = emotion_ids

    def __len__(self):
        return len(self.src_sequences)

    def __getitem__(self, idx):
        return self.src_sequences[idx], self.tgt_sequences[idx], self.emotion_ids[idx]

# --------------------------
# 音乐生成器工具类
# --------------------------
class MusicGeneratorUtils:
    def __init__(self, midi_dir, data_dir, models_dir, output_dir, sequence_length, 
                 generation_temperature, tempo_range, pitch_range):
        self.midi_dir = midi_dir
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.output_dir = output_dir
        self.sequence_length = sequence_length
        self.generation_temperature = generation_temperature
        self.tempo_range = tempo_range
        self.pitch_range = pitch_range
        
        # 确保目录存在
        os.makedirs(self.midi_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 动态获取情绪类别
        emotion_dirs = [d for d in os.listdir(self.midi_dir)
                        if os.path.isdir(os.path.join(self.midi_dir, d))]
        emotion_dirs.sort()
        self.emotion_to_id = {emo: idx for idx, emo in enumerate(emotion_dirs)}
        self.id_to_emotion = {idx: emo for emo, idx in self.emotion_to_id.items()}
        
        # 构建动态情绪配置
        self.emotion_config = {}
        for emo in self.emotion_to_id:
            temp = generation_temperature.get(emo, 1.0)
            tempo_rng = tempo_range.get(emo, (80, 120))
            pitch_rng = pitch_range.get(emo, (48, 84))
            self.emotion_config[emo] = {
                "temperature": temp,
                "tempo_range": tempo_rng,
                "pitch_range": pitch_rng
            }
    
    def get_notes_with_emotion(self):
        """从 MIDI 文件中提取音符并打上情绪标签"""
        pkl_path = f"{self.data_dir}/notes_with_emotion.pkl"

        if os.path.exists(pkl_path):
            print("📂 发现已处理的数据文件，正在加载...")
            with open(pkl_path, "rb") as f:
                notes, emotion_labels, emotion_to_id = pickle.load(f)
            emotion_counts = np.bincount(emotion_labels) if emotion_labels else []
            print(f"✅ 已加载 {len(notes)} 个音符，情绪分布: {emotion_counts}")
            return notes, emotion_labels, emotion_to_id

        print("🎵 正在从MIDI文件中提取带情绪标签的音符...")
        notes = []
        emotion_labels = []
        emotion_to_id = self.emotion_to_id

        for emotion_name, emotion_id in emotion_to_id.items():
            folder = os.path.join(self.midi_dir, emotion_name)
            if not os.path.exists(folder):
                continue

            midi_files = glob.glob(os.path.join(folder, "*.mid")) + \
                         glob.glob(os.path.join(folder, "*.midi"))
            print(f"📁 情绪 '{emotion_name}' 找到 {len(midi_files)} 个MIDI文件")

            for file in midi_files:
                try:
                    midi = converter.parse(file)
                    parts = instrument.partitionByInstrument(midi)
                    notes_to_parse = parts.parts[0].recurse() if parts else midi.flat.notes

                    song_notes = []
                    for element in notes_to_parse:
                        if isinstance(element, note.Note):
                            song_notes.append(str(element.pitch))
                        elif isinstance(element, chord.Chord):
                            song_notes.append('.'.join(str(n) for n in element.normalOrder))

                    if song_notes:
                        notes.extend(song_notes)
                        emotion_labels.extend([emotion_id] * len(song_notes))
                except Exception as e:
                    print(f"⚠️  处理 {file} 时出错: {e}")
                    continue

        with open(pkl_path, "wb") as f:
            pickle.dump((notes, emotion_labels, emotion_to_id), f)

        emotion_counts = np.bincount(emotion_labels) if emotion_labels else []
        print(f"✅ 共提取 {len(notes)} 个音符，情绪分布: {emotion_counts}")
        return notes, emotion_labels, emotion_to_id

    def prepare_sequences(self, notes, emotion_labels, note_to_int=None):
        """准备带情绪标签的训练序列"""
        print("⚙️  正在准备带情绪标签的训练序列...")

        if len(notes) != len(emotion_labels):
            raise ValueError("音符与情绪标签长度不一致")

        pitchnames = sorted(set(notes))
        if note_to_int is None:
            note_to_int = {note: i for i, note in enumerate(pitchnames)}
        vocab_size = len(pitchnames)

        src_sequences = []
        tgt_sequences = []
        emotion_ids = []

        if len(notes) < 2 * self.sequence_length:
            print(f"⚠️  音符数量不足")
            return None, None, None, None, None

        for i in range(len(notes) - 2 * self.sequence_length + 1):
            src_seq = notes[i:i + self.sequence_length]
            tgt_seq = notes[i + 1:i + 1 + self.sequence_length]
            emotion_id = emotion_labels[i + self.sequence_length // 2]

            try:
                src_int = [note_to_int[n] for n in src_seq]
                tgt_int = [note_to_int[n] for n in tgt_seq]
                src_sequences.append(src_int)
                tgt_sequences.append(tgt_int)
                emotion_ids.append(emotion_id)
            except KeyError:
                continue

        print(f"✅ 准备完成: {len(src_sequences)} 个样本")
        return src_sequences, tgt_sequences, emotion_ids, note_to_int, pitchnames

    def create_midi_from_notes(self, generated_notes, output_file, emotion):
        """将生成的音符序列转换为MIDI文件"""
        # 获取情绪配置
        tempo_range = self.emotion_config[emotion]["tempo_range"]
        current_tempo = np.random.randint(tempo_range[0], tempo_range[1] + 1)

        offset = 0
        output_notes = [tempo.MetronomeMark(number=current_tempo)]

        for pattern in generated_notes:
            if "." in pattern:  # 和弦
                notes_in_chord = pattern.split(".")
                chord_notes = []
                for n in notes_in_chord:
                    try:
                        chord_notes.append(note.Note(int(n)))
                    except:
                        continue
                if chord_notes:
                    ch = chord.Chord(chord_notes)
                    ch.offset = offset
                    output_notes.append(ch)
            else:  # 单音符
                try:
                    n = note.Note(pattern)
                    n.offset = offset
                    output_notes.append(n)
                except:
                    continue
            offset += 0.5

        midi_stream = stream.Stream(output_notes)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        midi_stream.write('midi', fp=output_file)

        print(f"✅ 生成完成: {output_file} (情绪: {emotion})")
        return output_file