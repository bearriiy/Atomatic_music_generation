import os
import hashlib
import music21
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# -----------------------------
# 评估单个 MIDI 文件
# -----------------------------
def evaluate_midi(midi_path):
    try:
        midi_data = music21.converter.parse(midi_path)
    except Exception as e:
        print(f"❌ 无法解析: {midi_path} - {e}")
        return None

    notes = midi_data.flat.notes
    if len(notes) == 0:
        print(f"⚠️ MIDI 文件无音符: {midi_path}")
        return None

    pitches = [n.pitch.midi for n in notes if n.isNote]
    durations = [float(n.duration.quarterLength) for n in notes if n.isNote]
    velocities = [n.volume.velocity for n in notes if n.isNote and n.volume.velocity is not None]

    # 1. Pitch Entropy
    pitch_count = Counter(pitches)
    total = len(pitches)
    probs = np.array(list(pitch_count.values())) / total
    pitch_entropy = -np.sum(probs * np.log2(probs + 1e-9))

    # 2. Pitch Smoothness
    if len(pitches) > 1:
        diffs = np.abs(np.diff(pitches))
        pitch_smoothness = np.mean(diffs)
    else:
        pitch_smoothness = 0

    # 3. Repetition Rate
    window = 4
    seen = set()
    repeat_cnt = 0
    for i in range(len(pitches) - window):
        segment = tuple(pitches[i:i + window])
        code = hashlib.md5(str(segment).encode()).hexdigest()
        if code in seen:
            repeat_cnt += 1
        else:
            seen.add(code)
    repetition_rate = repeat_cnt / max(1, len(pitches))

    # 4. Rhythm Entropy
    duration_count = Counter(durations)
    duration_total = len(durations)
    duration_probs = np.array(list(duration_count.values())) / duration_total
    rhythm_entropy = -np.sum(duration_probs * np.log2(duration_probs + 1e-9))

    # 5. Velocity Range
    if velocities:
        velocity_range = max(velocities) - min(velocities)
    else:
        velocity_range = 0

    return {
        "File": os.path.basename(midi_path),
        "Pitch Entropy": pitch_entropy,
        "Pitch Smoothness": pitch_smoothness,
        "Repetition Rate": repetition_rate,
        "Rhythm Entropy": rhythm_entropy,
        "Velocity Range": velocity_range
    }

# -----------------------------
# 批量评估 MIDI 文件并生成 Excel + 雷达图
# -----------------------------
def evaluate_folder(midi_folder, radar_folder="evaluation_radar"):
    os.makedirs(radar_folder, exist_ok=True)
    results = []

    files = [f for f in os.listdir(midi_folder) if f.endswith(".mid") or f.endswith(".midi")]
    if not files:
        print("❌ 未找到 MIDI 文件")
        return

    for f in files:
        path = os.path.join(midi_folder, f)
        print(f"🎵 正在评估: {f}")
        result = evaluate_midi(path)
        if result:
            results.append(result)

    if not results:
        print("❌ 没有有效 MIDI 文件")
        return

    df = pd.DataFrame(results)

    # -----------------------------
    # Min-Max 归一化
    # -----------------------------
    norm_cols = ["Pitch Entropy", "Pitch Smoothness", "Repetition Rate", "Rhythm Entropy", "Velocity Range"]
    for col in norm_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val - min_val > 1e-9:
            df[f"Norm {col}"] = (df[col] - min_val) / (max_val - min_val)
        else:
            df[f"Norm {col}"] = 0.5

    # 综合评分
    df["Overall Score"] = (
        df["Norm Pitch Entropy"] * 0.3 +
        (1 - df["Norm Pitch Smoothness"]) * 0.2 +
        (1 - df["Norm Repetition Rate"]) * 0.2 +
        df["Norm Rhythm Entropy"] * 0.2 +
        df["Norm Velocity Range"] * 0.1
    ) * 10

    # -----------------------------
    # 只保留五个核心指标 + 归一化 + Overall Score
    # -----------------------------
    keep_cols = ["File"] + norm_cols + [f"Norm {c}" for c in norm_cols] + ["Overall Score"]
    df = df[keep_cols]

    excel_path = "evaluate.xlsx"
    df.to_excel(excel_path, index=False)
    print(f"✅ Excel 已保存: {excel_path}")

    # -----------------------------
    # 生成雷达图
    # -----------------------------
    categories = norm_cols

    for _, row in df.iterrows():
        values = [row[f"Norm {c}"] for c in categories]
        values += values[:1]  # 闭合

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
        ax.plot(angles, values, linewidth=2, linestyle='solid')
        ax.fill(angles, values, 'b', alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_yticks([0,0.25,0.5,0.75,1])
        ax.set_yticklabels(["0","0.25","0.5","0.75","1"])
        ax.set_title(f"{row['File']}")

        out_file = os.path.join(radar_folder, f"radar_{row['File'].replace('.mid','')}.png")
        plt.tight_layout()
        plt.savefig(out_file)
        plt.close()
        print(f"✅ 雷达图已保存: {out_file}")

# -----------------------------
# 主程序
# -----------------------------
if __name__ == "__main__":
    midi_folder = "output"  # 修改为你的 MIDI 文件夹
    evaluate_folder(midi_folder)
