import os
from gtts import gTTS

# 定義你需要生成的物品清單
items = ["杯子", "水壺", "手機", "筆電", "碗", "書", "鍵盤", "滑鼠"]
output_dir = "music"

# 確保資料夾存在
os.makedirs(output_dir, exist_ok=True)

for item in items:
    file_path = os.path.join(output_dir, f"{item}.wav")
    if not os.path.exists(file_path):
        print(f"正在生成: {item}.wav ...")
        # 使用 gTTS 生成中文語音
        tts = gTTS(text=item, lang='zh-TW')
        tts.save(file_path)
        print(f"✅ 已儲存至 {file_path}")
    else:
        print(f"⏩ {item}.wav 已存在，跳過。")

print("所有語音檔生成完畢！")