# xlsr-english-l2
## Thành viên
Nguyễn Văn Lâm Hùng  
Lê Sỹ Long Nhật  
Trần Văn Nam

## Hướng dẫn chạy model

```python
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Đường dẫn đến file wav của bạn
file_path = "19-198-0002.wav"

audio, sr = librosa.load(file_path, sr=16000)

print(f"Dạng dữ liệu: {type(audio)}")
print(f"Sample rate: {sr}")
print(f"Độ dài mảng: {audio.shape}")

# Tải model and processor đã được train
processor_l2 = Wav2Vec2Processor.from_pretrained("Lam-Hung/xlsr-english-l2", sampling_rate=16000)
model_l2 = Wav2Vec2ForCTC.from_pretrained("Lam-Hung/xlsr-english-l2")

# tokenize
input_values_l2 = processor_l2(audio, return_tensors="pt").input_values # type: ignore

with torch.no_grad():
    logits = model_l2(input_values_l2).logits

# Lây argmax and decode
predicted_ids_l2 = torch.argmax(logits, dim=-1)
transcription_l2 = processor_l2.batch_decode(predicted_ids_l2)
for c in transcription_l2[0]:
    print(c, end=' ')

# n i ð ɝ ð i ɔ θ ɝ n ɔ ɹ ð ʌ p ʌ b l ɪ k h æ v ɛ n i ʌ ð ɝ k ʌ n s ɝ n d æ n æ z s ʌ m ɑ b z ɝ v e ɪ ʃ ʌ n ɪ z n ɛ s ʌ s ɛ ɹ i ʌ p ɑ n ð o ʊ z p ɑ ɹ t s ʌ v ð ʌ w ɝ k h w ɪ t ʃ θ ɝ t i n j ɪ ɹ z h æ v m e ɪ d k ʌ m p ɛ ɹ ʌ t ɪ v l i ɑ b s ʌ l i t 

```

## Hướng đẫn train với google Colab
Môi trường: google colab  
Phần cứng: H100  
Thời gian huấn luyện 2 tiếng
Thay đổi file ML-main/scripts/ipa_transcription/wav2vec2_train.py thành file [wav2vec2_train.py](./wav2vec2_train.py) của repo này  
Code: copy từng ô trong file [gg_colab_fintune.ipynb](./gg_colab_finetune.ipynb)

## Hướng dẫn đánh giá sau khi train
Tải github repo https://github.com/KoelLabs/ML.git
copy file [eval.ipynb](./eval.ipynb) vào thưc mục ML-main/scripts/eval, chú ý thay đổi tên model cho đúng  
Sau đó chạy tất cả (lưu ý cần phải tải đầy đủ thư viện)
