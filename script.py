import openai
import whisper
from pydub import AudioSegment
from pyannote.audio import Pipeline
from tqdm import tqdm
import time
import torch
import os

# Установите ваши API ключи
# З.Ы. openai требует отдельной оплаты за api-ключи, так что у меня не завелось)

openai_api_key = ''  # Замените 'YOUR_OPENAI_API_KEY' на ваш реальный API-ключ
pyannote_api_key = ''  # Замените 'YOUR_PYANNOTE_API_KEY' на ваш реальный API-ключ из pyannote  

# Параметризация отправки текстов на обработку
use_openai_processing = False  # Установите True, если хотите отправлять текст на обработку через API OpenAI

# Установка API-ключа OpenAI
openai.api_key = openai_api_key

# Проверка наличия GPU и использование его (для GPU требуется установка nvidia cuda)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Проверка доступности CUDA
if torch.cuda.is_available():
    print("CUDA is available:", torch.cuda.is_available())
    print("Number of GPUs available:", torch.cuda.device_count())
    print("CUDA device name:", torch.cuda.get_device_name(0))

# Начало замера времени выполнения скрипта
start_time = time.time()

# Установите путь к вашему аудиофайлу или директории с аудиофайлами
audio_file_path = r"C:\Users\Nihil\Downloads\Recording_58.wav"  # Может быть файлом или директорией
output_audio_path = r"C:\Users\Nihil\Downloads\Kashirkin\files1"  # Директория для выходных аудио
output_text_path = r"C:\Users\Nihil\Downloads\Kashirkin\texts"  # Директория для текстовых файлов
processed_text_path = r"C:\Users\Nihil\Downloads\Kashirkin\processed_texts"  # Директория для обработанных текстовых файлов

# Выбор модели Whisper
model_name = "large-v3"  # Здесь можно выбрать модель: tiny, base, small, medium, large
model_location = r"D:\ffmpeg\model"  # Укажите путь к директории с моделями

# Проверка существования директории для моделей
if not os.path.exists(model_location):
    os.makedirs(model_location)

# Загрузка модели Whisper
model = whisper.load_model(model_name, download_root=model_location).to(device)
print(f"Whisper model '{model_name}' is using device: {device}")

# Функция для обработки одного файла
def process_audio_file(audio_file, output_audio_dir, output_text_dir, processed_text_dir, use_processing):
    # Конвертация аудиофайла в нужный формат с помощью pydub
    audio = AudioSegment.from_wav(audio_file)
    audio = audio.set_channels(1)  # Установить моно
    audio = audio.set_frame_rate(16000)  # Установить частоту дискретизации 16000 Гц

    output_audio_file = os.path.join(output_audio_dir, os.path.basename(audio_file))
    audio.export(output_audio_file, format="wav")

    # Распознавание речи с использованием модели Whisper
    result = model.transcribe(output_audio_file, language='ru')
    transcription = result["text"]
    print(f"Initial transcription of {audio_file} complete.")

    # Диаризация аудио с помощью pyannote.audio
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=pyannote_api_key)
    diarization = pipeline(output_audio_file)

    # Получение информации о спикерах и временных метках
    segments = [(turn.start, turn.end, speaker) for turn, _, speaker in diarization.itertracks(yield_label=True)]

    # Конвертация временных меток в текст с учетом спикеров и буферов
    recognized_text_with_speakers = []
    buffer_duration = 0.3  # Уменьшение буфера до 0.1 секунды

    for start, end, speaker in tqdm(segments, desc="Processing segments"):
        start = max(0, start - buffer_duration)  # Начало сегмента с буфером
        end = min(len(audio) / 1000, end + buffer_duration)  # Конец сегмента с буфером
        segment_audio = audio[start * 1000:end * 1000]
        segment_audio_path = "segment.wav"
        segment_audio.export(segment_audio_path, format="wav")
        segment_result = model.transcribe(segment_audio_path, language='ru')
        recognized_text_with_speakers.append((speaker, segment_result['text']))

    # Сохранение результата в текстовый файл
    output_text_file = os.path.join(output_text_dir, os.path.splitext(os.path.basename(audio_file))[0] + ".txt")
    with open(output_text_file, "w", encoding="utf-8") as f:
        f.write("Full transcription:\n")
        f.write(transcription + "\n\n")
        f.write("Speaker segments:\n")
        for speaker, text in recognized_text_with_speakers:
            line = f"Speaker {speaker}: {text}\n"
            f.write(line)
            print(line)

    # Отправка текста на обработку через OpenAI API, если параметр включен
    if use_processing:
        with open(output_text_file, "r", encoding="utf-8") as f:
            original_text = f.read()

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a text editor that improves readability and adds punctuation."},
                {"role": "user", "content": f"Please edit the following text, adding punctuation and improving readability:\n\n{original_text}"}
            ],
            max_tokens=4096,
            temperature=0.5
        )

        processed_text = response['choices'][0]['message']['content']

        # Сохранение обработанного текста в новый файл
        processed_text_file = os.path.join(processed_text_dir, os.path.splitext(os.path.basename(audio_file))[0] + "_processed.txt")
        with open(processed_text_file, "w", encoding="utf-8") as f:
            f.write(processed_text)

# Проверка и создание необходимых директорий
if not os.path.exists(output_audio_path):
    os.makedirs(output_audio_path)

if not os.path.exists(output_text_path):
    os.makedirs(output_text_path)

if not os.path.exists(processed_text_path):
    os.makedirs(processed_text_path)

# Обработка всех файлов в директории или одного файла
if os.path.isdir(audio_file_path):
    audio_files = [f for f in os.listdir(audio_file_path) if f.endswith('.wav')]
    for audio_file in audio_files:
        audio_file_full_path = os.path.join(audio_file_path, audio_file)
        process_audio_file(audio_file_full_path, output_audio_path, output_text_path, processed_text_path, use_openai_processing)
else:
    process_audio_file(audio_file_path, output_audio_path, output_text_path, processed_text_path, use_openai_processing)

# Конец замера времени выполнения скрипта
end_time = time.time()
execution_time = end_time - start_time
print(f"Время выполнения скрипта: {execution_time:.2f} секунд")
