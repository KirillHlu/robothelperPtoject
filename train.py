import torch
import torch.nn as nn
from torch.nn import functional as F
import json
import os
import math
import requests
import re
import time
import sys
import serial
import serial.tools.list_ports
from urllib.parse import quote

# ========== НАСТРОЙКИ ==========
config = {
    "block_size": 512,
    "n_embed": 512,
    "n_head": 8,
    "n_layer": 8,
    "dropout": 0.1,
    "temperature": 0.3,
    "top_k": 40,
    "max_new_tokens": 100,
}

# Голосовые настройки
VOICE_CONFIG = {
    "wake_word": "иван",
    "wake_word_variants": ["иван", "ивана", "ивану", "ван", "иваныч"],
    "language": "ru",
    "energy_threshold": 300,
    "pause_threshold": 0.8,
    "timeout": 5,
    "phrase_time_limit": 10,
}

# Проверка импорта голосовых библиотек
try:
    import speech_recognition as sr

    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    print("Предупреждение: SpeechRecognition не установлен")

try:
    import pyttsx3

    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("Предупреждение: pyttsx3 не установлен")

try:
    import pyaudio

    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("Предупреждение: PyAudio не установлен")


# ========== КЛАСС ДЛЯ СВЯЗИ С ESP32 ==========
class ESP32Connector:
    def __init__(self):
        self.serial = None
        self.connected = False
        self.port = None

    def find_esp32(self):
        """Поиск ESP32 по COM-порту"""
        ports = serial.tools.list_ports.comports()

        for port in ports:
            print(f"Найден порт: {port.device} - {port.description}")
            if 'COM' in port.device.upper() or 'ttyUSB' in port.device or 'ttyACM' in port.device:
                return port.device
        return None

    def connect(self):
        """Подключение к ESP32"""
        self.port = self.find_esp32()

        if not self.port:
            print("ESP32 не найден! Проверьте подключение.")
            return False

        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=115200,
                timeout=2,
                write_timeout=2
            )
            time.sleep(2)  # Ждём инициализации ESP32
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()

            self.connected = True
            print(f"Подключен к ESP32 на порту {self.port}")

            # Тест связи
            self.send_command("TEST")
            time.sleep(1)

            return True

        except Exception as e:
            print(f"Ошибка подключения к ESP32: {e}")
            return False

    def send_command(self, command):
        """Отправка команды на ESP32"""
        if not self.serial or not self.connected:
            return False

        try:
            cmd = command + "\n"
            self.serial.write(cmd.encode('utf-8'))
            return True
        except Exception as e:
            print(f"Ошибка отправки: {e}")
            return False

    def send_text(self, text):
        """Отправка текста на ESP32 для озвучивания"""
        if not self.serial or not self.connected:
            return False

        # Очищаем текст
        clean_text = re.sub(r'[*_#`>\n]', '', text)
        clean_text = re.sub(r'\[[0-9]+\]', '', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        if clean_text:
            # Ограничиваем длину
            if len(clean_text) > 200:
                clean_text = clean_text[:200] + "..."

            print(f"Отправка на ESP32: {clean_text}")

            # Отправляем текст
            self.serial.write((clean_text + "\n").encode('utf-8'))
            return True

        return False

    def beep(self):
        """Звуковой сигнал на ESP32"""
        self.send_command("BEEP")

    def test(self):
        """Тест динамика ESP32"""
        self.send_command("TEST")

    def close(self):
        """Закрытие соединения"""
        if self.serial:
            self.serial.close()
            self.connected = False


# ========== КЛАСС ДЛЯ СИНТЕЗА РЕЧИ (локально) ==========
class SpeechSynthesizer:
    def __init__(self):
        self.engine = None
        self.available = TTS_AVAILABLE
        self.use_esp32 = False
        self.esp32 = None

        if self.available:
            self.init_engine()

    def set_esp32(self, esp32):
        """Установка ESP32 для вывода звука"""
        self.esp32 = esp32
        self.use_esp32 = True

    def init_engine(self):
        """Инициализация движка синтеза речи"""
        try:
            self.engine = pyttsx3.init()
            voices = self.engine.getProperty('voices')
            for voice in voices:
                if 'russian' in voice.name.lower() or 'русский' in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    break

            self.engine.setProperty('rate', 170)
            self.engine.setProperty('volume', 1.0)
            print("Голосовой движок инициализирован")
        except Exception as e:
            print(f"Ошибка инициализации TTS: {e}")
            self.available = False

    def speak(self, text):
        """Озвучивание текста (через ESP32 или локально)"""
        if not text:
            return

        # Приоритет: ESP32 -> локальный TTS
        if self.use_esp32 and self.esp32 and self.esp32.connected:
            print(f"[ESP32] Отправка на озвучивание: {text[:100]}...")
            self.esp32.send_text(text)
        elif self.available and self.engine:
            clean_text = re.sub(r'[*_#`>]', '', text)
            clean_text = re.sub(r'\n+', '. ', clean_text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()

            if clean_text:
                print(f"[ЛОКАЛЬНЫЙ TTS]: {clean_text[:100]}...")
                self.engine.say(clean_text)
                self.engine.runAndWait()
        else:
            print(f"[ТЕКСТ]: {text}")

    def speak_sync(self, text):
        self.speak(text)


# ========== КЛАСС ДЛЯ РАСПОЗНАВАНИЯ РЕЧИ ==========
class SpeechRecognizer:
    def __init__(self, wake_word="иван"):
        self.recognizer = None
        self.wake_word = wake_word.lower()
        self.wake_word_variants = [wake_word.lower(), "ивана", "ивану", "ван", "иваныч"]
        self.microphone = None
        self.available = SPEECH_RECOGNITION_AVAILABLE and PYAUDIO_AVAILABLE

        if not self.available:
            print("Речь через микрофон недоступна. Будет использован текстовый ввод.")
            return

        try:
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = VOICE_CONFIG["energy_threshold"]
            self.recognizer.pause_threshold = VOICE_CONFIG["pause_threshold"]
            self.recognizer.dynamic_energy_threshold = True

            self.microphone = sr.Microphone()
            print("Настройка микрофона...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print(f"Микрофон настроен")
        except Exception as e:
            print(f"Ошибка доступа к микрофону: {e}")
            self.available = False

    def listen_for_wake_word(self):
        """Слушает и ждёт активационное слово"""
        if not self.available:
            return self.text_mode_wake()

        try:
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=VOICE_CONFIG["timeout"], phrase_time_limit=3)

            try:
                text = self.recognizer.recognize_google(audio, language="ru-RU").lower()
                print(f"  Распознано: {text}")

                for variant in self.wake_word_variants:
                    if variant in text:
                        print(f"  [АКТИВАЦИЯ] Слово '{variant}' обнаружено!")
                        return True
                return False

            except sr.UnknownValueError:
                return False
            except sr.RequestError as e:
                print(f"Ошибка сервиса распознавания: {e}")
                return False

        except sr.WaitTimeoutError:
            return False
        except Exception as e:
            return False

    def text_mode_wake(self):
        """Режим текстового ввода"""
        print("\n[Текстовый режим] Нажмите Enter для вопроса")
        input(">>> ")
        return True

    def listen_for_command(self):
        """Слушает команду после активации"""
        if not self.available:
            return self.text_mode_command()

        print("[СЛУШАЮ КОМАНДУ] Говорите вопрос...")

        try:
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=VOICE_CONFIG["timeout"],
                                               phrase_time_limit=VOICE_CONFIG["phrase_time_limit"])

            try:
                text = self.recognizer.recognize_google(audio, language="ru-RU")
                print(f"  Распознано: {text}")
                return text.strip()

            except sr.UnknownValueError:
                print("  НЕ РАСПОЗНАНО")
                return None
            except sr.RequestError as e:
                print(f"  Ошибка сервиса: {e}")
                return None

        except sr.WaitTimeoutError:
            print("  ТАЙМАУТ")
            return None
        except Exception as e:
            print(f"  Ошибка: {e}")
            return None

    def text_mode_command(self):
        """Текстовый ввод команды"""
        print("[Введите вопрос]: ", end="")
        command = input().strip()
        if command and command.lower() not in ['выход', 'quit', 'exit']:
            return command
        return None


# ========== КЛАСС ДЛЯ ПОИСКА В ИНТЕРНЕТЕ ==========
class InternetSearch:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def search_wikipedia(self, query):
        """Поиск в Википедии"""
        try:
            clean_query = re.sub(
                r'^(что такое|кто такой|что значит|что означает|определение|объясните|расскажите о)\s+', '',
                query.lower())
            clean_query = clean_query.strip()

            if not clean_query:
                clean_query = query

            url = "https://ru.wikipedia.org/api/rest_v1/page/summary/" + quote(clean_query)
            response = self.session.get(url, timeout=8)

            if response.status_code == 200:
                data = response.json()
                if 'extract' in data:
                    extract = data['extract']
                    extract = re.sub(r'\[[0-9]+\]', '', extract)
                    if len(extract) > 500:
                        extract = extract[:500] + "..."
                    return extract

            search_url = "https://ru.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'list': 'search',
                'srsearch': clean_query,
                'format': 'json',
                'utf8': 1,
                'srlimit': 1
            }

            response = self.session.get(search_url, params=params, timeout=8)
            data = response.json()

            for page in data.get('query', {}).get('search', []):
                content_url = "https://ru.wikipedia.org/api/rest_v1/page/summary/" + quote(page['title'])
                content_resp = self.session.get(content_url, timeout=8)
                if content_resp.status_code == 200:
                    content_data = content_resp.json()
                    if 'extract' in content_data:
                        extract = content_data['extract']
                        extract = re.sub(r'\[[0-9]+\]', '', extract)
                        if len(extract) > 500:
                            extract = extract[:500] + "..."
                        return extract

            return None
        except Exception as e:
            print(f"Ошибка Википедии: {e}")
            return None


# ========== МОДЕЛЬ ==========
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        assert n_embed % n_head == 0
        self.n_head = n_head
        self.n_embed = n_embed
        self.head_dim = n_embed // n_head

        self.c_attn = nn.Linear(n_embed, 3 * n_embed)
        self.c_proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(config['dropout'])

        self.register_buffer("bias", torch.tril(torch.ones(config["block_size"], config["block_size"]))
                             .view(1, 1, config["block_size"], config["block_size"]))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if hasattr(F, 'scaled_dot_product_attention'):
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        y = self.dropout(y)
        return y


class MLP(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.c_fc = nn.Linear(n_embed, 4 * n_embed)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embed, n_embed)
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embed)
        self.attn = CausalSelfAttention(n_embed, n_head)
        self.ln_2 = nn.LayerNorm(n_embed)
        self.mlp = MLP(n_embed)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class SmartGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, config["n_embed"])
        self.wpe = nn.Embedding(config["block_size"], config["n_embed"])
        self.drop = nn.Dropout(config['dropout'])

        self.blocks = nn.ModuleList([
            TransformerBlock(config["n_embed"], config["n_head"])
            for _ in range(config["n_layer"])
        ])

        self.ln_f = nn.LayerNorm(config["n_embed"])
        self.lm_head = nn.Linear(config["n_embed"], vocab_size, bias=False)

        self.wte.weight = self.lm_head.weight

    def forward(self, idx):
        B, T = idx.shape
        assert T <= config["block_size"]

        tok_emb = self.wte(idx)
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


# ========== ЗАГРУЗКА МОДЕЛИ ==========
def load_model():
    print("=" * 60)
    print("ЗАГРУЗКА МОДЕЛИ")
    print("=" * 60)

    with open("cache/vocab.json", "r", encoding="utf-8") as f:
        vocab_data = json.load(f)

    stoi = vocab_data['stoi']
    itos = {int(k): v for k, v in vocab_data['itos'].items()}
    vocab_size = vocab_data['vocab_size']
    special_tokens = vocab_data.get('special_tokens', {
        '<BOS>': 0, '<EOS>': 1, '<Q>': 2, '<A>': 3, '<PAD>': 4
    })

    print(f"Загружен словарь: {vocab_size} токенов")

    checkpoint = torch.load("best_model.pth", map_location='cpu')
    model = SmartGPT(vocab_size)

    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
        print(f"Модель загружена! Шаг: {checkpoint.get('step', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        print("Модель загружена!")

    return model, stoi, itos, special_tokens


def should_search_online(query):
    query_lower = query.lower()
    search_patterns = [
        r'^что такое\s',
        r'^кто такой\s',
        r'^что значит\s',
        r'^что означает\s',
        r'^определение\s',
        r'^объясните\s',
        r'^расскажите о\s',
    ]
    for pattern in search_patterns:
        if re.match(pattern, query_lower):
            return True
    return False


@torch.no_grad()
def generate_response(model, prompt, stoi, itos, special_tokens,
                      max_new_tokens=100, temperature=0.7, top_k=40):
    model.eval()
    device = next(model.parameters()).device

    prompt = f"<BOS><Q>{prompt}<A>"

    context = []
    for ch in prompt:
        if ch in stoi:
            context.append(stoi[ch])
        else:
            context.append(special_tokens.get('<PAD>', 4))

    context = torch.tensor(context, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        if context.size(1) > config["block_size"]:
            context = context[:, -config["block_size"]:]

        logits = model(context)
        logits = logits[0, -1, :] / temperature

        if top_k > 0:
            top_k_val = min(top_k, len(logits))
            top_k_logits, _ = torch.topk(logits, top_k_val)
            indices_to_remove = logits < top_k_logits[..., -1, None]
            logits[indices_to_remove] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)

        if next_token.item() == special_tokens.get('<EOS>', 1):
            break

        if len(context[0]) > 50 and next_token.item() == stoi.get('.', 0):
            break

        context = torch.cat([context, next_token.unsqueeze(0)], dim=1)

    response = ''
    for t in context[0]:
        token = itos[t.item()]
        if token not in special_tokens:
            response += token

    if '<A>' in response:
        response = response.split('<A>')[-1].strip()

    response = response.replace('<EOS>', '').replace('<BOS>', '').replace('<Q>', '').strip()

    if len(response) > 300:
        response = response[:300] + "..."

    return response if response else "Извините, я не могу ответить."


# ========== ОСНОВНОЙ КЛАСС АССИСТЕНТА ==========
class VoiceAssistant:
    def __init__(self):
        self.model = None
        self.stoi = None
        self.itos = None
        self.special_tokens = None
        self.searcher = None
        self.synthesizer = None
        self.recognizer = None
        self.esp32 = None
        self.running = True
        self.use_voice = True
        self.esp32_enabled = False

    def initialize(self):
        print("\n" + "=" * 60)
        print("ГОЛОСОВОЙ ПОМОЩНИК С ESP32")
        print("=" * 60)

        # Загружаем модель
        self.model, self.stoi, self.itos, self.special_tokens = load_model()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        self.model.eval()

        # Инициализируем поиск
        self.searcher = InternetSearch()

        # Подключаемся к ESP32
        print("\nПоиск ESP32...")
        self.esp32 = ESP32Connector()
        if self.esp32.connect():
            self.esp32_enabled = True
            print("ESP32 готов к работе!")
            self.esp32.beep()  # Звуковой сигнал
        else:
            print("ESP32 не найден. Буду использовать локальный TTS")

        # Инициализируем голосовые компоненты
        print("\nИнициализация TTS...")
        self.synthesizer = SpeechSynthesizer()
        if self.esp32_enabled:
            self.synthesizer.set_esp32(self.esp32)

        print("Инициализация распознавания речи...")
        self.recognizer = SpeechRecognizer(wake_word=VOICE_CONFIG["wake_word"])

        if not self.recognizer.available:
            self.use_voice = False

        print("\n" + "=" * 60)
        print("ГОТОВ К РАБОТЕ")
        print("=" * 60)

        if self.esp32_enabled:
            print("Аудио выход: ESP32 + динамик")
        else:
            print("Аудио выход: локальный TTS")

        if self.use_voice:
            print(f"Активационное слово: {VOICE_CONFIG['wake_word'].upper()}")
            print("Скажите активационное слово, затем задайте вопрос")
        else:
            print("Текстовый режим: нажмите Enter для вопроса")

        print("Для выхода скажите 'выход' или нажмите Ctrl+C")
        print("=" * 60 + "\n")

        # Приветствие
        welcome = "Голосовой помощник активирован"
        print(welcome)
        self.synthesizer.speak_sync(welcome)

    def process_and_respond(self, query):
        """Обработка запроса и отправка ответа на ESP32"""
        if not query:
            return

        if query.lower() in ['выход', 'пока', 'стоп', 'quit', 'exit']:
            self.running = False
            farewell = "До свидания"
            print(f"Ответ: {farewell}")
            self.synthesizer.speak_sync(farewell)
            return

        # Поиск ответа
        if should_search_online(query):
            print("  Поиск в Википедии...")
            answer = self.searcher.search_wikipedia(query)
            if answer:
                response = answer
            else:
                response = "Не удалось найти информацию в Википедии"
        else:
            print("  Генерация ответа нейросетью...")
            response = generate_response(
                self.model, query, self.stoi, self.itos, self.special_tokens,
                temperature=0.7, max_new_tokens=100
            )

        print(f"\nОтвет: {response}\n")

        # Отправляем на озвучивание (ESP32 или локальный TTS)
        print("  Озвучивание ответа...")
        self.synthesizer.speak(response)

        # Сигнал окончания через ESP32
        if self.esp32_enabled:
            self.esp32.beep()

    def run(self):
        """Основной цикл"""
        try:
            while self.running:
                if self.recognizer.listen_for_wake_word():
                    print("\n[АКТИВИРОВАН]")

                    # Сигнал активации
                    if self.esp32_enabled:
                        self.esp32.beep()
                    else:
                        self.synthesizer.speak_sync("Да")

                    command = self.recognizer.listen_for_command()

                    if command:
                        print(f"\n[ВОПРОС]: {command}")
                        self.process_and_respond(command)
                        print("\n[ОЖИДАНИЕ] Скажите 'Иван' для следующего вопроса")
                    else:
                        error_msg = "Не расслышал вопрос, повторите"
                        print(error_msg)
                        self.synthesizer.speak_sync(error_msg)

        except KeyboardInterrupt:
            print("\n\nЗавершение...")
        finally:
            if self.esp32:
                self.esp32.close()
            print("Программа завершена")


def main():
    # Проверка файлов
    if not os.path.exists("best_model.pth"):
        print("Ошибка: best_model.pth не найден!")
        return

    if not os.path.exists("cache/vocab.json"):
        print("Ошибка: cache/vocab.json не найден!")
        return

    assistant = VoiceAssistant()
    assistant.initialize()
    assistant.run()


if __name__ == "__main__":
    main()
