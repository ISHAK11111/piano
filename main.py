import threading
import math
import numpy as np
import sounddevice as sd
import tkinter as tk
from dataclasses import dataclass, field
from typing import List, Optional

# --------------------------------
# Конфигурация (как в исходном коде)
# --------------------------------
SAMPLE_RATE = 22050
A4_KEY_NUMBER = 49
A4_FREQ = 440.0

# Базовая длительность ноты без удержания пробела (сек)
NOTE_DURATION = 0.7

# ADSR-параметры
ATTACK = 0.05
DECAY = 0.10
SUSTAIN_LEVEL = 0.7
RELEASE = 0.40

# Гармоники (отношение частоты, амплитуда)
HARMONICS = [
    (1.0, 1.0),
    (2.0, 0.6),
    (3.0, 0.4),
    (4.0, 0.3),
    (5.0, 0.2),
    (6.0, 0.15),
    (7.0, 0.1),
    (8.0, 0.08),
]

# Глобальная громкость частичных и мастера
PARTIALS_AMPLITUDE = 0.35
MASTER_GAIN = 0.9

# Клавиатура
START_KEY = 36     # C3
NUM_OCTAVES = 2    # количество октав

# Рисование клавиатуры
WHITE_W, WHITE_H = 50, 160
BLACK_W, BLACK_H = 32, 100
BLACK_OFFSET = 0.68  # смещение чёрных между белыми (в долях ширины белой)


# --------------------------------
# Вспомогательные функции
# --------------------------------
def key_to_freq(key_number: int) -> Optional[float]:
    if not 1 <= key_number <= 88:
        return None
    n = key_number - A4_KEY_NUMBER
    return A4_FREQ * (2 ** (n / 12.0))


@dataclass
class KeyInfo:
    full: str
    freq: float
    is_black: bool
    white_index: Optional[int] = None
    white_before_index: Optional[int] = None
    canvas_id: Optional[int] = None


def build_keys(start_key=36, num_octaves=2) -> List[KeyInfo]:
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    num_keys = num_octaves * 12 + 1

    keys: List[KeyInfo] = []
    white_count = 0  # индекс белых клавиш слева направо
    for i in range(num_keys):
        key_number = start_key + i
        note_index = key_number % 12
        note_name = note_names[note_index]
        is_black = ('#' in note_name)
        octave = (key_number + 8) // 12
        full_name = f"{note_name}{octave}"
        freq = key_to_freq(key_number)

        if not is_black:
            keys.append(KeyInfo(full=full_name, freq=freq, is_black=False, white_index=white_count))
            white_count += 1
        else:
            keys.append(KeyInfo(full=full_name, freq=freq, is_black=True, white_before_index=max(0, white_count - 1)))
    return keys


# --------------------------------
# Аудио: аддитивный синтез + ADSR в real-time
# --------------------------------
TWO_PI = 2.0 * math.pi

@dataclass
class ActiveNote:
    key_full: str
    freq: float
    start_frame: int
    phases: List[float] = field(default_factory=list)
    release_frame: Optional[int] = None
    release_amp: Optional[float] = None
    auto_release_frame: Optional[int] = None

    def __post_init__(self):
        if not self.phases:
            self.phases = [0.0 for _ in HARMONICS]


def env_level_held(t: float) -> float:
    """
    Амплитуда ADS без релиза (Attack->Decay->Sustain).
    t — время с начала ноты (сек).
    """
    if t < 0:
        return 0.0
    if ATTACK > 0 and t < ATTACK:
        return t / ATTACK
    t2 = t - ATTACK
    if DECAY > 0 and t2 < DECAY:
        # линейный спад от 1 до SUSTAIN_LEVEL
        return 1.0 - (1.0 - SUSTAIN_LEVEL) * (t2 / DECAY)
    return SUSTAIN_LEVEL


def env_level(note: ActiveNote, frame: int) -> float:
    """
    Амплитуда ADSR для конкретного кадра.
    """
    t = (frame - note.start_frame) / SAMPLE_RATE
    if note.release_frame is None:
        return env_level_held(t)
    # Если релиз уже начался
    if frame < note.release_frame:
        return env_level_held(t)
    # Релиз
    rel_t = (frame - note.release_frame) / SAMPLE_RATE
    base = note.release_amp if note.release_amp is not None else env_level_held((note.release_frame - note.start_frame) / SAMPLE_RATE)
    if rel_t >= RELEASE:
        return 0.0
    return max(0.0, base * (1.0 - (rel_t / RELEASE)))


class PianoApp:
    def __init__(self):
        # GUI
        self.root = tk.Tk()
        self.root.title("Python Piano (Real‑Time, Spacebar Sustain)")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        tk.Label(self.root, text="Кликните по клавише для звука. Удерживайте ПРОБЕЛ — нота звучит дольше. Отпустите — затухает.",
                 fg="#555").pack(padx=8, pady=(8, 2), anchor="w")

        self.status_var = tk.StringVar()
        self.status_var.set("Sustain: OFF")
        self.status_label = tk.Label(self.root, textvariable=self.status_var)
        self.status_label.pack(padx=8, pady=(2, 8), anchor="w")

        self.keys_data: List[KeyInfo] = build_keys(START_KEY, NUM_OCTAVES)
        white_count = sum(1 for k in self.keys_data if not k.is_black)
        width = white_count * WHITE_W
        height = WHITE_H
        self.canvas = tk.Canvas(self.root, width=width, height=height, bg="#ddd", highlightthickness=0)
        self.canvas.pack(padx=8, pady=8)
        self.canvas.focus_set()  # Чтобы ловить пробел

        self.draw_keyboard()

        # Состояние
        self.sustain_event = threading.Event()  # True -> sustain ON (hold space)
        self.notes_lock = threading.Lock()
        self.active_notes: List[ActiveNote] = []
        self.global_frame = 0  # счётчик воспроизведённых сэмплов
        self.running = True

        # Аудио
        self.stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            blocksize=256,
            latency='low',
            callback=self.audio_callback
        )
        self.stream.start()

        # События
        self.root.bind("<KeyPress-space>", self.on_space_down)
        self.root.bind("<KeyRelease-space>", self.on_space_up)

    # -------------------- GUI --------------------
    def draw_keyboard(self):
        # сначала белые
        for k in self.keys_data:
            if not k.is_black:
                x = k.white_index * WHITE_W
                k.canvas_id = self.canvas.create_rectangle(
                    x, 0, x + WHITE_W, WHITE_H, fill="#ffffff", outline="#000000"
                )
                # bind click
                self.canvas.tag_bind(k.canvas_id, "<Button-1>", lambda e, kk=k: self.handle_key_click(kk))

        # затем чёрные
        for k in self.keys_data:
            if k.is_black:
                left_white = k.white_before_index
                x = (left_white + BLACK_OFFSET) * WHITE_W - (BLACK_W / 2)
                k.canvas_id = self.canvas.create_rectangle(
                    x, 0, x + BLACK_W, BLACK_H, fill="#333333", outline="#000000"
                )
                self.canvas.tag_bind(k.canvas_id, "<Button-1>", lambda e, kk=k: self.handle_key_click(kk))

    def set_status(self):
        self.status_var.set("Sustain: ON (space)" if self.sustain_event.is_set() else "Sustain: OFF")

    def handle_key_click(self, k: KeyInfo):
        # Щелчок по клавише — запускаем ноту
        self.canvas.focus_set()
        self.start_note(k)

    def on_space_down(self, event=None):
        if not self.sustain_event.is_set():
            self.sustain_event.set()
            self.set_status()
            # отменяем авто-релиз для уже звучащих
            with self.notes_lock:
                for n in self.active_notes:
                    n.auto_release_frame = None

    def on_space_up(self, event=None):
        if self.sustain_event.is_set():
            self.sustain_event.clear()
            self.set_status()
            # отпускаем все ноты
            self.release_all_notes()

    def on_close(self):
        self.running = False
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass
        self.root.destroy()

    # -------------------- Логика нот --------------------
    def start_note(self, k: KeyInfo):
        if k.freq is None:
            return
        with self.notes_lock:
            start_frame = self.global_frame
            note = ActiveNote(
                key_full=k.full,
                freq=k.freq,
                start_frame=start_frame
            )
            # авто-релиз если sustain не удерживается
            if not self.sustain_event.is_set():
                rel_delay = max(0.0, NOTE_DURATION - RELEASE)
                note.auto_release_frame = start_frame + int(rel_delay * SAMPLE_RATE)
            self.active_notes.append(note)

    def release_note(self, note: ActiveNote, frame_now: int):
        if note.release_frame is not None:
            return
        note.release_frame = frame_now
        # зафиксируем амплитуду на момент релиза
        t_rel = (frame_now - note.start_frame) / SAMPLE_RATE
        note.release_amp = env_level_held(t_rel)

    def release_all_notes(self):
        with self.notes_lock:
            frame_now = self.global_frame
            for n in self.active_notes:
                if n.release_frame is None:
                    self.release_note(n, frame_now)

    # -------------------- Аудио коллбэк --------------------
    def audio_callback(self, outdata, frames, time_info, status):
        if status:
            # можно печатать/логировать
            pass

        buf = np.zeros(frames, dtype=np.float32)
        # берём нотный список под замком на время блока
        with self.notes_lock:
            # локальные ссылки для скорости
            notes = self.active_notes
            sr = SAMPLE_RATE
            gf = self.global_frame

            # микшируем
            to_remove_indices = set()

            for i in range(frames):
                s = 0.0
                frame_idx = gf + i

                # суммируем все ноты
                for ni, note in enumerate(notes):
                    # авто-релиз, если пора и sustain выключен
                    if (note.auto_release_frame is not None
                        and not self.sustain_event.is_set()
                        and note.release_frame is None
                        and frame_idx >= note.auto_release_frame):
                        self.release_note(note, frame_idx)

                    amp_env = env_level(note, frame_idx)
                    if amp_env <= 0.0:
                        # если релиз завершился, пометим ноту для удаления позже
                        if (note.release_frame is not None
                            and frame_idx >= note.release_frame + int(RELEASE * sr) + 1):
                            to_remove_indices.add(ni)
                        continue

                    # аддитивный синтез: сумма синусов гармоник
                    for h_idx, (ratio, h_amp) in enumerate(HARMONICS):
                        phi = note.phases[h_idx]
                        s += math.sin(phi) * (h_amp * PARTIALS_AMPLITUDE) * amp_env
                        # приращение фазы
                        phi += TWO_PI * (note.freq * ratio) / sr
                        # лёгкая нормализация фазы
                        if phi >= TWO_PI:
                            phi -= TWO_PI
                        note.phases[h_idx] = phi

                # мастер-гейн и клиппинг
                s *= MASTER_GAIN
                if s > 1.0:
                    s = 1.0
                elif s < -1.0:
                    s = -1.0

                buf[i] = s

            # удалим завершённые ноты (с конца, чтобы индексы не смещать)
            if to_remove_indices:
                # конвертируем в отсортированный список убыв.
                for idx in sorted(to_remove_indices, reverse=True):
                    if 0 <= idx < len(notes):
                        notes.pop(idx)

            # продвинем глобальный счётчик кадров
            self.global_frame += frames

        outdata[:, 0] = buf

    # -------------------- Запуск --------------------
    def run(self):
        self.set_status()
        self.root.mainloop()


if __name__ == "__main__":
    app = PianoApp()
    print("🎹 Python Piano (реальное время, управление длительностью через пробел)")
    app.run()