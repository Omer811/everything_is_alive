#!/usr/bin/env python3
"""
Legacy Input mock demo.

Simulates a “next-gen glove” introduction through a fullscreen terminal shell that
relies solely on textual storytelling. The terminal is configurable via YAML so
every color, font, and line of dialog can be tuned to match the macOS Terminal
feel.
"""

import array
import heapq
import math
import os
import time
import warnings
from enum import Enum

from PIL import Image

import pygame
import yaml


class State(Enum):
    LEGACY_NORMAL = 1
    GLOVE_INTRO = 2
    GLOVE_CALIBRATING = 3
    BYPASS_WARNING = 4
    RESISTANCE_ACTIVE = 5
    FINAL_CONFIRM = 6
    POST_SWITCH = 7


class AudioManager:
    def __init__(self):
        self.enabled = pygame.mixer.get_init() is not None
        self.keyboard_click = None
        self.mouse_breathe = None
        self.glove_ping = None
        if self.enabled:
            self.keyboard_click = self._make_tone(480, 0.08, volume=0.08)
            self.mouse_breathe = self._make_tone(240, 0.3, volume=0.1)
            self.glove_ping = self._make_tone(660, 0.12, volume=0.15)

    def _make_tone(self, freq, duration, volume=0.1):
        sample_rate = 44_100
        sample_count = int(sample_rate * duration)
        amplitude = int(32_000 * volume)
        buffer = array.array("h")
        for i in range(sample_count):
            theta = 2 * math.pi * freq * i / sample_rate
            buffer.append(int(amplitude * math.sin(theta)))
        try:
            return pygame.mixer.Sound(buffer=buffer.tobytes())
        except pygame.error:
            return None

    def play_keyboard_tick(self):
        if self.keyboard_click:
            self.keyboard_click.play()

    def play_mouse_breathe(self, volume=1.0):
        if self.mouse_breathe:
            self.mouse_breathe.set_volume(volume)
            self.mouse_breathe.play()

    def play_glove_ping(self):
        if self.glove_ping:
            self.glove_ping.play()


MAC_TERMINAL_ICON_PATHS = [
    "/System/Applications/Utilities/Terminal.app/Contents/Resources/Terminal.icns",
    "/Applications/Utilities/Terminal.app/Contents/Resources/Terminal.icns",
]


def load_macos_terminal_icon():
    for path in MAC_TERMINAL_ICON_PATHS:
        if not os.path.exists(path):
            continue
        try:
            icon_image = Image.open(path)
        except (FileNotFoundError, OSError):
            continue
        icon_image = icon_image.convert("RGBA")
        if icon_image.width > 128:
            resampling = getattr(Image, "Resampling", None)
            if resampling:
                resample_filter = getattr(resampling, "LANCZOS", None)
                if resample_filter is None:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", DeprecationWarning)
                        resample_filter = Image.LANCZOS
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    resample_filter = Image.LANCZOS
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                icon_image = icon_image.resize((128, 128), resample_filter)
        icon_data = icon_image.tobytes()
        try:
            return pygame.image.frombuffer(icon_data, icon_image.size, "RGBA")
        except pygame.error:
            return None
    return None


CONFIG_FILE = "legacy_input_config.yaml"


def normalize_color(value, default):
    if isinstance(value, (list, tuple)):
        return tuple(max(0, min(int(c), 255)) for c in value[:3])
    return default


def _build_font_config(name, size, bold=False, italic=False):
    return {"name": name, "size": size, "bold": bold, "italic": italic}


def load_config(path=CONFIG_FILE):
    defaults = {
        "terminal": {
            "background": (255, 243, 163),
            "border": (180, 170, 150),
            "text": (0, 0, 0),
            "prompt": (0, 0, 0),
            "cursor": (0, 0, 0),
            "prompt_prefix": "omerdvora@MacBook-Air / %",
        },
        "fonts": {
            "main": _build_font_config("SF Mono", 15),
            "panel": _build_font_config("SF Mono", 12),
        },
        "story": {
            "ready": "System ready. Legacy devices nominal.",
            "initial": [
                "Type the demo commands to keep the story on track.",
                "Try: ./glove_init → glove calibrate --quick → glove commit",
            ],
            "connect_prompt": "Connect the accessory, then press Y to continue.",
            "connect_confirmation": "Device connected. Preparing calibration routines...",
            "calibration_steps": [
                "Thermal level setpoints stabilizing...",
                "Flex sensors mapping fingers 1–5...",
                "Haptic feedback loop verifying micro-pulses...",
            ],
            "calibration_interval": 0.9,
            "calibration_done": "Calibration complete. Legacy devices still audible.",
            "glove_intro": [
                "Glove accessory attached. Initializing firmware...",
                "Legacy keyboard/mouse running diagnostics.",
            ],
            "calibration": "Calibration started. Ensuring click-trace fidelity.",
            "bypass": "NOTICE: primary input will switch to GLOVE. Legacy input becomes secondary.",
            "resistance": "Legacy devices are pleading. They slow, but keep letting you work.",
            "final_prompt": "Switch permanently to glove input? [Y/n]",
            "post_switch": "Glove input engaged. Movement is instant, empty, and efficient.",
            "warning": "Legacy input sensors humming louder... Something is changing.",
            "suggestion": "autocomplete suggestion: {suggestion} (keyboard pleads)",
            "command_unknown": "Command uncertain; legacy peripherals whisper.",
        },
        "commands": [
            "./glove_init",
            "./glove_calibrate",
            "./glove status",
            "./glove attach --override-input",
        ],
        "timing": {
            "response_delay_ms": 50,
        },
    }
    if not os.path.exists(path):
        return defaults
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError):
        return defaults
    terminal = defaults["terminal"].copy()
    if "terminal" in data:
        terminal_config = data["terminal"] or {}
        terminal["prompt_prefix"] = terminal_config.get("prompt_prefix", terminal["prompt_prefix"])
        for key in ("background", "border", "text", "prompt", "cursor"):
            terminal[key] = normalize_color(terminal_config.get(key), terminal[key])
    fonts = defaults["fonts"].copy()
    if "fonts" in data:
        font_config = data["fonts"] or {}
        for entry in fonts:
            entry_data = fonts[entry].copy()
            override = font_config.get(entry)
            if isinstance(override, dict):
                entry_data.update(
                    {k: override[k] for k in ("name", "size", "bold", "italic") if k in override}
                )
            else:
                entry_data["name"] = override or entry_data["name"]
                entry_data["size"] = font_config.get(f"{entry}_size", entry_data["size"])
                entry_data["bold"] = bool(font_config.get(f"{entry}_bold", entry_data["bold"]))
                entry_data["italic"] = bool(font_config.get(f"{entry}_italic", entry_data["italic"]))
            fonts[entry] = entry_data
    commands = defaults.get("commands", []).copy()
    story = defaults["story"].copy()
    timing = defaults.get("timing", {}).copy()
    if "story" in data:
        story_config = data["story"] or {}
        for key, default_value in defaults["story"].items():
            override = story_config.get(key)
            story[key] = override if override is not None else default_value
    if "commands" in data:
        commands = data["commands"] or commands
    if "timing" in data:
        timing.update(data["timing"] or {})
    return {
        "terminal": terminal,
        "fonts": fonts,
        "story": story,
        "commands": commands,
        "timing": timing,
    }


def _create_font(font_spec):
    bold = bool(font_spec.get("bold", False))
    italic = bool(font_spec.get("italic", False))
    font_path = pygame.font.match_font(font_spec["name"], bold=bold, italic=italic)
    if font_path:
        return pygame.font.Font(font_path, font_spec["size"])
    return pygame.font.SysFont(font_spec["name"], font_spec["size"], bold=bold, italic=italic)


class LegacyInputDemo:
    def __init__(self, screen_rect, config):
        self.screen_rect = screen_rect
        self.state = State.LEGACY_NORMAL
        self.state_entry = time.time()
        self.font = _create_font(config["fonts"]["main"])
        self.small_font = _create_font(config["fonts"]["panel"])
        self.terminal_lines = []
        self.command_line = ""
        self.command_history = []
        self.last_key_time = time.time()
        self.cursor_visible = True
        self.cursor_blink_interval = 0.5
        self.last_cursor_toggle = time.time()
        self.await_connection = False
        self.calibration_queue = []
        self.calibration_done_time = 0.0
        self.calibration_done_logged = False
        self.terminal_palette = {
            "background": pygame.Color(*config["terminal"]["background"]),
            "border": pygame.Color(*config["terminal"]["border"]),
            "text": pygame.Color(*config["terminal"]["text"]),
            "prompt": pygame.Color(*config["terminal"]["prompt"]),
            "cursor": pygame.Color(*config["terminal"]["cursor"]),
        }
        self.prompt_prefix = config["terminal"].get("prompt_prefix", "omerdvora@MacBook-Air / %")
        self.story = config.get("story", {})
        self.audio = AudioManager()
        self.initial_story_pending = list(self.story.get("initial", []))
        self.initial_story_shown = False
        timing = config.get("timing", {})
        self.response_delay = timing.get("response_delay_ms", 50) / 1000.0
        self.pending_outputs = []
        self.next_output_time = time.time()
        self.available_commands = config.get("commands", [])
        self.command_index = None
        self.await_glove_worn = False
        self.selection_start_line = None
        self.selection_end_line = None
        self.selection_active = False

    def log(self, text, with_prefix=False, delay=0.0):
        prefix = "[terminal] " if with_prefix else ""
        content = f"{prefix}{text}"
        self.schedule_output(content, delay)

    def schedule_output(self, text, delay=0.0):
        base = max(time.time(), self.next_output_time)
        release = base + delay
        self.next_output_time = release + 0.0001
        heapq.heappush(self.pending_outputs, (release, text))

    def flush_pending_outputs(self, now):
        while self.pending_outputs and self.pending_outputs[0][0] <= now:
            _, text = heapq.heappop(self.pending_outputs)
            self.terminal_lines.append((text, self.terminal_palette["text"]))

    def log_story(self, key):
        entry = self.story.get(key)
        if isinstance(entry, list):
            for line in entry:
                self.log(line, delay=self.response_delay)
        elif entry:
            self.log(entry, delay=self.response_delay)

    def reveal_initial_story(self):
        if self.initial_story_shown:
            return
        self.log_story("ready")
        for line in self.initial_story_pending:
            self.log(line, delay=self.response_delay)
        self.initial_story_shown = True

    def cycle_command_history(self, direction):
        if not self.available_commands:
            return
        length = len(self.available_commands)
        if self.command_index is None:
            self.command_index = 0 if direction > 0 else length - 1
        else:
            self.command_index = (self.command_index + direction) % length
        self.command_line = self.available_commands[self.command_index]

    def clear_selection(self):
        self.selection_start_line = None
        self.selection_end_line = None
        self.selection_active = False

    def y_to_line_index(self, y):
        line_height = self.font.get_linesize()
        base_y = 10
        if line_height == 0:
            return 0
        idx = int((y - base_y) // line_height)
        return max(0, min(len(self.terminal_lines) - 1, idx)) if self.terminal_lines else 0

    def handle_mouse_down(self, pos):
        idx = self.y_to_line_index(pos[1])
        self.selection_start_line = idx
        self.selection_end_line = idx
        self.selection_active = True

    def handle_mouse_motion(self, pos):
        if not self.selection_active:
            return
        idx = self.y_to_line_index(pos[1])
        self.selection_end_line = idx

    def handle_mouse_up(self, pos):
        if not self.selection_active:
            return
        self.selection_active = False
        idx = self.y_to_line_index(pos[1])
        self.selection_end_line = idx

    def copy_selection(self):
        if self.selection_start_line is None or self.selection_end_line is None:
            return
        start = min(self.selection_start_line, self.selection_end_line)
        end = max(self.selection_start_line, self.selection_end_line)
        end = min(end, len(self.terminal_lines) - 1)
        if start < 0 or end < 0 or start > end:
            return
        lines = [line for line, _ in self.terminal_lines[start : end + 1]]
        text = "\n".join(lines)
        try:
            pygame.scrap.put(pygame.SCRAP_TEXT, text.encode("utf-8"))
        except pygame.error:
            pass

    def transition(self, new_state):
        if new_state == self.state:
            return
        self.state = new_state
        self.state_entry = time.time()
        if new_state == State.GLOVE_INTRO:
            self.log_story("glove_intro")
        elif new_state == State.GLOVE_CALIBRATING:
            self.log_story("calibration")
        elif new_state == State.BYPASS_WARNING:
            self.log_story("bypass")
        elif new_state == State.RESISTANCE_ACTIVE:
            self.log_story("resistance")
        elif new_state == State.FINAL_CONFIRM:
            self.log_story("final_prompt")
        elif new_state == State.POST_SWITCH:
            self.log_story("post_switch")
            self.audio.play_glove_ping()

    def update(self, dt):
        now = time.time()
        idle_time = now - self.last_key_time
        self.cursor_blink_interval = 0.5 if idle_time < 1.5 else 1.1
        if now - self.last_cursor_toggle > self.cursor_blink_interval:
            self.cursor_visible = not self.cursor_visible
            self.last_cursor_toggle = now
        self.flush_pending_outputs(now)
        self.process_calibration_queue(now)

    def process_calibration_queue(self, now):
        while self.calibration_queue and now >= self.calibration_queue[0][0]:
            _, line = self.calibration_queue.pop(0)
            self.log(line)
        if not self.calibration_queue and not self.calibration_done_logged and now >= self.calibration_done_time:
            self.log_story("calibration_done")
            self.calibration_done_logged = True
            self.await_glove_worn = True
            self.log_story("wear_prompt")

    def start_connection_sequence(self):
        if self.await_connection:
            return
        self.await_connection = True
        self.log_story("connect_prompt")

    def confirm_connection(self):
        if not self.await_connection:
            return
        self.await_connection = False
        self.log_story("connect_confirmation")
        steps = self.story.get("calibration_steps", [])
        interval = self.story.get("calibration_interval", 0.8)
        dot_count = self.story.get("calibration_dots", 3)
        dot_text = self.story.get("calibration_dot_text", ".")
        dot_interval = self.story.get("calibration_dot_interval", 0.5)
        now = time.time()
        schedule = []
        current = now
        for step in steps:
            current += interval
            schedule.append((current, step))
            for _ in range(dot_count):
                current += dot_interval
                schedule.append((current, dot_text))
        self.calibration_queue = schedule
        self.calibration_done_time = (current + 0.4) if schedule else now
        self.calibration_done_logged = False
        self.transition(State.GLOVE_CALIBRATING)

    def handle_key(self, event):
        if event.key == pygame.K_ESCAPE:
            pygame.event.post(pygame.event.Event(pygame.QUIT))
            return
        if self.await_connection and event.key == pygame.K_y:
            self.confirm_connection()
            return
        if self.await_glove_worn and event.key == pygame.K_y:
            self.await_glove_worn = False
            self.log_story("wear_confirmation")
            self.transition(State.GLOVE_INTRO)
            return
        if event.key == pygame.K_RETURN:
            self.submit_command()
            return
        if event.key == pygame.K_UP:
            self.cycle_command_history(-1)
            return
        if event.key == pygame.K_DOWN:
            self.cycle_command_history(1)
            return
        if event.key == pygame.K_BACKSPACE:
            if self.command_line:
                self.command_line = self.command_line[:-1]
            return
        if event.unicode and event.unicode.isprintable():
            self.command_index = None
            self.command_line += event.unicode
            self.last_key_time = time.time()
            self.audio.play_keyboard_tick()
            if len(self.command_line) > 140:
                self.command_line = self.command_line[:140]

    def submit_command(self):
        cmd = self.command_line.strip()
        self.command_line = ""
        if not cmd:
            return
        if not self.initial_story_shown:
            self.reveal_initial_story()
        self.command_history.append(cmd)
        self.log(f"> {cmd}", delay=self.response_delay)
        self.command_index = None
        normalized = cmd.lower()
        if normalized in ("./glove_calibrate", "./glove_calibration", "./calibrate_glove"):
            self.start_connection_sequence()
            return
        if normalized == "./glove_init":
            self.transition(State.GLOVE_INTRO)
        elif normalized.startswith("glove status"):
            self.log("glove status: battery healthy, inputs synced.", delay=self.response_delay)
        elif normalized.startswith("glove attach"):
            self.transition(State.BYPASS_WARNING)
        elif normalized.startswith("glove enable"):
            self.transition(State.RESISTANCE_ACTIVE)
        elif normalized.startswith("glove commit"):
            self.transition(State.FINAL_CONFIRM)
        else:
            self.log(
                self.story.get("command_unknown", "Command uncertain; legacy peripherals whisper."),
                delay=self.response_delay,
            )

    def draw_terminal(self, surface):
        surface.fill(self.terminal_palette["background"])
        base_y = 10
        line_height = self.font.get_linesize()
        max_lines = max(1, (self.screen_rect.height - 20) // line_height)
        visible_lines = self.terminal_lines[-max_lines:]
        if self.selection_start_line is not None and self.selection_end_line is not None:
            start = min(self.selection_start_line, self.selection_end_line)
            end = max(self.selection_start_line, self.selection_end_line)
            start = max(0, start)
            end = min(len(self.terminal_lines) - 1, end)
            for idx in range(start, end + 1):
                rect = pygame.Rect(10, base_y + idx * line_height, self.screen_rect.width - 20, line_height)
                highlight = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
                highlight.fill((200, 200, 200, 90))
                surface.blit(highlight, rect.topleft)
        for idx, (line, color) in enumerate(visible_lines):
            text_surface = self.font.render(line, True, color)
            surface.blit(text_surface, (10, base_y + idx * line_height))
        prompt = f"{self.prompt_prefix} {self.command_line}"
        prompt_surface = self.font.render(prompt, True, self.terminal_palette["prompt"])
        prompt_y = base_y + len(visible_lines) * line_height
        prompt_y = min(prompt_y, self.screen_rect.height - line_height - 10)
        surface.blit(prompt_surface, (10, prompt_y))
        if self.cursor_visible:
            cursor_x = 10 + prompt_surface.get_width() + 4
            cursor_y = prompt_y
            cursor_width = 3
            cursor_height = max(4, line_height - 6)
            pygame.draw.rect(
                surface,
                self.terminal_palette["cursor"],
                (cursor_x, cursor_y, cursor_width, cursor_height),
            )

    def render(self, surface):
        self.draw_terminal(surface)

    def resize_screen(self, new_rect):
        self.screen_rect = pygame.Rect(new_rect)


def main():
    pygame.mixer.pre_init(44100, -16, 1, 512)
    pygame.init()
    screen = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
    pygame.display.set_caption("Last Peripheral Demo")
    icon_surface = load_macos_terminal_icon()
    if icon_surface:
        pygame.display.set_icon(icon_surface)
    clock = pygame.time.Clock()
    config = load_config()
    demo = LegacyInputDemo(screen.get_rect(), config)
    running = True
    start_time = time.time()
    max_runtime = 0.0
    runtime_env = os.environ.get("LEGACY_INPUT_DEMO_RUNTIME")
    if runtime_env:
        try:
            max_runtime = float(runtime_env)
        except ValueError:
            max_runtime = 0.0
    while running:
        dt = clock.tick(60) / 1000.0
        if max_runtime and (time.time() - start_time) >= max_runtime:
            running = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                demo.resize_screen(screen.get_rect())
            elif event.type == pygame.KEYDOWN:
                demo.handle_key(event)
        demo.update(dt)
        demo.render(screen)
        pygame.display.flip()
    pygame.quit()


if __name__ == "__main__":
    main()
