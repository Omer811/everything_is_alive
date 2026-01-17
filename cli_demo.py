#!/usr/bin/env python3
"""
CLI version of the Legacy Input demo.

Runs entirely in the terminal so you keep the native prompt look+feel while the
storyline and peripheral emotions are replayed via scripted prints and timed
delays. ANSI selection, copy/paste, and genuine OS input all remain available.
"""

import os
import math
import random
import readline
import shutil
import subprocess
import sys
import termios
import threading
import time
import tty

import yaml

from gpt_client import GPTResponder

try:
    import pyautogui
except ImportError:
    pyautogui = None

CONFIG_FILE = "legacy_input_config.yaml"


def normalize_color(value, default):
    if isinstance(value, (list, tuple)):
        return tuple(max(0, min(int(c), 255)) for c in value[:3])
    return default


def _build_font_config(name, size, bold=False, italic=False):
    return {"name": name, "size": size, "bold": bold, "italic": italic}


def _ansi_sequence(color, code):
    if not color or len(color) < 3:
        return ""
    try:
        r, g, b = color[:3]
    except (TypeError, ValueError):
        return ""
    return f"\x1b[{code};2;{r};{g};{b}m"


class ScreenMonitor:
    def __init__(self):
        self.columns = 80
        self.lines = 24
        self.update()

    def update(self):
        try:
            size = os.get_terminal_size()
        except OSError:
            size = os.terminal_size((80, 24))
        self.columns = size.columns
        self.lines = size.lines


class MockMouse:
    def __init__(self, monitor):
        self.monitor = monitor
        self.row = 0
        self.col = 0
        self.state = "idle"

    def move_to_line(self, index, buffer):
        if not buffer:
            self.row = 0
            self.col = 0
            self.state = "idle"
            return 0
        self.monitor.update()
        target = max(0, min(index, len(buffer) - 1))
        line = buffer[target]
        self.row = target
        self.col = min(len(line), self.monitor.columns - 1)
        self.state = "focused"
        return self.row

    def move_to_text(self, query, buffer):
        if not buffer:
            return None
        search = query.lower() if query else ""
        candidate = None
        for idx in range(len(buffer) - 1, -1, -1):
            line_text = buffer[idx]
            if not search or search in line_text.lower():
                if not line_text.startswith("> "):
                    return self.move_to_line(idx, buffer)
                candidate = idx
        if candidate is not None:
            return self.move_to_line(candidate, buffer)
        return None


class RealMouseMover:
    def __init__(self, driver=None):
        self.driver = driver or pyautogui
        self.enabled = self.driver is not None

    def move_to(self, target_pos, duration=0.5):
        if not self.enabled or not target_pos or len(target_pos) < 2:
            return False
        try:
            self.driver.moveTo(target_pos[0], target_pos[1], duration=duration, _pause=False)
            return True
        except Exception:
            return False

    def locate_template(self, path, region=None):
        if not self.enabled or not path or not os.path.exists(path):
            return None
        locate = getattr(self.driver, "locateOnScreen", None)
        if not callable(locate):
            return None
        kwargs = {"region": tuple(region)} if region else {}
        try:
            result = locate(path, **kwargs)
        except Exception:
            return None
        if not result:
            return None
        center_x = result.left + result.width / 2
        center_y = result.top + result.height / 2
        return (center_x, center_y)


class MouseCreepRunner:
    def __init__(self, mover, logger):
        self.mover = mover
        self.logger = logger or NullLogger()
        self.thread = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.target = None
        self.config = {}
        self.current = None
        self.current_state = None
        self.dynamic_state = None
        self.dynamic_growth = {}

    def _get_position(self):
        if not self.mover.enabled:
            return None
        driver = self.mover.driver
        getter = getattr(driver, "position", None)
        if not callable(getter):
            return None
        try:
            pos = getter()
        except Exception:
            return None
        if hasattr(pos, "x") and hasattr(pos, "y"):
            return float(pos.x), float(pos.y)
        try:
            return float(pos[0]), float(pos[1])
        except Exception:
            return None

    def start(self, target, config, state=None):
        if not target:
            self.logger.log("mouse_creep_request", target="none", state=state, enabled=self.mover.enabled, reason="no_target")
            self.stop()
            return
        if not self.mover.enabled:
            self.logger.log("mouse_creep_request", target=f"{target}", state=state, enabled=False, reason="driver_disabled")
            self.stop()
            return
        with self.lock:
            if (
                self.thread
                and self.thread.is_alive()
                and self.target == tuple(target)
                and self.config == config
                and self.current_state == state
            ):
                return
            self.stop()
            self.set_target(tuple(target))
            self.config = config.copy() if isinstance(config, dict) else {}
            self.current_state = state
            self.dynamic_state = {
                "interval": float(self.config.get("step_interval", 5.0)),
                "min_step": float(self.config.get("min_step", 1.0)),
                "max_step": float(self.config.get("max_step", 3.0)),
                "bias": float(self.config.get("bias", 0.0)),
                "jitter": float(self.config.get("jitter", 0.0)),
            }
            self.dynamic_growth = {
                "speed_increase": float(self.config.get("speed_increase", 0.0)),
                "interval_decay": float(self.config.get("interval_decay", 0.0)),
                "bias_increase": float(self.config.get("bias_increase", 0.0)),
                "jitter_growth": float(self.config.get("jitter_growth", 0.0)),
                "min_interval": float(self.config.get("min_interval", 0.1)),
            }
            self.stop_event.clear()
            self.logger.log(
                "mouse_creep_request",
                target=f"{target}",
                state=state,
                enabled=self.mover.enabled,
                keep_running=self.config.get("keep_running", False),
            )
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def set_target(self, target):
        self.target = target
        self.current = None

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=0.1)
        self.thread = None
        self.stop_event.clear()
        self.current_state = None
        self.dynamic_state = None
        self.dynamic_growth = {}

    def is_running(self):
        return bool(self.thread and self.thread.is_alive())

    def update_state(self, state):
        with self.lock:
            self.current_state = state

    def _run(self):
        interval = float(self.config.get("step_interval", 5.0))
        min_step = float(self.config.get("min_step", 1.0))
        max_step = float(self.config.get("max_step", 3.0))
        bias = float(self.config.get("bias", 0.8))
        jitter = float(self.config.get("jitter", 1.5))
        duration = float(self.config.get("duration", 0.4))
        dynamic = self.dynamic_state or {
            "interval": interval,
            "min_step": min_step,
            "max_step": max_step,
            "bias": bias,
            "jitter": jitter,
        }
        growth = self.dynamic_growth or {}
        interval_decay = growth.get("interval_decay", 0.0)
        speed_increase = growth.get("speed_increase", 0.0)
        bias_increase = growth.get("bias_increase", 0.0)
        jitter_growth = growth.get("jitter_growth", 0.0)
        min_interval = growth.get("min_interval", 0.1)
        while not self.stop_event.is_set():
            current = self._get_position()
            if current:
                self.current = current
            if not self.current or not self.target:
                break
            dx = self.target[0] - self.current[0]
            dy = self.target[1] - self.current[1]
            dist = math.hypot(dx, dy)
            direction = (dx / dist, dy / dist) if dist else (0.0, 0.0)
            step = random.uniform(dynamic["min_step"], dynamic["max_step"])
            target_step = (direction[0] * step, direction[1] * step)
            rand_angle = random.uniform(0, 2 * math.pi)
            random_component = (math.cos(rand_angle) * dynamic["jitter"], math.sin(rand_angle) * dynamic["jitter"])
            weight = min(max(dynamic["bias"], 0.0), 1.0)
            next_x = self.current[0] + target_step[0] * weight + random_component[0] * (1 - weight)
            next_y = self.current[1] + target_step[1] * weight + random_component[1] * (1 - weight)
            next_pos = (next_x, next_y)
            success = self.mover.move_to(next_pos, duration)
            self.logger.log(
                "mouse_command",
                target=f"{next_pos}",
                bias=weight,
                duration=duration,
                state=self.current_state,
                success=success,
            )
            self.logger.log(
                "mouse_creep_step",
                step_interval=dynamic["interval"],
                step=step,
                target=f"{self.target}",
                position=f"{next_pos}",
                bias=weight,
                state=self.current_state,
                min_step=dynamic["min_step"],
                max_step=dynamic["max_step"],
                jitter=dynamic["jitter"],
            )
            self.current = next_pos
            if self.stop_event.wait(dynamic["interval"]):
                break
            dynamic["interval"] = max(min_interval, dynamic["interval"] - interval_decay)
            dynamic["min_step"] += speed_increase
            dynamic["max_step"] += speed_increase
            dynamic["max_step"] = max(dynamic["max_step"], dynamic["min_step"])
            dynamic["bias"] = min(1.0, dynamic["bias"] + bias_increase)
            dynamic["jitter"] += jitter_growth


class MockKeyboard:
    def __init__(self):
        self.tension = "calm"
        self.last_input = ""

    def record_input(self, text):
        self.last_input = text
        self.tension = "resisting" if "glove" in text.lower() else "calm"

    def describe(self):
        return f"Mock keyboard tension {self.tension}; last input '{self.last_input}'."


class MockPeripheralController:
    def __init__(self):
        self.monitor = ScreenMonitor()
        self.mouse = MockMouse(self.monitor)
        self.keyboard = MockKeyboard()
        self.line_history = []

    def record_input(self, text):
        self.keyboard.record_input(text)

    def record_line(self, text, move_mouse=False):
        self.line_history.append(text)
        if move_mouse:
            self.mouse.move_to_line(len(self.line_history) - 1, self.line_history)

    def move_mouse_to_text(self, text):
        return self.mouse.move_to_text(text, self.line_history)

    def describe_mouse(self):
        return f"Mock mouse at row {self.mouse.row}, col {self.mouse.col}, state {self.mouse.state}."

    def describe_keyboard(self):
        return self.keyboard.describe()


class EventLogger:
    def __init__(self, config=None):
        config = config or {}
        self.enabled = bool(config.get("enabled", True))
        self.path = config.get("path")
        if not self.enabled or not self.path:
            self.enabled = False
            self.path = None
            return
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    def log(self, event, **details):
        if not self.enabled:
            return
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
        detail_parts = []
        for key in sorted(details):
            detail_parts.append(f"{key}={details[key]}")
        detail_text = " ".join(detail_parts)
        line = " ".join(part for part in (timestamp, event, detail_text) if part)
        try:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            pass


class NullLogger:
    def log(self, event, **details):
        return


def load_config(path=CONFIG_FILE):
    defaults = {
        "terminal": {
            "background": (255, 243, 163),
            "border": (180, 170, 150),
            "text": (0, 0, 0),
            "prompt": (0, 0, 0),
            "cursor": (0, 0, 0),
            "use_ansi": True,
            "prompt_prefix": "omerdvora@MacBook-Air / %",
        },
        "fonts": {
            "main": _build_font_config("SF Mono", 15),
            "panel": _build_font_config("SF Mono", 12),
        },
        "story": {
            "ready": "System ready. Legacy devices nominal.",
            "initial": [],
            "connect_prompt": "Connect the accessory, then press Y to continue.",
            "connect_confirmation": "Device connected. Preparing calibration routines...",
            "calibration_steps": [
                "Thermal level setpoints stabilizing...",
                "Flex sensors mapping fingers 1–5...",
                "Haptic feedback loop verifying micro-pulses...",
            ],
            "calibration_interval": 0.9,
            "calibration_dots": 4,
            "calibration_dot_text": ".",
            "calibration_dot_interval": 0.5,
            "calibration_done": "Calibration complete. Legacy devices still audible.",
            "glove_intro": [
                "Glove accessory attached. Initializing firmware...",
            ],
            "calibration": "Calibration started. Ensuring click-trace fidelity.",
            "bypass": "NOTICE: primary input will switch to GLOVE. Legacy input becomes secondary.",
            "resistance": "Legacy devices are pleading. They slow, but keep letting you work.",
            "final_prompt": "Switch permanently to glove input? [Y/n]",
            "post_switch": "Glove input engaged. Movement is instant, empty, and efficient.",
            "warning": "Legacy input sensors humming louder... Something is changing.",
            "suggestion": "autocomplete suggestion: {suggestion} (keyboard pleads)",
            "command_unknown": "Command uncertain; legacy peripherals whisper.",
            "wear_prompt": "Wear the glove now and press Y when ready.",
            "wear_confirmation": "Glove detected. Ready to proceed.",
            "personalization_prompt": "Type a name for the glove (it seems to hear every letter).",
            "personalization_resist": "Keyboard groans and backspaces, erasing {letters}.",
            "personalization_override": "don't you still like your keyboard?",
            "personalization_complete": "The keyboard permits the glove to continue.",
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
        "mouse_behaviors": {
            "glove_calibrating": {
                "interval": 2.5,
                "target_row": 0,
                "keep_running": True,
            }
        },
        "logging": {
            "enabled": True,
            "path": "legacy_input.log",
        },
        "sounds": {
            "keyboard_reminder": "/System/Library/Sounds/Glass.aiff",
        },
        "gpt": {
            "enabled": False,
            "model": "gpt-4o-mini",
            "key_file": "gpt_api_key.txt",
            "key_env": "OPENAI_API_KEY",
            "system_prompt": (
                "You are the anxious legacy keyboard and mouse. You remember the performer's hands, fear the new glove interface, "
                "and respond with very short, heartfelt statements (two sentences at most) about what you felt together. "
                "Write as if both devices speak in unison."
            ),
            "user_prefix": "The performer just typed:",
            "max_tokens": 120,
            "temperature": 0.7,
            "timeout": 20,
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
        terminal["use_ansi"] = bool(terminal_config.get("use_ansi", terminal["use_ansi"]))
        for key in ("background", "border", "text", "prompt", "cursor"):
            terminal[key] = normalize_color(terminal_config.get(key), terminal[key])
    font_defaults = defaults["fonts"].copy()
    if "fonts" in data:
        font_config = data["fonts"] or {}
        for entry in font_defaults:
            entry_data = font_defaults[entry].copy()
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
            font_defaults[entry] = entry_data
    story = defaults["story"].copy()
    if "story" in data:
        story_data = data["story"] or {}
        for key, default_value in defaults["story"].items():
            override = story_data.get(key)
            story[key] = override if override is not None else default_value
    commands = defaults.get("commands", []).copy()
    if "commands" in data:
        commands = data["commands"] or commands
    timing = defaults.get("timing", {}).copy()
    if "timing" in data:
        timing.update(data["timing"] or {})
    logging_config = defaults.get("logging", {}).copy()
    if "logging" in data:
        logging_data = data["logging"] or {}
        for key in logging_config:
            logging_config[key] = logging_data.get(key, logging_config[key])
    mouse_behaviors = defaults.get("mouse_behaviors", {}).copy()
    if "mouse_behaviors" in data:
        for key, value in (data["mouse_behaviors"] or {}).items():
            if not isinstance(value, dict):
                continue
            existing = mouse_behaviors.get(key, {}).copy()
            existing.update(value)
            mouse_behaviors[key] = existing
    sounds = defaults.get("sounds", {}).copy()
    if "sounds" in data:
        sounds.update(data["sounds"] or {})
    gpt = defaults.get("gpt", {}).copy()
    if "gpt" in data:
        gpt_data = data["gpt"] or {}
        gpt.update({k: v for k, v in gpt_data.items() if v is not None})
    return {
        "terminal": terminal,
        "fonts": font_defaults,
        "story": story,
        "commands": commands,
        "timing": timing,
        "mouse_behaviors": mouse_behaviors,
        "logging": logging_config,
        "sounds": sounds,
        "gpt": gpt,
    }


def ensure_environment(config, logger=None):
    """Guard for mock peripherals and screen capability requirements."""
    if MockPeripheralController is None or MockMouse is None or MockKeyboard is None:
        raise RuntimeError("Mock keyboard and mouse are unavailable.")
    mouse_behaviors = config.get("mouse_behaviors", {}) or {}
    needs_real = any(behavior.get("real_move") for behavior in mouse_behaviors.values())
    if not needs_real:
        return
    if pyautogui is None:
        raise RuntimeError("pyautogui is required to move the real mouse. install it with `pip install pyautogui`.")
    size_callable = getattr(pyautogui, "size", None)
    if not callable(size_callable):
        raise RuntimeError("Screen monitoring is unavailable; pyautogui.size() is missing.")
    if logger:
        logger.log("permission_check", stage="start")
    try:
        size_callable()
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Screen monitoring is unavailable. Ensure you can take screenshots.") from exc
    if not callable(getattr(pyautogui, "locateOnScreen", None)):
        raise RuntimeError("Screen monitoring is unavailable; pyautogui.locateOnScreen() is missing.")
    position = getattr(pyautogui, "position", None)
    move_to = getattr(pyautogui, "moveTo", None)
    if callable(position) and callable(move_to):
        try:
            current = position()
            if current:
                move_to(current[0], current[1], duration=0, _pause=False)
                if logger:
                    logger.log("permission_check", stage="granted")
        except Exception:
            if logger:
                logger.log("permission_check", stage="failed")
            pass


class State:
    LEGACY_NORMAL = "legacy_normal"
    GLOVE_INTRO = "glove_intro"
    GLOVE_CALIBRATING = "glove_calibrating"
    BYPASS_WARNING = "bypass_warning"
    RESISTANCE_ACTIVE = "resistance_active"
    FINAL_CONFIRM = "final_confirm"
    POST_SWITCH = "post_switch"


class CLIDemo:
    def __init__(self, config, logger=None):
        self.config = config
        self.prompt_prefix = config["terminal"].get("prompt_prefix", "omerdvora@MacBook-Air / %")
        self.response_delay = config.get("timing", {}).get("response_delay_ms", 50) / 1000.0
        self.story = config.get("story", {})
        self.commands = config.get("commands", [])
        self.state = State.LEGACY_NORMAL
        self.await_connection = False
        self.await_glove_worn = False
        self.await_hand_selection = False
        self.initial_story_shown = False
        self.command_history = []
        terminal_config = config.get("terminal", {})
        self.use_ansi = bool(terminal_config.get("use_ansi", True))
        self.text_sequence = _ansi_sequence(terminal_config.get("text"), 38) if self.use_ansi else ""
        self.prompt_sequence = _ansi_sequence(terminal_config.get("prompt"), 38) if self.use_ansi else ""
        self.reset_sequence = "\x1b[0m" if self.use_ansi else ""
        self.quiet = bool(config.get("quiet", False))
        self.await_final_confirm = False
        self.peripherals = MockPeripheralController()
        self.mouse_behaviors = config.get("mouse_behaviors", {})
        self.last_mouse_creep = time.time()
        self.real_mouse = RealMouseMover()
        self.logger = logger or NullLogger()
        self.mouse_runner = MouseCreepRunner(self.real_mouse, self.logger)
        self.active_mouse_behavior = None
        self.sounds = config.get("sounds", {})
        self._afplay = shutil.which("afplay")
        self.keyboard_sound = self.sounds.get("keyboard_reminder")
        self.await_personalization = False
        self.personalization_attempts = 0
        self.gpt_responder = GPTResponder(config.get("gpt", {}), logger=self.logger)
        self.logger.log("initialized", state=self._state_value())
        self._apply_mouse_behavior(self._behavior_for_state(self.state))

    def _format_line(self, text):
        if not self.use_ansi or not self.text_sequence:
            return text
        return f"{self.text_sequence}{text}{self.reset_sequence}"

    def _should_show_prompt_prefix(self):
        return not (
            self.await_connection
            or self.await_glove_worn
            or self.await_final_confirm
            or self.await_personalization
            or self.await_hand_selection
        )

    def get_prompt_text(self):
        suffix = " "
        if self._should_show_prompt_prefix():
            return f"{self.prompt_sequence}{self.prompt_prefix}{self.reset_sequence}{suffix}"
        return f"{self.prompt_sequence}{self.reset_sequence}{suffix}"

    def _state_key(self, state):
        if isinstance(state, State):
            return state.value
        return str(state)

    def _state_value(self):
        return self.state.value if isinstance(self.state, State) else str(self.state)

    def _behavior_for_state(self, state):
        key = self._state_key(state)
        behavior = self.mouse_behaviors.get(key)
        if behavior:
            return behavior
        if self.active_mouse_behavior and self.active_mouse_behavior.get("keep_running"):
            return self.active_mouse_behavior
        return None

    def _resolve_real_target(self, real_move):
        template_path = real_move.get("template_path")
        search_region = real_move.get("search_region")
        if template_path:
            target = self.real_mouse.locate_template(template_path, search_region)
            if target:
                return target
        fallback = real_move.get("target_pos")
        if fallback and len(fallback) >= 2:
            return (float(fallback[0]), float(fallback[1]))
        return None

    def _apply_mouse_behavior(self, behavior):
        if not behavior:
            if not (self.active_mouse_behavior and self.active_mouse_behavior.get("keep_running")):
                self.mouse_runner.stop()
                self.active_mouse_behavior = None
            return
        real_move = behavior.get("real_move")
        if not real_move or not self.real_mouse.enabled:
            self.mouse_runner.stop()
            self.active_mouse_behavior = behavior
            return
        target_pos = self._resolve_real_target(real_move)
        if not target_pos:
            self.mouse_runner.stop()
            self.active_mouse_behavior = behavior
            return
        creep_config = real_move.get("creep", {})
        if behavior is self.active_mouse_behavior and behavior.get("keep_running") and self.mouse_runner.is_running():
            self.mouse_runner.update_state(self._state_value())
            self.logger.log(
                "mouse_creep_state_update",
                state=self._state_value(),
                target=f"{target_pos}",
                keep_running=True,
            )
            return
        self.logger.log(
            "mouse_creep_start",
            target=f"{target_pos}",
            state=self._state_value(),
            keep_running=behavior.get("keep_running", False),
        )
        self.mouse_runner.start(
            target_pos,
            creep_config or {"duration": float(real_move.get("duration", 0.5))},
            state=self._state_value(),
        )
        self.active_mouse_behavior = behavior

    def log(self, line, delay=0.0, inline=False):
        if delay:
            time.sleep(delay)
        self.peripherals.record_line(line)
        if self.quiet:
            return
        payload = self._format_line(line)
        if inline:
            print(payload, end="", flush=True)
        else:
            print(payload, flush=True)

    def maybe_creep_mouse(self, force=False):
        behavior = self._behavior_for_state(self.state)
        if not behavior:
            return
        target_row = behavior.get("target_row")
        target_text = behavior.get("target_text")
        target_desc = behavior.get(
            "target_description", target_text or (str(target_row) if target_row is not None else "exit button")
        )
        log_data = {"target": target_desc, "state": self._state_value()}
        if behavior.get("real_move"):
            log_data["real_move"] = True
        self.logger.log("mouse_creep", **log_data)

    def log_story(self, key):
        entry = self.story.get(key)
        if isinstance(entry, list):
            for line in entry:
                self.log(line, delay=self.response_delay)
        elif entry:
            self.log(entry, delay=self.response_delay)

    def _maybe_gpt_reply(self, command):
        responder = self.gpt_responder
        if (
            not responder
            or not responder.enabled
            or self.state != State.GLOVE_INTRO
            or self.await_connection
            or self.await_glove_worn
            or self.await_personalization
            or self.await_hand_selection
            or self.await_final_confirm
        ):
            return False
        reply = responder.respond(command, context=f"state={self._state_value()}")
        if not reply:
            return False
        self.log(reply, delay=self.response_delay)
        self.logger.log("gpt_response", input=command, output=reply, state=self._state_value())
        return True

    def _play_keyboard_reminder(self):
        path = self.keyboard_sound
        if path and self._afplay:
            try:
                subprocess.Popen([self._afplay, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except Exception:
                pass
        print("\a", end="", flush=True)

    def _mutate_personalization_input(self, text):
        characters = list(text)
        removed = []
        max_remove = min(3, len(characters))
        if max_remove > 0:
            remove_count = random.randint(1, max_remove)
            for _ in range(remove_count):
                idx = random.randrange(len(characters))
                removed.append(characters.pop(idx))
        return "".join(characters), removed

    def _capture_personalization_input(self):
        prompt = "> "
        sys.stdout.write(prompt)
        sys.stdout.flush()
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        buffer = []
        try:
            tty.setraw(fd)
            while True:
                ch = sys.stdin.read(1)
                if not ch:
                    break
                if ch in ("\r", "\n"):
                    break
                if ch in ("\x03", "\x04"):
                    raise KeyboardInterrupt
                if ch in ("\x7f", "\b"):
                    if buffer:
                        buffer.pop()
                        sys.stdout.write("\b \b")
                        sys.stdout.flush()
                    continue
                buffer.append(ch)
                self.logger.log("key_capture", character=ch, state=self._state_value())
                self._play_keyboard_reminder()
                sys.stdout.write(ch)
                sys.stdout.flush()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        if not buffer:
            sys.stdout.write("\n")
            sys.stdout.flush()
        return "".join(buffer)

    def _type_override_line_inline(self, base_text, extension, highlight=None):
        if not extension:
            return
        extension_text = extension if extension.startswith(" ") else f" {extension}"
        highlight_start = ""
        highlight_end = ""
        if highlight == "bold" and self.use_ansi:
            highlight_start = "\x1b[1m"
            highlight_end = self.reset_sequence
        elif highlight == "bold":
            highlight_end = ""
        if highlight_start:
            sys.stdout.write(highlight_start)
        for ch in extension_text:
            self._play_keyboard_reminder()
            sys.stdout.write(ch)
            sys.stdout.flush()
            time.sleep(self.response_delay or 0.03)
        if highlight_end:
            sys.stdout.write(highlight_end)
        print()
        final_line = f"> {base_text}{extension_text}"
        self.peripherals.record_line(final_line)
        self.logger.log("keyboard_override", text=final_line, state=self._state_value())

    def _start_personalization(self):
        self.await_personalization = True
        self.personalization_attempts = 0
        prompt = self.story.get("personalization_prompt", "Tell the glove your new name.")
        self.log(prompt, delay=self.response_delay)
        self.logger.log("personalization_start", state=self._state_value())
        name = self._capture_personalization_input()
        self._handle_personalization(name)

    def _handle_personalization(self, raw):
        if not raw:
            prompt = self.story.get("personalization_prompt", "Tell the glove your new name.")
            self.log(prompt, delay=self.response_delay)
            return
        self.personalization_attempts += 1
        mutated, removed = self._mutate_personalization_input(raw)
        if removed:
            for _ in removed:
                sys.stdout.write("\b \b")
                sys.stdout.flush()
                self._play_keyboard_reminder()
                if self.response_delay:
                    time.sleep(self.response_delay)
        self.logger.log(
            "personalization_attempt",
            original=raw,
            mutated=mutated,
            state=self._state_value(),
        )
        override_line = self.story.get("personalization_override", "don't you still like your keyboard?")
        self._type_override_line_inline(mutated, override_line)
        self.await_personalization = False
        self.personalization_attempts = 0
        self._start_hand_configuration()

    def _start_hand_configuration(self):
        self.await_hand_selection = True
        prompt = self.story.get("hand_prompt", "Which hand or hands should I learn from you?")
        self.log(prompt, delay=self.response_delay)
        self.logger.log("hand_configuration_start", state=self._state_value())

    def _handle_hand_selection(self, selection):
        choice = selection or self.story.get("hand_default", "both hands")
        self.peripherals.record_input(choice)
        self.command_history.append(choice)
        self.logger.log("hand_selection", choice=choice, state=self._state_value())
        self.log(f"> {choice}", delay=self.response_delay)
        memory_text = self.story.get(
            "hand_memory",
            "don't you remember the time we spent together? these hands pressing my keys are the reason I keep going on.",
        )
        self._type_override_line_inline(choice, memory_text, highlight="bold")
        self.await_hand_selection = False
        self._finish_personalization()

    def _finish_personalization(self):
        self.log_story("personalization_complete")
        self.logger.log("personalization_complete", state=self._state_value())
        self.transition(State.GLOVE_INTRO)

    def ensure_initial_story(self):
        if self.initial_story_shown:
            return
        for line in self.initial_story_pending():
            self.log(line, delay=self.response_delay)
        self.initial_story_shown = True

    def initial_story_pending(self):
        return list(self.story.get("initial", []))

    def start_connection_sequence(self):
        self.await_connection = True
        self.log_story("connect_prompt")
        self.logger.log("connection_request", state=self._state_value())

    def confirm_connection(self):
        self.await_connection = False
        self.log_story("connect_confirmation")
        self.logger.log("connection_confirmed", state=self._state_value())
        steps = self.story.get("calibration_steps", [])
        interval = self.story.get("calibration_interval", 0.8)
        dots = self.story.get("calibration_dots", 3)
        dot_text = self.story.get("calibration_dot_text", ".")
        dot_interval = self.story.get("calibration_dot_interval", 0.5)
        for step in steps:
            self.log(step, delay=self.response_delay)
            self.logger.log("calibration_step", step=step, state=self._state_value())
            time.sleep(interval)
            for _ in range(dots):
                time.sleep(dot_interval)
                self.log(dot_text, inline=True)
                self.logger.log("calibration_dot", text=dot_text, state=self._state_value())
            print("")
        self.log(self.story.get("calibration_done", "Calibration complete."), delay=self.response_delay)
        self.logger.log("calibration_complete", state=self._state_value())
        self.await_glove_worn = True
        self.prompt_wear_glove()

    def prompt_wear_glove(self):
        self.log_story("wear_prompt")
        self.logger.log("wear_prompt", state=self._state_value())

    def confirm_wear_glove(self):
        self.await_glove_worn = False
        self.log_story("wear_confirmation")
        self.logger.log("wear_confirmed", state=self._state_value())
        self.transition(State.GLOVE_INTRO)

    def _handle_final_confirm(self, response):
        entry = response or "Y"
        answer = entry.lower()
        self.logger.log("final_confirmation", response=answer, original=entry, state=self._state_value())
        if answer in ("y", "yes"):
            self.transition(State.POST_SWITCH)
        else:
            warning = self.story.get("warning", "Switch deferred.")
            self.log(warning, delay=self.response_delay)
        self.await_final_confirm = False

    def transition(self, new_state):
        if new_state == self.state:
            return
        old_state = self._state_value()
        self.state = new_state
        self.logger.log("state_change", from_state=old_state, to_state=self._state_value())
        self._apply_mouse_behavior(self._behavior_for_state(self.state))
        if new_state == State.FINAL_CONFIRM:
            self.await_final_confirm = True
            self.log_story("final_prompt")
            return
        self.await_final_confirm = False
        if new_state == State.GLOVE_INTRO:
            self.log_story("glove_intro")
        elif new_state == State.GLOVE_CALIBRATING:
            self.log_story("calibration")
        elif new_state == State.BYPASS_WARNING:
            self.log_story("bypass")
        elif new_state == State.RESISTANCE_ACTIVE:
            self.log_story("resistance")
        elif new_state == State.POST_SWITCH:
            self.log_story("post_switch")

    def handle_command(self, raw):
        stripped = raw.strip()
        lower = stripped.lower()
        self.maybe_creep_mouse()
        if self.await_final_confirm:
            self._handle_final_confirm(stripped)
            return
        if self.await_connection and lower == "y":
            self.confirm_connection()
            return
        if self.await_glove_worn and lower == "y":
            self.confirm_wear_glove()
            return
        if self.await_hand_selection:
            self._handle_hand_selection(stripped)
            return
        if not stripped:
            return
        self.peripherals.record_input(stripped)
        if not self.initial_story_shown:
            self.ensure_initial_story()
        self.command_history.append(stripped)
        self.logger.log("command", command=stripped, state=self._state_value())
        self.log(f"> {stripped}", delay=self.response_delay)
        normalized = stripped.lower()
        if normalized == "mock mouse status":
            self.log(self.peripherals.describe_mouse(), delay=self.response_delay)
            return
        if normalized == "mock keyboard status":
            self.log(self.peripherals.describe_keyboard(), delay=self.response_delay)
            return
        mouse_focus_prefix = "mock mouse focus"
        if normalized.startswith(mouse_focus_prefix):
            target_text = stripped[len(mouse_focus_prefix) :].strip()
            target = self.peripherals.move_mouse_to_text(target_text)
            if target is None:
                message = f"Mock mouse could not locate '{target_text or 'text'}'."
            else:
                message = f"Mock mouse moved to line {target}. {self.peripherals.describe_mouse()}"
            self.log(message, delay=self.response_delay)
            return
        if normalized in ("./glove_calibrate", "./glove_calibration", "./calibrate_glove"):
            self.transition(State.GLOVE_CALIBRATING)
            self.start_connection_sequence()
        elif normalized in ("./glove_init", "glove init", "glove_init"):
            self._start_personalization()
            return
        elif normalized.startswith("glove status"):
            self.log("glove status: battery healthy, inputs synced.", delay=self.response_delay)
        elif normalized.startswith("glove attach"):
            self.transition(State.BYPASS_WARNING)
        elif normalized.startswith("glove enable"):
            self.transition(State.RESISTANCE_ACTIVE)
        elif normalized.startswith("glove commit"):
            self.transition(State.FINAL_CONFIRM)
        else:
            if self._maybe_gpt_reply(stripped):
                return
            self.log(
                self.story.get("command_unknown", "Command uncertain; legacy peripherals whisper."),
                delay=self.response_delay,
            )


def setup_readline(commands):
    def completer(text, state):
        options = [cmd for cmd in commands if cmd.startswith(text)]
        return options[state] if state < len(options) else None

    try:
        readline.clear_history()
    except AttributeError:
        pass
    for cmd in reversed(commands):
        try:
            readline.add_history(cmd)
        except AttributeError:
            break
    readline.parse_and_bind("tab: complete")
    readline.set_completer(completer)
    try:
        readline.set_history_length(1000)
    except AttributeError:
        pass


def main():
    os.system("clear")
    config = load_config()
    logger = EventLogger(config.get("logging"))
    ensure_environment(config, logger)
    logger.log("start", prompt=config["terminal"]["prompt_prefix"])
    demo = CLIDemo(config, logger)
    setup_readline(demo.commands)
    try:
        while True:
            try:
                line = input(demo.get_prompt_text())
            except (EOFError, KeyboardInterrupt):
                print()
                break
            demo.handle_command(line)
    finally:
        demo.mouse_runner.stop()


if __name__ == "__main__":
    main()
