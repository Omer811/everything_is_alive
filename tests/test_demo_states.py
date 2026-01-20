import copy
import os
import tempfile
import unittest
import io
from unittest.mock import patch

import cli_demo

from cli_demo import CLIDemo, State, load_config


class LoggingRecorder(cli_demo.NullLogger):
    def __init__(self):
        self.events = []

    def log(self, event, **details):
        self.events.append((event, details))


class CLIDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    def setUp(self):
        config = copy.deepcopy(load_config())
        config["timing"]["response_delay_ms"] = 0
        config["quiet"] = True
        config.setdefault("logging", {})["enabled"] = False
        self.demo = CLIDemo(config)

    def test_initial_story_hidden_until_command(self):
        self.assertFalse(self.demo.initial_story_shown)
        self.demo.handle_command("   ")
        self.assertFalse(self.demo.initial_story_shown)
        def capture(self):
            value = "alias"
            for ch in value:
                self.logger.log("key_capture", character=ch, state=self._state_value())
            return value
        with patch.object(CLIDemo, "_capture_personalization_input", capture):
            self.demo.handle_command("./glove_init")
        self.assertTrue(self.demo.initial_story_shown)

    def test_glove_calibration_orders_flags(self):
        self.demo.handle_command("./glove_calibrate")
        self.assertTrue(self.demo.await_connection)
        self.demo.handle_command("Y")
        self.assertTrue(self.demo.await_glove_worn)
        self.assertFalse(self.demo.await_connection)

    def test_final_prompt_accepts_default(self):
        self.demo.handle_command("glove commit")
        self.assertEqual(self.demo.state, State.FINAL_CONFIRM)
        self.assertTrue(self.demo.await_final_confirm)
        self.demo.handle_command("")
        self.assertEqual(self.demo.state, State.POST_SWITCH)
        self.assertFalse(self.demo.await_final_confirm)

    def test_glove_init_aliases(self):
        config = copy.deepcopy(load_config())
        config["timing"]["response_delay_ms"] = 0
        config["quiet"] = True
        logger = LoggingRecorder()
        def capture(self):
            value = "alias"
            for ch in value:
                logger.log("key_capture", character=ch, state=self._state_value())
            return value
        with patch.object(CLIDemo, "_capture_personalization_input", capture):
            demo = CLIDemo(config, logger=logger)
            demo.handle_command("glove init")
        events = [event for event, _ in logger.events]
        self.assertIn("personalization_start", events)

    def test_mock_mouse_focus_command(self):
        self.demo.log("calibration started", delay=0.0)
        self.demo.handle_command("mock mouse focus calibration")
        self.assertEqual(self.demo.peripherals.mouse.row, 0)
        self.assertEqual(self.demo.peripherals.mouse.state, "focused")

    def test_mock_keyboard_status_command(self):
        self.demo.handle_command("mock keyboard status")
        self.assertIn("Mock keyboard", self.demo.peripherals.line_history[-1])

    def test_mouse_creep_behavior(self):
        self.demo.state = State.GLOVE_CALIBRATING
        self.demo.mouse_behaviors = {
            State.GLOVE_CALIBRATING: {
                "interval": 0,
                "target_row": 0,
            }
        }
        self.demo.peripherals.record_line("line a")
        self.demo.peripherals.record_line("line b")
        self.demo.last_mouse_creep = 0
        self.demo.maybe_creep_mouse(force=True)
        self.assertEqual(self.demo.peripherals.mouse.row, 0)
        self.assertEqual(len(self.demo.peripherals.line_history), 2)

    def test_real_mouse_move_called(self):
        config = copy.deepcopy(load_config())
        config["timing"]["response_delay_ms"] = 0
        config["quiet"] = True
        config["mouse_behaviors"] = {
            State.GLOVE_CALIBRATING: {
                "interval": 0,
                "real_move": {
                    "target_pos": [200, 100],
                    "duration": 0.1,
                    "creep": {"step_interval": 0.01, "min_step": 10, "max_step": 10, "bias": 1.0},
                },
            }
        }
        calls = []

        class DummyMover:
            def __init__(self):
                self.enabled = True

            def locate_template(self, *args, **kwargs):
                return None

            def move_to(self, target_pos, duration=0.5):
                calls.append((tuple(target_pos), duration))
                return True

        class DummyRunner:
            def __init__(self, mover, logger):
                self.started = []

            def start(self, target, config, state=None):
                calls.append((tuple(target), tuple(sorted(config.items())), state))

            def stop(self):
                pass

        original_runner = cli_demo.MouseCreepRunner
        original_mover = cli_demo.RealMouseMover
        try:
            cli_demo.RealMouseMover = lambda *args, **kwargs: DummyMover()
            cli_demo.MouseCreepRunner = DummyRunner
            demo = CLIDemo(config)
            demo.transition(State.GLOVE_CALIBRATING)
            self.assertTrue(calls)
        finally:
            cli_demo.MouseCreepRunner = original_runner
            cli_demo.RealMouseMover = original_mover

    def test_mouse_runner_keep_running_across_states(self):
        config = copy.deepcopy(load_config())
        config["timing"]["response_delay_ms"] = 0
        config["quiet"] = True
        config["mouse_behaviors"] = {
            State.GLOVE_CALIBRATING: {
                "real_move": {
                    "target_pos": [15, 15],
                    "duration": 0.1,
                    "creep": {"step_interval": 0.01, "min_step": 0.1, "max_step": 0.1, "bias": 0.9},
                },
                "keep_running": True,
            }
        }

        class DummyMover:
            def __init__(self):
                self.enabled = True

        class DummyRunner:
            def __init__(self, mover, logger):
                self.started = []
                self.updated = []
                self.running = False

            def start(self, target, config, state=None):
                self.started.append((tuple(target), state))
                self.running = True

            def stop(self):
                self.running = False

            def is_running(self):
                return self.running

            def update_state(self, state):
                self.updated.append(state)

        original_runner = cli_demo.MouseCreepRunner
        original_mover = cli_demo.RealMouseMover
        try:
            cli_demo.MouseCreepRunner = DummyRunner
            cli_demo.RealMouseMover = lambda *args, **kwargs: DummyMover()
            demo = CLIDemo(config)
            demo.transition(State.GLOVE_CALIBRATING)
            self.assertEqual(len(demo.mouse_runner.started), 1)
            demo.transition(State.GLOVE_INTRO)
            self.assertEqual(len(demo.mouse_runner.started), 1, "runner should not restart when keep_running is true")
            self.assertIn(State.GLOVE_INTRO, demo.mouse_runner.updated)
        finally:
            cli_demo.MouseCreepRunner = original_runner
            cli_demo.RealMouseMover = original_mover

    def test_personalization_resistance_and_override(self):
        config = copy.deepcopy(load_config())
        config["timing"]["response_delay_ms"] = 0
        config["quiet"] = True
        config["sounds"] = {"keyboard_reminder": None}
        logger = LoggingRecorder()
        demo = CLIDemo(config, logger=logger)
        def capture(self):
            value = "persistent"
            for ch in value:
                logger.log("key_capture", character=ch, state=self._state_value())
            return value
        with patch.object(CLIDemo, "_capture_personalization_input", capture):
            demo.handle_command("./glove_init")
        self.assertFalse(demo.await_personalization)
        demo.handle_command("persistent")
        self.assertFalse(demo.await_personalization)
        self.assertEqual(demo.state, State.GLOVE_INTRO)
        matches = [
            line for line in demo.peripherals.line_history if line.endswith(" don't you still like your keyboard?")
        ]
        self.assertTrue(matches, f"Expected override line in history, got {demo.peripherals.line_history}")
        keypresses = [event for event, details in logger.events if event == "key_capture"]
        self.assertGreaterEqual(len(keypresses), len("persistent"))

    def test_hand_configuration_inserts_memory(self):
        config = copy.deepcopy(load_config())
        config["timing"]["response_delay_ms"] = 0
        config["quiet"] = True
        logger = LoggingRecorder()
        demo = CLIDemo(config, logger=logger)
        def capture(self):
            return "alias"
        with patch.object(CLIDemo, "_capture_personalization_input", capture):
            demo.handle_command("./glove_init")
        self.assertTrue(demo.await_hand_selection)
        demo.handle_command("left hand")
        self.assertFalse(demo.await_hand_selection)
        self.assertEqual(demo.state, State.GLOVE_INTRO)
        events = [event for event, _ in logger.events]
        self.assertIn("hand_selection", events)

    def test_personalization_keypress_logging(self):
        config = copy.deepcopy(load_config())
        config["timing"]["response_delay_ms"] = 0
        config["quiet"] = True
        config["sounds"] = {"keyboard_reminder": None}
        logger = LoggingRecorder()
        demo = CLIDemo(config, logger=logger)
        sounds = []
        demo._play_keyboard_reminder = lambda: sounds.append("beep")
        def capture(self):
            value = "xyz"
            for ch in value:
                logger.log("key_capture", character=ch, state=self._state_value())
                self._play_keyboard_reminder()
            return value
        with patch.object(CLIDemo, "_capture_personalization_input", capture):
            demo._capture_personalization_input()
        self.assertEqual(len(sounds), 3)
        keypresses = [details["character"] for event, details in logger.events if event == "key_capture"]
        self.assertEqual("".join(keypresses), "xyz")

    def test_gpt_reply_after_init(self):
        config = copy.deepcopy(load_config())
        config["timing"]["response_delay_ms"] = 0
        config["quiet"] = True
        config["gpt"]["enabled"] = True
        logger = LoggingRecorder()
        class DummyResponder:
            def __init__(self, *args, **kwargs):
                self.enabled = True

            def verify_connection(self):
                return True

            def respond(self, text, context=None):
                return "keyboard and mouse remember every tremor."

        original = cli_demo.GPTResponder
        try:
            cli_demo.GPTResponder = DummyResponder
            demo = CLIDemo(config, logger=logger)
            demo.state = State.GLOVE_INTRO
            demo.handle_command("feelings")
            self.assertIn("keyboard and mouse remember", demo.peripherals.line_history[-1])
            events = [event for event, _ in logger.events]
            self.assertIn("gpt_response", events)
        finally:
            cli_demo.GPTResponder = original

    def test_init_fails_when_gpt_unavailable(self):
        config = copy.deepcopy(load_config())
        config["gpt"]["enabled"] = True
        class DummyResponder:
            def __init__(self, *args, **kwargs):
                self.enabled = True

            def verify_connection(self):
                return False

        original = cli_demo.GPTResponder
        try:
            cli_demo.GPTResponder = DummyResponder
            with self.assertRaises(RuntimeError):
                CLIDemo(config)
        finally:
            cli_demo.GPTResponder = original

    def test_stop_kills_audio_processes(self):
        config = copy.deepcopy(load_config())
        config["timing"]["response_delay_ms"] = 0
        config["quiet"] = True
        config["sounds"] = {"keyboard_reminder": "/path/to/sound"}
        demo = CLIDemo(config)
        class DummyProc:
            def __init__(self):
                self.killed = False
                self.wait_called = False

            def poll(self):
                return None

            def kill(self):
                self.killed = True

            def wait(self, timeout=None):
                self.wait_called = True

        proc = DummyProc()
        proc.wait_called = False

        proc = DummyProc()
        def fake_popen(*args, **kwargs):
            return proc

        with patch("cli_demo.subprocess.Popen", fake_popen):
            demo._play_keyboard_reminder()
        self.assertTrue(demo._afplay_procs)
        demo.stop()
        self.assertFalse(demo._afplay_procs)
        self.assertTrue(proc.killed)
        self.assertTrue(proc.wait_called)

    def test_history_navigation_up_down(self):
        config = copy.deepcopy(load_config())
        config["quiet"] = True
        demo = CLIDemo(config)
        prompt = demo.get_prompt_text()
        demo.command_history = ["./glove_init", "./glove_calibrate"]
        buffer = []
        prev_len = len(prompt)
        history_nav = len(demo.command_history)
        fake_out = io.StringIO()
        with patch("cli_demo.sys.stdout", fake_out):
            prev_len, history_nav = demo._handle_escape_sequence("[A", prompt, buffer, prev_len, history_nav)
        self.assertEqual("".join(buffer), "./glove_calibrate")
        self.assertEqual(history_nav, len(demo.command_history) - 1)
        with patch("cli_demo.sys.stdout", fake_out):
            prev_len2, history_nav = demo._handle_escape_sequence("[B", prompt, buffer, prev_len, history_nav)
        self.assertEqual(buffer, [])
        self.assertEqual(history_nav, len(demo.command_history))

    def test_history_navigation_with_extended_sequence(self):
        config = copy.deepcopy(load_config())
        config["quiet"] = True
        demo = CLIDemo(config)
        prompt = demo.get_prompt_text()
        demo.command_history = ["./glove_init", "./glove_calibrate"]
        buffer = []
        prev_len = len(prompt)
        history_nav = len(demo.command_history)
        fake_out = io.StringIO()
        with patch("cli_demo.sys.stdout", fake_out):
            prev_len, history_nav = demo._handle_escape_sequence("[1;5A", prompt, buffer, prev_len, history_nav)
        self.assertEqual("".join(buffer), "./glove_calibrate")
        self.assertEqual(history_nav, len(demo.command_history) - 1)

    def test_sound_input_only_during_personalization(self):
        config = copy.deepcopy(load_config())
        config["quiet"] = True
        demo = CLIDemo(config)
        self.assertFalse(demo.should_use_sound_input())
        demo.await_personalization = True
        self.assertTrue(demo.should_use_sound_input())
        demo.await_personalization = False
        demo.await_hand_selection = True
        self.assertTrue(demo.should_use_sound_input())
        demo.await_hand_selection = False
        self.assertFalse(demo.should_use_sound_input())

    def test_default_mouse_behavior_keeps_running(self):
        config = load_config()
        behavior = config["mouse_behaviors"].get("glove_calibrating", {})
        self.assertTrue(
            behavior.get("keep_running"),
            "Expected the default glove_calibrating behavior to keep creeping after state changes.",
        )

    def test_environment_requires_pyautogui(self):
        config = copy.deepcopy(load_config())
        config["mouse_behaviors"] = {
            State.GLOVE_CALIBRATING: {"real_move": {"target_pos": [0, 0], "duration": 0.1}}
        }
        original = cli_demo.pyautogui
        try:
            cli_demo.pyautogui = None
            with self.assertRaises(RuntimeError) as ctx:
                cli_demo.ensure_environment(config, cli_demo.NullLogger())
            self.assertIn("pyautogui", str(ctx.exception))
        finally:
            cli_demo.pyautogui = original

    def test_environment_requires_screen_monitoring(self):
        config = copy.deepcopy(load_config())
        config["mouse_behaviors"] = {
            State.GLOVE_CALIBRATING: {"real_move": {"target_pos": [0, 0], "duration": 0.1}}
        }
        class DummyScreen:
            def size(self):
                raise OSError("no display")
        original = cli_demo.pyautogui
        try:
            cli_demo.pyautogui = DummyScreen()
            with self.assertRaises(RuntimeError) as ctx:
                cli_demo.ensure_environment(config, cli_demo.NullLogger())
            self.assertIn("Screen monitoring", str(ctx.exception))
        finally:
            cli_demo.pyautogui = original

    def test_event_logger_writes_file(self):
        path = tempfile.NamedTemporaryFile(delete=False)
        try:
            path_name = path.name
        finally:
            path.close()
        try:
            logger = cli_demo.EventLogger({"path": path_name, "enabled": True})
            logger.log("command", detail="value")
            with open(path_name, "r", encoding="utf-8") as fh:
                contents = fh.read()
            self.assertIn("command", contents)
        finally:
            os.remove(path_name)
