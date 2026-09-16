"""Cascade WAM: World Model -> visual subgoal -> Action Expert -> action chunk.

The public API mirrors a minimal agent:

    wam = CascadeWAM(world_model, action_expert, ...)
    wam.reset()
    wam.set_instruction("Go to the hollow circle.")
    chunk = wam.step()   # runs WM (async) + AE, returns (H,4) discrete-ish chunk
    wam.observe(frame)   # push a new observation into the history buffer

π0.7-style asynchronous subgoal: the world model runs in a background thread and
refreshes the visual subgoal at a *low frequency* (``subgoal_update_every``
steps), while the action expert consumes the latest available (possibly stale)
subgoal every step — it never blocks waiting for a fresh one.  The first subgoal
is generated synchronously so the very first action can be taken immediately.
"""

from __future__ import annotations

import threading

import numpy as np
import torch

from .language_encoder import tokenize, PAD_ID


class CascadeWAM:
    def __init__(self, world_model, action_expert, *, history: int = 4,
                 execute_steps: int = 2, device: str = "cpu",
                 subtask_mode: str = "oracle", subtask_policy=None,
                 subgoal_update_every: int = 5, async_mode: bool = True):
        self.wm = world_model
        self.ae = action_expert
        self.history = history
        self.execute_steps = execute_steps
        self.device = device
        self.subtask_mode = subtask_mode
        self.subtask_policy = subtask_policy
        self.subgoal_update_every = subgoal_update_every
        self.async_mode = async_mode

        self._buffer = None
        self._instruction = ""
        self._subtask = ""
        self._lang_ids = None
        self._subtask_ids = None
        self._meta = None

        # subgoal state (shared between the WM thread and the action loop)
        self._subgoal = None
        self._subgoal_lock = threading.Lock()
        self._step_count = 0

        # world-model worker thread
        self._wm_thread = None
        self._wm_stop = threading.Event()
        self._wm_trigger = threading.Event()
        self._wm_context = None

    # -- setup ------------------------------------------------------------ #
    def reset(self):
        self._buffer = None
        self._instruction = ""
        self._subtask = ""
        self._subgoal = None
        self._step_count = 0

    def set_instruction(self, text: str):
        self._instruction = text
        self._subtask = self._oracle_subtask(text)
        self._lang_ids = torch.tensor([tokenize(text)], device=self.device)
        self._subtask_ids = torch.tensor([tokenize(self._subtask)], device=self.device)
        self._meta = [{"speed": "normal", "quality": 1.0, "mistake": False,
                       "control_mode": "keyboard"}]

    @staticmethod
    def _oracle_subtask(text: str) -> str:
        low = text.lower()
        if "circle" in low:
            obj = "circle"
        elif "triangle" in low:
            obj = "triangle"
        else:
            obj = "square"
        fill = "hollow" if "hollow" in low else ("solid" if "solid" in low else None)
        phrase = f"{fill} {obj}" if fill else obj
        return f"move toward the {phrase}"

    # -- observation ------------------------------------------------------ #
    def observe(self, frame: np.ndarray):
        """Push a single 48x48 frame into the history buffer.

        ``frame`` is expected in [0, 1] (as produced by ``env.render``); it is
        normalised to [-1, 1] to match the training distribution.
        """
        f = torch.as_tensor(frame, dtype=torch.float32, device=self.device)
        f = f.reshape(1, 1, 48, 48) * 2.0 - 1.0  # [0,1] -> [-1,1]
        if self._buffer is None:
            self._buffer = f.repeat(1, self.history, 1, 1)  # (1,K,48,48)
        else:
            self._buffer = torch.cat([self._buffer[:, 1:], f], dim=1)

    # -- world model (async, low-frequency) ------------------------------- #
    def _start_wm_thread(self):
        if self._wm_thread is None:
            self._wm_thread = threading.Thread(target=self._wm_loop, daemon=True)
            self._wm_thread.start()

    def _wm_loop(self):
        while not self._wm_stop.is_set():
            self._wm_trigger.wait()
            self._wm_trigger.clear()
            if self._wm_stop.is_set():
                break
            buffer, lang, sub, meta = self._wm_context
            with torch.no_grad():
                sg = self.wm.sample(buffer, lang, sub, meta).clamp(-1.0, 1.0)
            with self._subgoal_lock:
                self._subgoal = sg

    def _request_subgoal_async(self):
        self._start_wm_thread()
        self._wm_context = (self._buffer, self._lang_ids, self._subtask_ids, self._meta)
        self._wm_trigger.set()

    def _request_subgoal_sync(self):
        with torch.no_grad():
            sg = self.wm.sample(self._buffer, self._lang_ids, self._subtask_ids, self._meta)
        self._subgoal = sg.clamp(-1.0, 1.0)

    def _maybe_refresh_subgoal(self):
        """Low-frequency subgoal refresh; returns True if a refresh was requested."""
        self._step_count += 1
        if self._subgoal is None:
            self._request_subgoal_sync()   # need one now for the first action
            return True
        if self._step_count % self.subgoal_update_every == 0:
            if self.async_mode:
                self._request_subgoal_async()   # non-blocking; AE keeps stale subgoal
            else:
                self._request_subgoal_sync()
            return True
        return False

    # -- inference -------------------------------------------------------- #
    @torch.no_grad()
    def step(self) -> torch.Tensor:
        if self._buffer is None:
            raise RuntimeError("observe() before step()")
        self._maybe_refresh_subgoal()
        with self._subgoal_lock:
            subgoal = self._subgoal
        chunk = self.ae.sample(self._buffer, subgoal, self._lang_ids,
                               self._subtask_ids, self._meta)
        return chunk.squeeze(0)  # (H, 4) continuous

    def shutdown(self):
        self._wm_stop.set()
        self._wm_trigger.set()   # wake the worker so it can exit
        if self._wm_thread is not None:
            self._wm_thread.join(timeout=2)

    # -- helpers ---------------------------------------------------------- #
    @property
    def instruction(self) -> str:
        return self._instruction

    @property
    def subtask(self) -> str:
        return self._subtask

    @property
    def subgoal(self) -> torch.Tensor | None:
        with self._subgoal_lock:
            return self._subgoal
