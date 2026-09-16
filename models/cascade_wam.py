"""Cascade WAM: World Model -> visual subgoal -> Action Expert -> action chunk.

The public API mirrors a minimal agent:

    wam = CascadeWAM(world_model, action_expert, ...)
    wam.reset()
    wam.set_instruction("Go to the hollow circle.")
    chunk = wam.step()   # runs WM then AE, returns (H,4) discrete-ish chunk
    wam.observe(frame)   # push a new observation into the history buffer

Subtask is provided by an (oracle or learned) subtask policy; the first version
uses an oracle subtask derived from the instruction.  A stale-subgoal buffer is
used so the world model does not block every action prediction (asynchronous
inference), with synchronous mode as the default.
"""

from __future__ import annotations

import numpy as np
import torch

from .language_encoder import tokenize, PAD_ID


class CascadeWAM:
    def __init__(self, world_model, action_expert, *, history: int = 4,
                 execute_steps: int = 2, device: str = "cpu",
                 subtask_mode: str = "oracle", subtask_policy=None,
                 synchronous: bool = True):
        self.wm = world_model
        self.ae = action_expert
        self.history = history
        self.execute_steps = execute_steps
        self.device = device
        self.subtask_mode = subtask_mode
        self.subtask_policy = subtask_policy
        self.synchronous = synchronous

        self._buffer = None
        self._instruction = ""
        self._subtask = ""
        self._lang_ids = None
        self._subtask_ids = None
        self._meta = None
        self._subgoal = None

    # -- setup ------------------------------------------------------------ #
    def reset(self):
        self._buffer = None
        self._instruction = ""
        self._subtask = ""
        self._subgoal = None

    def set_instruction(self, text: str):
        self._instruction = text
        self._subtask = self._oracle_subtask(text)
        self._lang_ids = torch.tensor([tokenize(text)], device=self.device)
        self._subtask_ids = torch.tensor([tokenize(self._subtask)], device=self.device)
        self._meta = [{"speed": "normal", "quality": 1.0, "mistake": False,
                       "control_mode": "keyboard"}]

    @staticmethod
    def _oracle_subtask(text: str) -> str:
        # derive a subtask phrase from the instruction (simple lexical mapping)
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
        """Push a single 48x48 frame (or float history) into the buffer."""
        f = torch.as_tensor(frame, dtype=torch.float32, device=self.device)
        if f.ndim == 2:
            f = f.unsqueeze(0)
        if self._buffer is None:
            self._buffer = f.repeat(self.history, 1, 1).unsqueeze(0)  # (1,K,48,48)
        else:
            self._buffer = torch.cat([self._buffer[:, 1:], f.unsqueeze(0).unsqueeze(1)], dim=1)

    # -- inference -------------------------------------------------------- #
    @torch.no_grad()
    def _world_step(self):
        if self._buffer is None:
            raise RuntimeError("observe() before step()")
        sub = self.wm.sample(self._buffer, self._lang_ids, self._subtask_ids, self._meta)
        self._subgoal = sub.clamp(-1.0, 1.0)  # keep subgoal in valid image range

    @torch.no_grad()
    def step(self) -> torch.Tensor:
        if self._buffer is None:
            raise RuntimeError("observe() before step()")

        if self._subgoal is None:
            self._world_step()
        elif self.synchronous:
            self._world_step()  # refresh every step in synchronous mode

        chunk = self.ae.sample(self._buffer, self._subgoal, self._lang_ids,
                               self._subtask_ids, self._meta)
        return chunk.squeeze(0)  # (H, 4) continuous

    # -- helpers ---------------------------------------------------------- #
    @property
    def instruction(self) -> str:
        return self._instruction

    @property
    def subtask(self) -> str:
        return self._subtask

    @property
    def subgoal(self) -> torch.Tensor | None:
        return self._subgoal
