"""Web demo: a browser UI for the cascade WAM.

Flow (fixed by design):
  1. A scene is generated FIRST (random target + distractors + obstacle) and
     rendered for the user to see, with a list of what's in it.
  2. The user types an instruction describing an object they see.
  3. Run -> the agent navigates to that object (re-targeting the env to the
     named object), streaming observation + async subgoal + action chunk live.

    python scripts/web_demo.py [--port 8000] [--device cuda]
"""

import argparse
import asyncio
import base64
import os

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from training.utils import build_world_model, build_action_expert
from models.cascade_wam import CascadeWAM
from game.env import GameEnv
from game.language import parse_instruction
from game.tasks import generate_task
from game.renderer import to_uint8

ACTION_NAMES = ("W", "A", "S", "D")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI()

_AGENT = None
_CURRENT_TASK = None


def build_agent(wm_path, ae_path, device):
    global _AGENT
    if _AGENT is None:
        wm, _ = build_world_model(wm_path, device)
        ae, _ = build_action_expert(ae_path, device)
        _AGENT = CascadeWAM(wm, ae, device=device, subgoal_update_every=5,
                            async_mode=True)
    return _AGENT


def _enc(u8: np.ndarray) -> str:
    return base64.b64encode(u8.astype(np.uint8).tobytes()).decode()


def _subgoal_u8(agent) -> np.ndarray:
    sg = agent.subgoal
    if sg is None:
        return np.zeros((48, 48), dtype=np.uint8)
    arr = sg[0, 0].detach().cpu().numpy()          # [-1, 1]
    return to_uint8((arr + 1.0) / 2.0)             # -> [0, 1] -> uint8


def _retarget(task, shape, fill) -> bool:
    """Make the object matching (shape, fill) the new target.  Returns success."""
    all_objs = [task.target] + task.distractors
    for obj in all_objs:
        if obj.semantic() == (shape, fill):
            task.target = obj
            task.distractors = [o for o in all_objs if o is not obj]
            return True
    return False


def new_scene(min_distractors=2, max_distractors=4, n_obstacles=1):
    global _CURRENT_TASK
    rng = np.random.default_rng(int(np.random.randint(0, 2 ** 31)))
    _CURRENT_TASK = generate_task(rng, min_distractors=min_distractors,
                                  max_distractors=max_distractors,
                                  n_obstacles=n_obstacles)
    env = GameEnv()
    env.reset(seed=0, task=_CURRENT_TASK)
    objects = [{"shape": o.shape, "fill": o.fill,
                "x": int(round(o.x)), "y": int(round(o.y))}
               for o in _CURRENT_TASK.objects()]
    return {
        "type": "scene",
        "obs": _enc(to_uint8(env.render())),
        "objects": objects,
        "player": [int(round(_CURRENT_TASK.player_xy[0])),
                   int(round(_CURRENT_TASK.player_xy[1]))],
    }


def run_episode(agent, instruction, max_steps=200):
    if _CURRENT_TASK is None:
        yield {"type": "error", "message": "No scene yet — click 'new scene' first."}
        return
    try:
        sem = parse_instruction(instruction)
    except ValueError as e:
        yield {"type": "error",
               "message": f"Couldn't parse '{instruction}' — use a shape + fill, e.g. "
                          "'go to the hollow triangle'."}
        return

    shape, fill = sem["shape"], sem["fill"]
    if fill is None:
        # shape-only instruction: pick any object of that shape (prefer the current target)
        cand = [o for o in _CURRENT_TASK.objects() if o.shape == shape]
        if not cand:
            yield {"type": "error", "message": f"No {shape} in this scene."}
            return
        target_obj = cand[0]
        fill = target_obj.fill
    if not _retarget(_CURRENT_TASK, shape, fill):
        present = ", ".join(f"{o.fill} {o.shape}" for o in _CURRENT_TASK.objects())
        yield {"type": "error",
               "message": f"No {fill} {shape} in this scene. Present: {present}."}
        return

    env = GameEnv()
    env.reset(seed=0, task=_CURRENT_TASK)
    agent.reset()
    agent.set_instruction(instruction)

    obs = env.get_observation()
    step = 0
    done = False
    info = {}
    while not done and step < max_steps:
        agent.observe(obs[-1])
        chunk = agent.step()
        chunk_idx = chunk.argmax(-1).tolist()
        exec_chunk = chunk_idx[:agent.execute_steps]
        for a in exec_chunk:
            obs, r, done, info = env.step(int(a))
            if done:
                break
        step += 1
        yield {
            "type": "frame",
            "obs": _enc(to_uint8(env.render())),
            "subgoal": _enc(_subgoal_u8(agent)),
            "chunk": [ACTION_NAMES[i] for i in chunk_idx],
            "exec": [ACTION_NAMES[i] for i in exec_chunk],
            "step": step,
            "success": bool(info.get("success", False)),
            "done": bool(done),
            "instruction": instruction,
            "subtask": agent.subtask,
        }
    yield {"type": "done", "success": bool(info.get("success", False)),
           "steps": step}


@app.get("/")
async def index():
    return FileResponse(os.path.join(HERE, "web_demo", "index.html"))


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    agent = _AGENT
    try:
        while True:
            msg = await websocket.receive_json()
            t = msg.get("type")
            if t == "new_scene":
                await websocket.send_json(new_scene())
            elif t == "run":
                instruction = msg.get("instruction", "Go to the hollow triangle.")
                delay_ms = float(msg.get("delay_ms", 0.0))
                for frame in run_episode(agent, instruction):
                    await websocket.send_json(frame)
                    if delay_ms > 0 and frame.get("type") == "frame":
                        await asyncio.sleep(delay_ms / 1000.0)
            elif t == "stop":
                break
            elif t == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-model", default="checkpoints/world_model_goal_wide_best.pt")
    ap.add_argument("--action-expert", default="checkpoints/action_expert_goal_wide_best.pt")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    print(f"loading models on {args.device} ...", flush=True)
    build_agent(args.world_model, args.action_expert, args.device)
    print(f"ready -> http://localhost:{args.port}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
