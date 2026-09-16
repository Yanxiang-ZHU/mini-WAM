"""Human WASD keyboard control (optional; requires pygame).

Also provides a headless manual loop for environments without a display.
"""

from __future__ import annotations

_KEYMAP = {
    "w": 0, "up": 0,
    "a": 1, "left": 1,
    "s": 2, "down": 2,
    "d": 3, "right": 3,
}


def key_to_action(key: str) -> int | None:
    return _KEYMAP.get(key.lower())


def run_human(env, fps: int = 30):
    """Blocking pygame loop.  Press ESC to quit."""
    import pygame  # optional dependency

    pygame.init()
    scale = 10
    screen = pygame.display.set_mode((env.cfg.width * scale, env.cfg.height * scale))
    clock = pygame.time.Clock()
    env.reset(seed=0)
    action = None
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                name = pygame.key.name(e.key)
                if name in _KEYMAP:
                    action = _KEYMAP[name]
            elif e.type == pygame.KEYUP:
                action = None
        if action is not None:
            obs, r, done, info = env.step(action)
            if done:
                print("done:", info)
                env.reset(seed=None)
        frame = env.render()
        surf = pygame.surfarray.make_surface((frame * 255).astype("uint8"))
        surf = pygame.transform.scale(surf, (env.cfg.width * scale, env.cfg.height * scale))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(fps)
    pygame.quit()


if __name__ == "__main__":
    from .env import GameEnv
    run_human(GameEnv())
