from game.physics import Player, step_player, action_to_accel, ACTION_DIRS, NUM_ACTIONS


def test_friction_only():
    p = Player(0, 0, vx=2.0, vy=0.0)
    step_player(p, (0.0, 0.0), friction=0.8, max_speed=10.0)
    assert p.vx == 2.0 * 0.8
    assert p.x == p.vx  # x += v after update


def test_acceleration_direction():
    # W = up = (0, -1)
    ax, ay = action_to_accel(0, 1.0)
    assert ax == 0.0 and ay == -1.0
    ax, ay = action_to_accel(3, 1.0)
    assert ax == 1.0 and ay == 0.0


def test_max_speed_clamp():
    p = Player(0, 0, vx=0.0, vy=0.0)
    for _ in range(100):
        step_player(p, (1.0, 0.0), friction=1.0, max_speed=2.0)
    assert abs(p.vx) <= 2.0 + 1e-6


def test_num_actions():
    assert NUM_ACTIONS == 4
    assert set(ACTION_DIRS) == {0, 1, 2, 3}
