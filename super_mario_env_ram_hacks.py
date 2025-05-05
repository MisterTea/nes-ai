from typing import Any

import numpy as np

NdArrayUint8 = np.ndarray[np.dtype[np.uint8]]

# RAM hacking from:
#   https://github.com/Kautenja/gym-super-mario-bros/blob/master/gym_super_mario_bros/smb_env.py


def _skip_change_area(ram: NdArrayUint8):
    """Skip change area animations by by running down timers."""
    change_area_timer = ram[0x06DE]
    if change_area_timer > 1 and change_area_timer < 255:
        ram[0x06DE] = 1


def _get_player_state(ram: NdArrayUint8) -> np.uint8:
    """
    0x00 - Leftmost of screen
    0x01 - Climbing vine
    0x02 - Entering reversed-L pipe
    0x03 - Going down a pipe
    0x04 - Autowalk
    0x05 - Autowalk
    0x06 - Player dies
    0x07 - Entering area
    0x08 - Normal
    0x09 - Transforming from Small to Large (cannot move)
    0x0A - Transforming from Large to Small (cannot move)
    0x0B - Dying
    0x0C - Transforming to Fire Mario (cannot move)
    """
    return ram[0x000E]


def _get_y_viewport(ram: NdArrayUint8) -> np.uint8:
    """
    Return the current y viewport.

    Note:
        1 = in visible viewport
        0 = above viewport
        > 1 below viewport (i.e. dead, falling down a hole)
        up to 5 indicates falling into a hole

    """
    return ram[0x00b5]


def _is_dying(ram: NdArrayUint8):
    """Return True if Mario is in dying animation, False otherwise."""
    player_state = _get_player_state(ram)
    y_viewport = _get_y_viewport(ram)
    return player_state == 0x0b or y_viewport > 1


def _kill_mario(ram: NdArrayUint8):
    """Skip a death animation by forcing Mario to death."""
    # force Mario's state to dead
    ram[0x000e] = 0x06


def _is_world_over(ram: NdArrayUint8):
    """Return a boolean determining if the world is over."""
    # 0x0770 contains GamePlay mode:
    # 0 => Demo
    # 1 => Standard
    # 2 => End of world
    return ram[0x0770] == 2


# a set of state values indicating that Mario is "busy"
_BUSY_STATES = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x07]

def _is_busy(ram: NdArrayUint8) -> bool:
    """Return boolean whether Mario is busy with in-game garbage."""
    return _get_player_state(ram) in _BUSY_STATES


def _skip_change_area(ram: NdArrayUint8):
    """Skip change area animations by by running down timers."""
    change_area_timer = ram[0x06DE]
    if change_area_timer > 1 and change_area_timer < 255:
        ram[0x06DE] = 1


def _runout_prelevel_timer(ram: NdArrayUint8):
    """Force the pre-level timer to 0 to skip frames during a death."""
    ram[0x07A0] = 0


def _skip_occupied_states(nes: Any):
    """Skip occupied states by running out a timer and skipping frames."""
    ram = nes.ram()
    while _is_busy(ram) or _is_world_over(ram):
        _runout_prelevel_timer(ram)
        nes.run_frame()


def skip_after_step(nes: Any):
    """
    Handle any RAM hacking after a step occurs.

    Args:
        done: whether the done flag is set to true

    Returns:
        None
    """

    ram = nes.ram()

    # if mario is dying, then cut to the chase and kill hi,
    if _is_dying(ram):
        _kill_mario(ram)

    # skip world change scenes (must call before other skip methods)
    # if not self.is_single_stage_env:
    #     self._skip_end_of_world()

    # skip area change (i.e. enter pipe, flag get, etc.)
    _skip_change_area(ram)

    # skip occupied states like the black screen between lives that shows
    # how many lives the player has left
    _skip_occupied_states(nes)


def life(ram: NdArrayUint8):
    """Return the number of remaining lives."""
    return ram[0x075a]
