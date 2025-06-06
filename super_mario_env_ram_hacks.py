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


def _is_world_over(ram: NdArrayUint8):
    """Return a boolean determining if the world is over."""
    # 0x0770 contains GamePlay mode:
    # 0 => Demo
    # 1 => Standard
    # 2 => End of world
    return ram[0x0770] == 2


def _read_mem_range(ram: NdArrayUint8, address, length) -> int:
    """
    Read a range of bytes where each byte is a 10's place figure.

    Args:
        address (int): the address to read from as a 16 bit integer
        length: the number of sequential bytes to read

    Note:
        this method is specific to Mario where three GUI values are stored
        in independent memory slots to save processing time
        - score has 6 10's places
        - coins has 2 10's places
        - time has 3 10's places

    Returns:
        the integer value of this 10's place representation

    """
    return int(''.join(map(str, ram[address:address + length])))


def _time(ram: NdArrayUint8) -> int:
    """Return the time left (0 to 999)."""
    # time is represented as a figure with 3 10's places
    return _read_mem_range(ram, 0x07f8, 3)


def _skip_end_of_world(nes: Any):
    """Skip the cutscene that plays at the end of a world."""
    ram = nes.ram()

    if _is_world_over(ram):
        # get the current game time to reference
        time = _time(ram)
        # loop until the time is different
        while _time(ram) == time:
            # frame advance with NOP
            nes.run_frame()


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
    _skip_end_of_world(nes)

    # skip area change (i.e. enter pipe, flag get, etc.)
    _skip_change_area(ram)

    # skip occupied states like the black screen between lives that shows
    # how many lives the player has left
    _skip_occupied_states(nes)


def life(ram: NdArrayUint8):
    """Return the number of remaining lives."""
    return ram[0x075a]


# venv/lib/python3.11/site-packages/gym_super_mario_bros/_roms/decode_target.py
def decode_world_level(world: int, level: int, lost_levels: bool) -> tuple[int, int, int]:
    """
    Return the target area for target world and target stage.

    Args:
        target_world (None, int): the world to target
        target_stage (None, int): the stage to target
        lost_levels (bool): whether to use lost levels game

    Returns (int):
        the area to target to load the target world and stage

    """
    target = (world, level)

    # Type and value check the lost levels parameter
    if not isinstance(lost_levels, bool):
        raise TypeError('lost_levels must be of type: bool')
    # if there is no target, the world, stage, and area targets are all None
    if target is None:
        return None, None, None
    elif not isinstance(target, tuple):
        raise TypeError('target must be  of type tuple')
    # unwrap the target world and stage
    target_world, target_stage = target
    # Type and value check the target world parameter
    if not isinstance(target_world, int):
        raise TypeError('target_world must be of type: int')
    else:
        if lost_levels:
            if not 1 <= target_world <= 12:
                raise ValueError('target_world must be in {1, ..., 12}')
        elif not 1 <= target_world <= 8:
            raise ValueError('target_world must be in {1, ..., 8}')
    # Type and value check the target level parameter
    if not isinstance(target_stage, int):
        raise TypeError('target_stage must be of type: int')
    else:
        if not 1 <= target_stage <= 4:
            raise ValueError('target_stage must be in {1, ..., 4}')

    # no target are defined for no target world or stage situations
    if target_world is None or target_stage is None:
        return None
    # setup target area if target world and stage are specified
    target_area = target_stage
    # setup the target area depending on whether this is SMB 1 or 2
    if lost_levels:
        # setup the target area depending on the target world and stage
        if target_world in {1, 3}:
            if target_stage >= 2:
                target_area = target_area + 1
        elif target_world >= 5:
            # TODO: figure out why all worlds greater than 5 fail.
            # target_area = target_area + 1
            # for now just raise a value error
            worlds = set(range(5, 12 + 1))
            msg = 'lost levels worlds {} not supported'.format(worlds)
            raise ValueError(msg)
    else:
        # setup the target area depending on the target world and stage
        if target_world in {1, 2, 4, 7}:
            if target_stage >= 2:
                target_area = target_area + 1

    return target_world, target_stage, target_area


def encode_world_level(world_ram: int, level_ram: int) -> tuple[int, int]:
    world = world_ram + 1
    level = level_ram + 1

    if world in {1, 2, 4, 7}:
        if level >= 2:
            level = level - 1

    return (world_ram + 1, level)
