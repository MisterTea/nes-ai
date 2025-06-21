import numpy as np

from super_mario_env_search import _to_controller_presses


NdArrayUint8 = np.ndarray[np.dtype[np.uint8]]


_MASK_START_AND_SELECT = _to_controller_presses(['start', 'select']).astype(bool)


def flip_buttons(controller_presses: NdArrayUint8, flip_prob: float, ignore_button_mask: NdArrayUint8) -> NdArrayUint8:
    flip_mask = np.random.rand(len(controller_presses)) < flip_prob   # True where we want to flip
    flip_mask[ignore_button_mask] = 0
    result = np.where(flip_mask, 1 - controller_presses, controller_presses)
    return result


def flip_buttons_in_place(controller_presses: NdArrayUint8, flip_prob: float, ignore_button_mask: NdArrayUint8) -> NdArrayUint8:
    flip_mask = np.random.rand(len(controller_presses)) < flip_prob
    flip_mask[ignore_button_mask] = False
    controller_presses[flip_mask] ^= 1  # In-place bitwise flip (0->1, 1->0)
    return controller_presses


def flip_action_index_uniform_prob(current_action_index: int, num_actions: int, button_flip_prob: float, num_buttons: int) -> int:
    # NOTE: The transition probabilities here don't actually map to the probability of flipping buttons!
    #
    # Treat transitioning to a new action index as similar to flipping buttons.
    #
    # When flipping buttons, we have 8 buttons total, with 2 masked out (select and start).
    # There are 2^6, or 64 total combinations.
    #
    # Say the probability of flipping a button is 0.03.  So the probability of transitioning to a
    # new state is:
    #   P(new state) = P(button0=same) * P(button1=same) * ... P(button7=same)
    # or equivalently:
    #   P(new state) = 1 − P(no bits flip) = 1 − (1−p)
    # which is:
    #   P(new state) = 1 - (1 - 0.03)^6
    # ->:
    #   P(new state) = 0.167
    #
    # So, the probability of transitioning to a new state is 0.167, and the probability of staying
    # in the same state is 0.833.
    #
    # Now, we want to transition to another state with probability 0.833.

    p_new_state = 1 - (1 - button_flip_prob) ** num_buttons
    p_same_state = 1 - p_new_state

    # Probability of selecting a new action is distributed over all possible actions.
    p_new_state_i = p_new_state / (num_buttons - 1)

    probs = np.full(num_actions, fill_value=p_new_state_i)
    probs[current_action_index] = p_same_state

    # Probabilities should already be normalized.  We normalize again to avoid small numerical errors.
    probs /= probs.sum()

    choice_action_index = np.random.choice(num_actions, p=probs)

    return choice_action_index


def _hash_controller(controller_presses: NdArrayUint8) -> int:
    assert len(controller_presses) <= 8, f"Controller presses is too large for hashing: {len(controller_presses)} > 8"
    return controller_presses.view(np.uint64).item()


ACTION_INDEX_TO_CONTROLLER = [
    _to_controller_presses([]),
    _to_controller_presses(['up']),
    _to_controller_presses(['down']),
    _to_controller_presses(['a']),
    _to_controller_presses(['left']),
    _to_controller_presses(['left', 'a']),
    _to_controller_presses(['left', 'b']),
    _to_controller_presses(['left', 'a', 'b']),
    _to_controller_presses(['right']),
    _to_controller_presses(['right', 'a']),
    _to_controller_presses(['right', 'b']),
    _to_controller_presses(['right', 'a', 'b']),
]

CONTROLLER_TO_ACTION_INDEX = {
    _hash_controller(controller): i
    for i, controller in enumerate(ACTION_INDEX_TO_CONTROLLER)
}


def _hamming_prob(a: NdArrayUint8, b: NdArrayUint8, flip_prob: float):
    """
    Compute the probability of transitioning from bit vector `a` to `b`
    assuming each bit flips independently with probability `p`.

    The formula is:
        P(a → b) = p^d * (1 - p)^(n - d)
    where:
        d = Hamming distance between `a` and `b`
        n = number of bits (buttons)

    For example:
        a = [0, 0, 1, 0, 0, 0]   # "left"
        b = [0, 0, 1, 0, 1, 0]   # "left+a"
        d = 1  (only 'a' differs)
        P = (0.03)^1 * (0.97)^5 ≈ 0.03 * 0.8587 ≈ 0.0258
    """
    p = flip_prob
    d = np.sum(a != b)
    n = len(a)
    return (p ** d) * ((1 - p) ** (n - d))


def build_controller_transition_matrix(actions: list[NdArrayUint8], flip_prob: float) -> np.ndarray:
    """
    Build a transition matrix where each element T[i, j] represents the
    probability of transitioning from action i to action j due to random
    independent bit flipping.

    Inputs:
        actions: list of N bit vectors of length K (each is a numpy array of 0/1)
        p: flip probability for each bit

    Outputs:
        T: (N, N) numpy array of transition probabilities

    E.g.:
        From \ To    left   left+a left+a+b right+a a
        ------------ ------ ------ -------- ------- ------
        left         0.8360 0.0258 0.0008   0.0008  0.0008
        left+a       0.0258 0.8360 0.0258   0.0008  0.0008
        left+a+b     0.0008 0.0258 0.8360   0.0008  0.0258
        right+a      0.0008 0.0008 0.0008   0.8360  0.0258
        a            0.0008 0.0258 0.0008   0.0258  0.8360

    Example usage:
        Suppose we define 5 action combinations using 6 buttons: [up, down, left, right, a, b]
        combos = [
            "left"        -> [0, 0, 1, 0, 0, 0]
            "left+a"      -> [0, 0, 1, 0, 1, 0]
            "left+a+b"    -> [0, 0, 1, 0, 1, 1]
            "right+a"     -> [0, 0, 0, 1, 1, 0]
            "a"           -> [0, 0, 0, 0, 1, 0]
        ]
        The transition probability from "left+a" to "left+a+b" is:
        d = 1 (only 'b' bit flips)
        T[i, j] = (0.03)^1 * (0.97)^5 ≈ 0.0258
    """

    n_actions = len(actions)
    T = np.zeros((n_actions, n_actions))

    for i in range(n_actions):
        for j in range(n_actions):
            T[i, j] = _hamming_prob(actions[i], actions[j], flip_prob=flip_prob)
        T[i] /= T[i].sum()  # Normalize row to sum to 1
    return T


CONTROLLER_TRANSITION_MATRIX_P_03 = build_controller_transition_matrix(actions=ACTION_INDEX_TO_CONTROLLER, flip_prob=0.03)


def flip_buttons_by_action_in_place(
    controller_presses: NdArrayUint8,
    transition_matrix: np.ndarray,
    controller_to_action_index: dict[NdArrayUint8, int] = CONTROLLER_TO_ACTION_INDEX,
    action_index_to_controller: list[NdArrayUint8] = ACTION_INDEX_TO_CONTROLLER,
) -> NdArrayUint8:
    """
    Sample the next action index given the current index and transition matrix.
    """
    controller_presses_hash = _hash_controller(controller_presses)

    # Map the controller presses to their action index.
    current_action_index = controller_to_action_index[controller_presses_hash]

    # Perform an action transition.
    probs = transition_matrix[current_action_index]
    new_action_index = np.random.choice(len(probs), p=probs)

    # If we picked the same action, return the current controller.
    if new_action_index == current_action_index:
        return controller_presses

    # If we picked a new action.  Update the controller.
    new_controller = action_index_to_controller[new_action_index]
    controller_presses[:] = new_controller[:]

    return controller_presses
