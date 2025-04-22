# distutils: language=c++
# cython: language_level=3
# cython: profile=False

from time import time
from random import shuffle, choice
from typing import Union, Dict, List, Tuple, Hashable

cimport cython
from libc.math cimport log, sqrt, INFINITY
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.pair cimport pair


Move = Hashable # Moves can be any type, as long as they can be added to a set and used as dict keys.
Team = Union[int,str]
Rewards = Dict[Team,float]


class GameState:
    def get_current_team(self) -> Team:
        """The identifier of the current player's team."""
        raise NotImplementedError("GameState must implement get_current_team.")

    def get_legal_moves(self) -> List[Move]:
        """List of all legal moves from this state."""
        raise NotImplementedError("GameState must implement get_legal_moves.")

    def make_move(self, move:Move) -> GameState:
        """A new GameState--the result of applying the given move to this state.
        Note: The current state (self) should NOT be modified. Rather, modify a copy of it.
        """
        raise NotImplementedError("GameState must implement make_move.")

    def is_terminal(self) -> bool:
        """Is the game over?"""
        raise NotImplementedError("GameState must implement is_terminal.")

    def get_reward(self) -> Union[float,Rewards]:
        """The reward earned by the team that played the game-ending move (i.e. the team from the previous state).
        Typically 1 for win, -1 for loss, 0 for draw.
        Alternatively, returns a dict of teams/rewards: {team1:reward1, team2:reward2, ...}
        Note: This method is only called on terminal states.
        """
        raise NotImplementedError("GameState must implement get_reward.")

    def suggest_move(self) -> Tuple[Move,GameState]:
        """Optionally implement this method to serve as a rollout/playout policy.
        If not implemented, MCTS will choose a random legal move (random rollouts).
        Note: This method is only called on non-terminal states.
        """
        pass


cdef class Node:
    """Represents a game state node in the MCTS search tree.
    Args:
        state (GameState): The game state at the current node.
        parent (Node): The parent of the current node in the search tree.
        move (Move): The move that was played from the parent node to get to this node.
    Attributes:
        children (Dict[Move,Node]): The child nodes of the current node. These represent legal moves that have been visited.
        visits (int): The number of times the node has been visited.
        rewards (dict): All teams' rewards obtained from simulations through the node. Keys are teams; values are rewards.
        is_terminal (bool): Whether the node represents a terminal state.
        is_fully_expanded (bool): Whether all children of the node have been visited.
        remaining_moves (list): A list of moves that have not yet been tried.
    """
    cdef state, move, children, remaining_moves
    cdef Node parent
    cdef string team
    cdef map[string,double] rewards, rave_rewards
    cdef unsigned int visits, rave_visits
    cdef bint is_terminal, is_fully_expanded
    cdef double sqrtlog_visits, invsqrt_visits, avg_reward, avg_rave_reward

    def __init__(self, state:GameState, parent:Node=None, move:Move=None):
        self.state = state
        self.parent = parent
        self.move = move

        self.children:Dict[Move,Node] = {}
        self.visits = 0
        self.rave_visits = 0
        self.rewards = map[string,double]()
        self.rave_rewards = map[string,double]()
        self.is_terminal = self.state.is_terminal()
        self.is_fully_expanded = self.is_terminal
        if self.is_fully_expanded:
            self.remaining_moves = []
        else:
            self.remaining_moves = self.state.get_legal_moves()
            shuffle(self.remaining_moves)

        # The following are cached for performance.
        self.team = str(self.state.get_current_team()).encode()
        if self.parent is not None:
            self.rewards[self.parent.team] = 0
            self.rave_rewards[self.parent.team] = 0
        self.sqrtlog_visits = 0
        self.invsqrt_visits = 0
        self.avg_reward = 0
        self.avg_rave_reward = 0

    def get_state(self) -> GameState: return self.state
    def get_team(self) -> str: return self.team.decode()
    def get_move(self) -> Move: return self.move

    def get_parent(self) -> Node: return self.parent
    def get_children(self) -> Dict[Move,Node]: return self.children

    def get_visits(self) -> int: return self.visits
    def get_rewards(self) -> dict: return {item.first.decode():item.second for item in self.rewards}
    def get_avg_reward(self) -> float: return self.avg_reward

    def get_rave_visits(self) -> int: return self.rave_visits
    def get_rave_rewards(self) -> dict: return {item.first.decode():item.second for item in self.rave_rewards}
    def get_avg_rave_reward(self) -> float: return self.avg_rave_reward

    @cython.cdivision(True)
    cdef void visit(self, map[string,double] crewards, set moves):
        """Update this node's visits and rewards, and cache some variables for more efficient score calculation."""
        self.visits += 1

        self.sqrtlog_visits = sqrtlog(self.visits)
        self.invsqrt_visits = invsqrt(self.visits)

        # Update regular rewards.
        cdef pair[string,double] item
        cdef double total_reward = 0
        for item in crewards:
            if self.rewards.count(item.first) == 0:
                self.rewards[item.first] = 0
            self.rewards[item.first] += item.second
            total_reward += self.rewards[item.first]
        if self.parent is not None:
            # Average reward: (reward for my parent's team - rewards for all other teams) / num visits to this node
            self.avg_reward = ((2*self.rewards[self.parent.team]) - total_reward) / self.visits

        # Update RAVE rewards.
        #.only if rave bias is not 0?
        if len(moves) == 0:
            return
        cdef unsigned int rave_visits = 0
        cdef double total_rave_reward = 0
        for move in list(self.children.keys()) + self.remaining_moves:
            if move in moves:
                rave_visits += 1
        if rave_visits > 0:
            self.rave_visits += rave_visits
            for item in crewards:
                if self.rave_rewards.count(item.first) == 0:
                    self.rave_rewards[item.first] = 0
                self.rave_rewards[item.first] += rave_visits * item.second
                total_rave_reward += self.rave_rewards[item.first]
            if self.parent is not None:
                self.avg_rave_reward = ((2*self.rave_rewards[self.parent.team]) - total_rave_reward) / self.rave_visits

    cpdef double uncertainty(self, double exploration_bias):
        """The exploration term of the UCT formula: C * sqrt(ln(N) / n)
        Used to establish the upper and lower confidence bounds around the base score.
        """
        return exploration_bias * self.parent.sqrtlog_visits * self.invsqrt_visits

    @cython.cdivision(True)
    cpdef double rave_ratio(self, double rave_bias):
        """Determines the relative weight of RAVE rewards in the final score calculation.
        The RAVE effect diminishes with more visits.
        """
        if self.rave_visits == 0 or rave_bias == 0:
            return 0
        elif self.visits == 0:
            return 1
        return rave_bias / (rave_bias + self.visits)

    cpdef double base_score(self, double rave_bias):
        if rave_bias == 0: return self.avg_reward
        cdef double rave_ratio = self.rave_ratio(rave_bias)
        return ((1-rave_ratio) * self.avg_reward) + (rave_ratio * self.avg_rave_reward)

    cpdef double score(self, double exploration_bias, double rave_bias):
        return self.base_score(rave_bias) + self.uncertainty(exploration_bias)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef Node best_child(self, double exploration_bias, double rave_bias, double pruning_bias):
        cdef Node child
        cdef Node best_child = next(iter(self.children.values())) # Arbitrarily start with the first child to avoid compilation errors.
        cdef double score
        cdef double best_score = -INFINITY

        # Find the best child.
        lessers = []
        for move, child in self.children.items():
            score = child.score(exploration_bias, rave_bias)
            lessers.append((child, move, score))
            if score > best_score:
                best_score = score
                best_child = child

        # Prune any child whose optimistic estimate (UCB) is much worse than the best child's pessimistic estimate (LCB).
        # The highest pruning threshold (for pruning_bias=1) is equal to best child's LCB.
        # The lowest pruning threshold (for pruning_bias~=0) is equivalent to best child's LCB minus the lesser child's uncertainty.
        cdef double best_lcb, pruning_threshold
        if pruning_bias > 0:
            best_lcb = best_score - (2 * best_child.uncertainty(exploration_bias))
            for (child, move, score) in lessers:
                pruning_threshold = best_lcb - (child.uncertainty(exploration_bias) * (1-pruning_bias))
                if score <= pruning_threshold:
                    del self.children[move]

        return best_child


cdef class MCTS:
    cdef double _exploration_bias, _rave_bias, _pruning_bias

    def __init__(self, exploration_bias:float=1.414, rave_bias:float=0, pruning_bias:float=0):
        """Initializes an MCTS agent.
        Args:
            exploration_bias (float): Balances exploration (favoring untested moves) and exploitation (favoring good moves).
            rave_bias (float): Balances RAVE rewards (from unchosen but simulated moves) with regular rewards (from chosen moves).
            pruning_bias (float): Determines the aggressiveness of pruning.
        """
        if exploration_bias < 0: raise ValueError('Invalid exploration_bias. Must be non-negative.')
        if rave_bias < 0: raise ValueError('Invalid rave_bias. Must be non-negative.')
        if pruning_bias < 0 or pruning_bias > 1: raise ValueError('Invalid pruning_bias. Must be in range [0,1].')

        self._exploration_bias = exploration_bias
        self._rave_bias = rave_bias
        self._pruning_bias = pruning_bias

    @property
    def exploration_bias(self) -> float: return self._exploration_bias
    @property
    def rave_bias(self) -> float: return self._rave_bias
    @property
    def pruning_bias(self) -> float: return self._pruning_bias

    def search(self, state:GameState, *, max_iterations:int=None, max_time:Union[int,float]=None, return_type:str="state") -> Union[GameState,Move,Node]:
        """Searches for this state's best move until some limit has been reached.
        Args:
            state (GameState): The game state for which to find the best move.
            max_iterations (int): The maximum number of simulations to perform.
            max_time (int|float): The maximum time to search, in seconds.
            return_type (str): One of "state", "move", or "node".
        Returns:
            GameState: A new game state which is the result of applying the best move to the given state.
        """
        if max_iterations is None and max_time is None:
            raise ValueError('At least one of [max_iterations,max_time] is required.')
        if return_type not in {'state','move','node'}:
            raise ValueError(f'Invalid return_type "{return_type}" must be one of {{"state","move","node"}}')

        # Set stopping conditions.
        cdef unsigned int end_iteration
        if max_iterations is not None:
            end_iteration = max_iterations
        cdef double end_time
        if max_time is not None:
            end_time = time() + max_time

        cdef Node node = Node(state)
        cdef unsigned int i = 0
        while True:
            self.execute_round(node)

            # Check stopping conditions.
            i += 1
            if i%10 == 0:
                print("iteration",i)
            if max_iterations is not None:
                if end_iteration <= i:
                    break
            if max_time is not None:
                if end_time <= time():
                    break

        # Find the child with the highest average reward.
        # cdef Node best = max(node.children.values(), key=lambda x: x.avg_reward)
        # Find the child with the most visits.
        # cdef Node best = max(node.children.values(), key=lambda x: x.visits)
        cdef Node best = node.best_child(0, 0, 0)

        if return_type == "state":
            return best.state
        elif return_type == "move":
            return best.move
        elif return_type == "node":
            return best

    cdef void execute_round(self, Node node):
        """Step 1-2: Selection/Expansion"""
        node.state.refresh_env()
        node = self.select(node)

        """Step 3: Simulation"""
        cdef map[string,double] crewards
        cdef set moves
        (crewards, moves) = MCTS.simulate(node)

        """Step 4: Backpropagation"""
        if self._rave_bias == 0:
            moves = set()
        MCTS.backpropagate(node, crewards, moves)

    cdef Node select(self, Node node):
        while not node.is_terminal:
            if node.is_fully_expanded:
                node = node.best_child(self._exploration_bias, self._rave_bias, self._pruning_bias)
                node.state.refresh_env()
            else:
                return MCTS.expand(node)
        return node

    @staticmethod
    cdef Node expand(Node node):
        move = node.remaining_moves.pop()
        if len(node.remaining_moves) == 0:
            node.is_fully_expanded = True
        cdef Node child = Node(state=node.state.make_move(move), parent=node, move=move)
        node.children[move] = child
        return child

    @staticmethod
    cdef Tuple[map[string,double], set] simulate(Node node):
        cdef set moves = set()
        state = node.state

        if node.is_terminal:
            # If terminal, no simulation needed. Just get the reward.
            reward = state.get_reward()
            if not isinstance(reward, dict):
                reward = {node.parent.state.get_current_team(): reward}
        else:
            # Do the rollout.
            #if callable(getattr(state, 'suggest_move', False)):
            if False:
                # If the GameState has its own rollout/playout policy, use it.
                while not state.is_terminal():
                    prev_state = state
                    (move, state) = state.suggest_move()
                    moves.add(move)
            else:
                # Else, use the random rollout policy.
                while not state.is_terminal():
                    prev_state = state
                    move = choice(state.get_legal_moves())
                    moves.add(move)
                    state = state.make_move(move)
            # Prep the reward dict.
            reward = state.get_reward()
            if not isinstance(reward, dict):
                reward = {prev_state.get_current_team(): reward}

        # Convert rewards from Python dict to C map.
        cdef map[string,double] crewards = map[string,double]()
        for team in reward:
            if reward[team] != 0:
                crewards[str(team).encode()] = float(reward[team])

        return (crewards, moves)

    @staticmethod
    cdef void backpropagate(Node node, map[string,double] crewards, set moves):
        node.visit(crewards, moves)
        if node.parent is not None:
            MCTS.backpropagate(node.parent, crewards, moves)


# Cache some math to speed up Node visits.
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double sqrtlog(unsigned int x):
    return SQRTLOG[x] if 0 <= x < CACHE_MAX else sqrt(log(x))
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double invsqrt(unsigned int x):
    return INVSQRT[x] if 0 <= x < CACHE_MAX else 1 / sqrt(x)

cdef unsigned int CACHE_MAX = 10000
cdef double[10000] SQRTLOG, INVSQRT
cdef unsigned int x = 0
with cython.cdivision(True), cython.boundscheck(False), cython.wraparound(False):
    SQRTLOG[0] = 0
    INVSQRT[0] = 0
    for x in range(1,CACHE_MAX):
        SQRTLOG[x] = sqrt(log(x))
        INVSQRT[x] = 1 / sqrt(x)