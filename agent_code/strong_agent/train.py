import os
from collections import namedtuple, deque
from typing import List

import numpy as np

import events as e
import settings
from .callbacks import (  # noqa
    state_to_features,
    save_model,
    FEATURE_SIZE,
    get_blast_coords,
    ACTIONS,
)

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

TRANSITION_HISTORY_SIZE = 20

# Events
CLOSER_TO_COIN = "CLOSER_TO_COIN"
FURTHER_FROM_COIN = "FURTHER_FROM_COIN"
ESCAPABLE_BOMB = "ESCAPABLE_BOMB"
BOMB_DESTROYS_CRATE = "BOMB_DESTROYS_CRATE"
WAITED_TOO_LONG = "WAITED_TOO_LONG"
IN_BLAST_RADIUS = "IN_BLAST_RADIUS"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.rewards = []

    self.num_crates_destroyed = 0
    self.num_coins_collected = 0
    self.num_steps_survived = 0
    self.score_file = f"scores.npy"
    self.crates_coins_file = f"performance.npy"
    self.waited_for = 0
    self.killed_enemies = 0

    if os.path.isfile(self.score_file):
        self.past_scores = np.load(self.score_file)
    else:
        self.past_scores = np.full((0, 2), np.nan)
    if os.path.isfile(self.crates_coins_file):
        self.past_crates_coins = np.load(self.crates_coins_file)
    else:
        self.past_crates_coins = np.full((0, 2), np.nan)
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(
        f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}'
    )
    self.logger.debug(f"Tried to perform {self_action}")

    # Idea: Add your own events to hand out rewards
    old_features = state_to_features(self, old_game_state)
    new_features = state_to_features(self, new_game_state)

    crate, escape = check_placed_bomb(old_features, new_game_state, events)
    reward_coin_distance(old_game_state, new_game_state, events)

    if crate:
        events.append(BOMB_DESTROYS_CRATE)
    if escape:
        events.append(ESCAPABLE_BOMB)

    punish_long_wait(self, events)

    # Get rewards
    reward = reward_from_events(self, events)
    self.rewards.append(reward)

    # Save transition for batch learning
    self.transitions.append(Transition(old_features, self_action, new_features, reward))

    # Perform batch learning
    if len(self.transitions) == self.transitions.maxlen:
        train_batch(self, self.transitions)
    # Count special events for analysis
    if not self.last_action_random:
        self.num_crates_destroyed += events.count("CRATE_DESTROYED")
        self.num_coins_collected += events.count("COIN_COLLECTED")
    self.num_steps_survived += 1

    if "KILLED_OPPONENT" in events:
        self.killed_enemies += 1


def train_batch(self, batch: deque):
    batch_size: int = len(batch)
    if batch_size < 1:
        return
    self.logger.info(f"Training batch of {batch_size} transitions")
    x = np.full((batch_size, FEATURE_SIZE), np.nan)
    action_indices = np.full((batch_size,), -1, dtype=int)
    y = np.full((batch_size, 1), np.nan)
    for i, (old_features, action, new_features, reward) in enumerate(batch):
        if old_features is not None and action is not None:
            action_index = ACTIONS.index(action)
            if not self.model.is_model_fit:
                old_value = 0
                next_value = 0
            else:
                q_values = self.model.predict(old_features)[0]
                old_value = q_values[action_index]
                if new_features is not None:
                    next_value = np.max(self.model.predict(new_features))
                else:
                    next_value = 0
            q_value_updated = (1 - self.alpha) * old_value + self.alpha * (
                reward + self.gamma * next_value
            )
            action_indices[i] = action_index
            y[i] = np.array([q_value_updated])
            x[i] = old_features
    batch.clear()
    y_nan = np.any(np.isnan(y), axis=1)
    if np.sum(~y_nan) == 0:
        return
    self.model.fit_actions(x[~y_nan], y[~y_nan], action_indices[~y_nan])


def reward_coin_distance(old_game_state, new_game_state, events):
    if old_game_state is None or new_game_state is None:
        return

    old_pos = np.array(old_game_state["self"][-1])
    new_pos = np.array(new_game_state["self"][-1])
    old_coins = np.array(old_game_state["coins"])
    new_coins = np.array(new_game_state["coins"])

    persistent_coins = old_coins[np.all(np.isin(old_coins, new_coins))]

    if persistent_coins.size == 0:
        return
    else:
        (persistent_coins,) = persistent_coins
    # Calculate old distances to all persistent coins
    old_distances = np.sqrt(np.sum((persistent_coins - old_pos) ** 2, axis=-1))
    # Find the index of the previous closest coin
    closest_coin_index = np.argmin(old_distances)
    # Get the distance to this previously closest coin now
    new_distance = np.sqrt(
        np.sum(((persistent_coins[closest_coin_index] - new_pos) ** 2))
    )
    # Also get the old distance.
    # Should be the minimum of `old_distances`.
    old_distance = old_distances[closest_coin_index]
    if new_distance > old_distance:
        events.append(FURTHER_FROM_COIN)

    elif new_distance < old_distance:
        events.append(CLOSER_TO_COIN)


def punish_long_wait(self, events):
    max_wait = settings.EXPLOSION_TIMER
    if e.WAITED in events:
        self.waited_for += 1
    else:
        self.waited_for = 0
    if self.waited_for > max_wait:
        events.append(WAITED_TOO_LONG)


def check_placed_bomb(old_features, new_game_state, events):
    escapable_bomb = False
    destroy_crate = False
    field = new_game_state["field"]
    name, score, is_bomb_possible, (player_x, player_y) = new_game_state["self"]
    if "BOMB_DROPPED" in events:
        for bomb in new_game_state["bombs"]:
            (bomb_x, bomb_y), timer = bomb
            if [bomb_x, bomb_y] == [player_x, player_y]:
                if old_features is not None:
                    escapable_bomb = bool(old_features[0][4] > 0)
                blast_coord = get_blast_coords(bomb, field)
                for coord in blast_coord:
                    if field[coord] == 1:
                        destroy_crate = True
                        return destroy_crate, escapable_bomb
    return destroy_crate, escapable_bomb


def check_blast_radius(game_state, events):
    fields = game_state["field"]
    _, _, _, (player_x, player_y) = game_state["self"]
    for bomb in game_state["bombs"]:
        blast_coord = get_blast_coords(bomb, fields)
        if (player_x, player_y) in blast_coord:
            events.append(IN_BLAST_RADIUS)
            return


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(
        f'Encountered event(s) {", ".join(map(repr, events))} in final step'
    )
    self.transitions.append(
        Transition(
            state_to_features(self, last_game_state),
            last_action,
            None,
            reward_from_events(self, events),
        )
    )

    train_batch(self, self.transitions)

    if self.epsilon > self.epsilon_min:
        self.epsilon -= self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min

    mean_reward = np.mean(self.rewards)
    median_reward = np.median(self.rewards)
    reward_std = np.std(self.rewards)
    reward_sum = np.sum(self.rewards)
    if self.past_scores.size == 0:
        self.past_scores = np.array(
            [[mean_reward, median_reward, reward_std, reward_sum]]
        )
    else:
        self.past_scores = np.append(
            self.past_scores,
            np.array([[mean_reward, median_reward, reward_std, reward_sum]]),
            axis=0,
        )
    np.save(self.score_file, self.past_scores)
    self.rewards = []

    if self.past_crates_coins.size == 0:
        self.past_crates_coins = np.array(
            [
                [
                    self.num_crates_destroyed,
                    self.num_coins_collected,
                    self.num_steps_survived,
                    self.killed_enemies,
                ]
            ]
        )
    else:
        self.past_crates_coins = np.append(
            self.past_crates_coins,
            np.array(
                [
                    [
                        self.num_crates_destroyed,
                        self.num_coins_collected,
                        self.num_steps_survived,
                        self.killed_enemies,
                    ]
                ]
            ),
            axis=0,
        )
    np.save(self.crates_coins_file, self.past_crates_coins)
    self.num_crates_destroyed = 0
    self.num_coins_collected = 0
    self.num_steps_survived = 0
    self.killed_enemies = 0

    save_model(self.model, self.filename)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 200,
        e.KILLED_OPPONENT: 600,
        e.KILLED_SELF: -400,
        # e.BOMB_DROPPED: 10,
        # e.COIN_FOUND: 1,
        # e.CRATE_DESTROYED: 10,
        e.INVALID_ACTION: -50,
        # e.SURVIVED_ROUND: 100,
        BOMB_DESTROYS_CRATE: 40,
        ESCAPABLE_BOMB: 75,
        # e.MOVED_LEFT: 1,
        # e.MOVED_DOWN: 1,
        # e.MOVED_RIGHT: 1,
        # e.MOVED_UP: 1,
        # e.WAITED: 0,
        WAITED_TOO_LONG: -350,
        CLOSER_TO_COIN: 40,
        IN_BLAST_RADIUS: -7,
        # e.BOMB_DROPPED: -1,
        FURTHER_FROM_COIN: -30,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    # Useless bomb

    if e.BOMB_DROPPED in events and BOMB_DESTROYS_CRATE not in events:
        reward_sum -= 600
        # print("useless")
    # Inescapable bomb
    if e.BOMB_DROPPED in events and ESCAPABLE_BOMB not in events:
        reward_sum += game_rewards[e.KILLED_SELF]
    return reward_sum
