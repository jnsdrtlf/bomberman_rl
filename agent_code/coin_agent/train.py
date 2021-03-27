import os
from typing import List

import numpy as np

import events as e
from .callbacks import state_to_features

# Events
CLOSER_TO_COIN = "CLOSER_TO_COIN"
FURTHER_FROM_COIN = "FURTHER_FROM_COIN"

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.rewards_file = "rewards.npy"
    self.performance_file = "performance.npy"
    self.num_coins_collected = 0
    self.num_steps_survived = 0
    self.rewards = []

    if os.path.isfile(self.rewards_file):
        self.past_rewards = np.load(self.rewards_file)
    else:
        # Empty array to hold:
        # mean_reward, median_reward, reward_std, reward_sum
        self.past_rewards = np.full((0, 4), np.nan)

    if os.path.isfile(self.performance_file):
        self.past_performance = np.load(self.performance_file)
    else:
        self.past_performance = np.full((0, 2), np.nan)


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

    # Idea: Add your own events to hand out rewards
    if state_to_features(old_game_state) is not None:

        action = ACTIONS.index(self_action)

        old_state = state_to_features(old_game_state)
        new_state = state_to_features(new_game_state)

        old_coins = np.array(old_game_state["coins"]) - 1
        new_coins = np.array(new_game_state["coins"]) - 1

        # Check if coin was present and is still present
        if old_coins != [] and new_coins != []:

            old_pos = np.array(old_game_state["self"][-1]) - 1
            new_pos = np.array(new_game_state["self"][-1]) - 1

            old_distance = np.linalg.norm(old_coins - old_pos, axis=1)
            new_distance = np.linalg.norm(new_coins - new_pos, axis=1)

            index_old_coin = np.argmin(np.linalg.norm(old_coins - old_pos, axis=1))
            index_new_coin = np.argmin(np.linalg.norm(new_coins - old_pos, axis=1))

            # Check if it is the same coin
            if (old_coins[index_old_coin] == new_coins[index_new_coin]).all():

                if min(old_distance) > min(new_distance):

                    events.append(CLOSER_TO_COIN)
                elif min(old_distance) < min(new_distance):
                    events.append(FURTHER_FROM_COIN)

        # Update Q_matrix
        old_value = self.Q_table[old_state[0], old_state[1], old_state[2], action]

        next_max = np.max(
            self.Q_table[new_state[0], new_state[1], new_state[2], action]
        )
        reward = reward_from_events(self, events)
        self.rewards.append(reward)

        new_value = (1 - self.alpha) * old_value + self.alpha * (
            reward + self.gamma * next_max
        )

        self.Q_table[old_state[0], old_state[1], old_state[2], action] = new_value

    self.num_coins_collected += events.count(e.COIN_COLLECTED)
    self.num_steps_survived += 1


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    # Save rewards
    mean_reward = np.mean(self.rewards)
    median_reward = np.median(self.rewards)
    reward_std = np.std(self.rewards)
    reward_sum = np.sum(self.rewards)
    next_reward_values = np.array([
        [mean_reward, median_reward, reward_std, reward_sum]
    ])
    if self.past_rewards.size == 0:
        self.past_rewards = next_reward_values
    else:
        self.past_rewards = np.append(self.past_rewards, next_reward_values, axis=0)
    np.save(self.rewards_file, self.past_rewards)
    self.rewards = []

    # Save performance parameters
    next_performance_values = np.array([[
        self.num_steps_survived, self.num_coins_collected
    ]])
    if self.past_performance.size == 0:
        self.past_performance = next_performance_values
    else:
        self.past_performance = np.append(
            self.past_performance, next_performance_values, axis=0
        )
    np.save(self.performance_file, self.past_performance)
    self.num_coins_collected = 0
    self.num_steps_survived = 0

    # Save model
    np.save(self.filename, self.Q_table)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 10,
        e.KILLED_OPPONENT: 30,
        e.KILLED_SELF: -100,
        e.COIN_FOUND: 10,
        e.CRATE_DESTROYED: 3,
        e.INVALID_ACTION: -10,
        e.SURVIVED_ROUND: 100,
        CLOSER_TO_COIN: 5,
        FURTHER_FROM_COIN: -5,
    }
    reward_sum = 0

    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
