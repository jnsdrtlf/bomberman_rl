import math
import os
import pickle

import numpy as np
from sklearn.neural_network import MLPRegressor

import settings
from items import Bomb
from .util import view_port_state  # noqa

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
EPSILON_MIN = 0.1
EPSILON_MAX = 0.5
EPSILON_DECAY = 1e-3  # set to 0 to always use EPSILON_MIN
ALPHA = 0.2
GAMMA = 0.3
PCA_COMP = 7 * 7 + 20
MAX_BOMB_PROB = 0.4
WAIT_PROB = 0.1

ADDITIONAL_FEATURES = 17

FEATURE_SIZE = 7 * 7 * 7 + ADDITIONAL_FEATURES

escape_combinations = [
    [[1, 0], [1, 1]],
    [[1, 0], [2, 0], [2, 1]],
    [[1, 0], [2, 0], [3, 0], [3, 1]],
    [[1, 0], [2, 0], [3, 0], [4, 0]]
]

explosion_escape_combinations = [
    [[1, 0]],
    [[1, 0], [1, 1]],
    [[1, 0], [2, 0]],
    [[1, 0], [2, 0], [2, 1]],
    [[1, 0], [2, 0], [3, 0]],
    [[1, 0], [2, 0], [3, 0], [3, 1]],
    [[1, 0], [2, 0], [3, 0], [4, 0]]
]


class MultiMultiMulti:
    def __init__(self, regressor_class, *args, **kwargs):  # noqa
        self._models = []
        self._is_model_fit = np.full((len(ACTIONS),), False, dtype=bool)
        for _ in ACTIONS:
            self._models.append(
                regressor_class(*args, **kwargs)
            )

    def fit(self, x, y):
        n_samples, action_size = y.shape
        for i, model in enumerate(self._models):
            _y = y[:, i].reshape((n_samples,))
            model.fit(x, _y)
        self._is_model_fit[:] = True  # set all entries to true

    def predict(self, x):
        n_samples, feature_size = x.shape
        prediction = np.zeros((n_samples, len(ACTIONS)))
        for i, model in enumerate(self._models):
            if not self._is_model_fit[i]:
                continue
            prediction[:, i] = self._models[i].predict(x)
        return prediction

    def fit_actions(self, x: np.ndarray, y: np.ndarray, actions: np.ndarray):
        for action in np.unique(actions):
            if action < 0 or action > len(self._models):
                raise ValueError(f"Invalid action index {action}")
            _x = x[actions == action]
            n_samples, n_features = _x.shape
            _y = y[actions == action].reshape((n_samples,))
            self._models[action].fit(_x, _y)
        self._is_model_fit[actions] = True

    @property
    def is_model_fit(self):
        return np.all(self._is_model_fit)


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.epsilon_decay = EPSILON_DECAY
    self.epsilon_min = EPSILON_MIN
    self.epsilon = EPSILON_MAX
    self.alpha = ALPHA  # Learning rate
    self.gamma = GAMMA  # Discount factor
    self.filename = "multimulti.pt"
    self.pca_filename = "pca_model.pt"
    self.waited_for = 0

    self.last_action_random = False

    # if os.path.isfile(self.pca_filename):
    #     self.pca = load_model(self.pca_filename)
    # else:
    #     self.pca = PCA(n_components=PCA_COMP)
    #     self.pca.fit_transform(np.load("states.npy"))
    #     save_model(self.pca, self.pca_filename)

    if os.path.isfile(self.filename):
        self.logger.info(f"Load existing model from {self.filename}")
        self.model = load_model(self.filename)
        # Adjust epsilon for loaded model
        # No decay anymore
        self.epsilon_decay = 0
        self.epsilon = EPSILON_MIN
    elif self.train:
        self.logger.info("Setup new model")
        hidden_layers = int(np.ceil(2 / 3 * (FEATURE_SIZE + len(ACTIONS))))
        # hidden_layers = (110, 40)
        self.model = MultiMultiMulti(
            MLPRegressor,
            hidden_layer_sizes=hidden_layers, max_iter=400, warm_start=True
        )
    else:
        error_message = f"Unable to find '{self.filename}'. Is the model trained yet?"
        self.logger.error(error_message)
        raise FileNotFoundError(error_message)


def load_model(filename: str):
    with open(filename, "rb") as file:
        return pickle.load(file)


def save_model(model, filename):
    with open(filename, "wb") as file:
        pickle.dump(model, file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    if self.train and np.random.random() <= self.epsilon:
        self.last_action_random = True
        self.logger.debug("Performing a random action.")
        possible_crates = 176
        crate_count = np.sum(game_state["field"] == 1)
        current_crate_density = crate_count / possible_crates
        bomb_prob = MAX_BOMB_PROB * current_crate_density
        move_prob = (1 - bomb_prob - WAIT_PROB) / 4
        p = [move_prob, move_prob, move_prob, move_prob, WAIT_PROB, bomb_prob]
        return np.random.choice(ACTIONS, p=p)

    self.last_action_random = False
    self.logger.debug("Querying model for action.")
    features = state_to_features(self, game_state)
    if self.model.is_model_fit:
        prediction, = self.model.predict(features)  # Unpack prediction array
        action_index = np.argmax(prediction)
        return ACTIONS[int(action_index)]
    else:
        # Very first actions
        return np.random.choice(ACTIONS, p=[.225, .225, .225, .225, .0, .1])


def state_to_features(self, game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    """
    #1 1 if way up is obstructed, 0 else
    #2 1 if way right is obstructed, 0 else
    #3 1 if way left is obstructed, 0 else
    #4 1 if way down is obstructed, 0 else
    """
    feature_shape: tuple = (ADDITIONAL_FEATURES,)
    features = np.full(feature_shape, np.nan)

    name, score, is_bomb_possible, (player_x, player_y) = game_state["self"]

    field = game_state["field"]
    coins = np.array(game_state["coins"])

    escape_path = [[0, 0]]

    blast_coords = []

    bomb_pos = [[bomb_x, bomb_y] for (bomb_x, bomb_y), bomb_int in game_state["bombs"]]

    for bomb in game_state["bombs"]:
        blast_coord = get_blast_coords(bomb, field)
        blast_coords += blast_coord

        if (player_x, player_y) not in blast_coord:
            continue

        for route in explosion_escape_combinations:
            escape_array = check_all_paths_for_route(np.array(route), field, player_x, player_y, bomb_pos,
                                                     check_blast=True, blast_coord=blast_coord)

            if np.any(escape_array):
                escape_int, = np.where(escape_array)
                escape_path = get_escape_from_path_array(escape_int[0], route)
                break

    can_escape = False

    for route in escape_combinations:
        if check_all_paths_for_route(np.array(route), field, player_x, player_y, bomb_pos):
            can_escape = True
            break

    features[0] = np.abs(field[player_x + 1, player_y]) - 0.5 + int((player_x + 1, player_y) in bomb_pos)
    features[1] = np.abs(field[player_x - 1, player_y]) - 0.5 + int((player_x - 1, player_y) in bomb_pos)
    features[2] = np.abs(field[player_x, player_y + 1]) - 0.5 + int((player_x, player_y + 1) in bomb_pos)
    features[3] = np.abs(field[player_x, player_y - 1]) - 0.5 + int((player_x, player_y - 1) in bomb_pos)

    features[4] = int(can_escape) - 0.5
    features[5] = int(escape_path[0][0] == 1) - 0.5
    features[6] = int(escape_path[0][1] == 1) - 0.5
    features[7] = int(escape_path[0][0] == -1) - 0.5
    features[8] = int(escape_path[0][1] == -1) - 0.5
    features[9] = int(is_bomb_possible) - 0.5
    features[10] = int((player_x, player_y) in blast_coords) - 0.5

    features[11] = int((player_x + 1, player_y) in blast_coords) - 0.5
    features[12] = int((player_x - 1, player_y) in blast_coords) - 0.5
    features[13] = int((player_x, player_y + 1) in blast_coords) - 0.5
    features[14] = int((player_x, player_y - 1) in blast_coords) - 0.5
    features[15] = coin_degree(player_x, player_y, coins) - 0.5
    features[16] = int(bomb_destroys_crate(player_x, player_y, field)) - 0.5

    view_port = view_port_state(game_state)
    # reduced_map = self.pca.transform(view_port.reshape(1, -1)).reshape(-1)
    reduced_map = view_port
    return np.concatenate([features, reduced_map]).reshape((1, FEATURE_SIZE))


def is_path_free(path, fields, position_x, position_y, bomb_pos, check_blast=False, blast_coord=None):
    if check_blast:
        if (position_x + path[-1][0], position_y + path[-1][1]) in blast_coord:
            return False
    for x, y in path:
        if fields[position_x + x, position_y + y] == 0 and [position_x + x, position_y + y] not in bomb_pos:
            continue
        else:
            return False
    return True


def check_all_paths_for_route(route, fields, position_x, position_y, bomb_pos, check_blast=False, blast_coord=None):
    first_neg_route = np.copy(route)
    first_neg_route[:, 0] = - first_neg_route[:, 0]
    logic_array = np.array([
        # Equal sign routes
        is_path_free(route, fields, position_x, position_y, bomb_pos, check_blast, blast_coord),
        is_path_free(-route, fields, position_x, position_y, bomb_pos, check_blast, blast_coord),
        is_path_free(route[:, ::-1], fields, position_x, position_y, bomb_pos, check_blast, blast_coord),
        is_path_free(-route[:, ::-1], fields, position_x, position_y, bomb_pos, check_blast, blast_coord),
        # Partial Negative Routes:
        is_path_free(first_neg_route, fields, position_x, position_y, bomb_pos, check_blast, blast_coord),
        is_path_free(-first_neg_route, fields, position_x, position_y, bomb_pos, check_blast, blast_coord),
        is_path_free(first_neg_route[:, ::-1], fields, position_x, position_y, bomb_pos, check_blast, blast_coord),
        is_path_free(-first_neg_route[:, ::-1], fields, position_x, position_y, bomb_pos, check_blast, blast_coord),
    ])
    if check_blast:
        return logic_array

    return np.any(logic_array)


def bomb_destroys_crate(position_x, position_y, field):
    blast_coord = get_blast_coords(((position_x, position_y), 0), field)
    for coord in blast_coord:
        if field[coord] == 1:
            return True
    return False


def get_escape_from_path_array(route_int, route):
    route = np.array(route)
    first_neg_route = np.copy(route)
    first_neg_route[:, 0] = - first_neg_route[:, 0]
    if route_int == 0:
        return route
    elif route_int == 1:
        return -route
    elif route_int == 2:
        return route[:, ::-1]
    elif route_int == 3:
        return -route[:, ::-1]
    elif route_int == 4:
        return first_neg_route
    elif route_int == 5:
        return -first_neg_route
    elif route_int == 6:
        return first_neg_route[:, ::-1]
    elif route_int == 7:
        return -first_neg_route[:, ::-1]


def get_blast_coords(bomb, fields):
    """Get Blast Coordinates

    :param bomb:
    :param fields:

    :returns: numpy array with coordinate tuples
    """
    coordinates, timer = bomb
    bomb_obj = Bomb(coordinates, "agent", timer, settings.BOMB_POWER, "DUNKELROT!")
    return bomb_obj.get_blast_coords(fields)


def coin_degree(position_x, position_y, coins):
    if coins.size == 0:
        return 1
    my_position = np.array([position_x, position_y])
    # Calculate closest coins angle
    closest_coin_ind = np.argmin(np.linalg.norm(coins - my_position, axis=1))
    closest_coin = coins[closest_coin_ind]

    # angle in radians
    radians = math.atan2(my_position[1] - closest_coin[1],
                         my_position[0] - closest_coin[0]) + np.pi

    # Only consider 8 directions in which coin lies
    coin_degree = np.degrees(radians) // 45
    if coin_degree == 8:
        coin_degree = 0

    return coin_degree / 8
