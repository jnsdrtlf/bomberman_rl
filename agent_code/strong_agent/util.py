import numpy as np

import settings

VIEW_SIZE: int = 7


def get_object_map(object_xy_list):
    object_map = np.zeros((settings.COLS, settings.ROWS))
    for (x, y) in object_xy_list:
        object_map[x, y] = 1
    return object_map


def view_port_state(game_state: dict) -> np.ndarray:
    """
    Features per field:

     - 0: Free
     - 1: Breakable
     - 2: Obstructed
     - 3: Contains player
     - 4: Contains coin
     - 5: Danger level
     - 6: Contains explosion
     !- 7: Contains opponent
    """
    num_features_per_tile = 7

    feature_shape: tuple = (VIEW_SIZE * VIEW_SIZE * num_features_per_tile,)
    features = np.full(feature_shape, np.nan)
    _, _, _, (player_x, player_y) = game_state["self"]
    coins = game_state["coins"]
    opponent_coords = [(x, y) for _, _, _, (x, y) in game_state["others"]]

    coin_map = get_object_map(coins)
    opponent_map = get_object_map(opponent_coords)

    origin_x = player_x
    origin_y = player_y
    if (origin_x - VIEW_SIZE // 2) < 0:
        origin_x = VIEW_SIZE // 2
    if (origin_y - VIEW_SIZE // 2) < 0:
        origin_y = VIEW_SIZE // 2

    if (origin_x + VIEW_SIZE // 2) >= settings.COLS:
        origin_x = settings.COLS - VIEW_SIZE // 2 - 1
    if (origin_y + VIEW_SIZE // 2) >= settings.ROWS:
        origin_y = settings.ROWS - VIEW_SIZE // 2 - 1

    x_range = range(origin_x - VIEW_SIZE // 2, origin_x + VIEW_SIZE // 2 + 1)
    y_range = range(origin_y - VIEW_SIZE // 2, origin_y + VIEW_SIZE // 2 + 1)
    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            field_index = (
                np.ravel_multi_index((i, j), (VIEW_SIZE, VIEW_SIZE))
                * num_features_per_tile
            )
            field = game_state["field"][x, y]
            features[field_index + 0] = int(field == 0) - 0.5
            features[field_index + 1] = int(field == 1) - 0.5
            features[field_index + 2] = int(np.abs(field) == 1) - 0.5
            features[field_index + 3] = int(player_x == x and player_y == y) - 0.5
            features[field_index + 4] = coin_map[x, y] - 0.5
            features[field_index + 5] = 0
            for (bomb_x, bomb_y), timer in game_state["bombs"]:
                if bomb_x == x and bomb_y == y:
                    features[field_index + 5] = (
                            (settings.BOMB_TIMER - timer) / settings.BOMB_TIMER
                    )
                    break
            features[field_index + 6] = int(opponent_map[x, y]) - 0.5
    assert np.all(~np.isnan(features))
    return features
