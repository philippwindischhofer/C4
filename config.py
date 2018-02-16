import constants

class ModelConfig:
    num_conv_layers = 2
    num_cnn_filters = 128
    size_cnn_filter = 3
    l2_reg = 1e-4
    num_rows = constants.GAME_ROWS
    num_cols = constants.GAME_COLS

    value_dense_units = 256

class DeepPlayerConfig:
    MCTS_iter = 40
    temperature_switch_moves = 5
