from board import *
from constants import *
from RandomPlayer import *
from Game import *
from model import *
from DeepPlayer import *
from Trainer import *
from utils import Evaluator

#from config_manual_play import *
#from config import *
from config_deep import *

model_config = ModelConfig()
model = C4Model(model_config)
model.build()

player_config = DeepPlayerConfig()

# make a trainer to train the model on self-play data
# model.load(filename = 'best-1.tar') # keep going from the previous state of the art
trainer = Trainer(model, player_config)
trainer.setup()
trainer.train_epoch(games = 50, training_epochs = 50, generations = 50)

# evaluate the trained model against a random player
# model.load(filename = 'best-9.tar')
# wins, draws = Evaluator.combat_random(model, 100)
# print("wins = " + str(wins))

# evaluate the trained model against a human player
#model.load(folder = 'models', filename = 'model-1.tar')
#Evaluator.combat_human(model, 1, player_config)

#Evaluator.combat_humans(1)
