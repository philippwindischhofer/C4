from board import *
from constants import *
from RandomPlayer import *
from Game import *
from model import *
from DeepPlayer import *
from Trainer import *
from utils import Evaluator

from config import ModelConfig

config = ModelConfig()
model = C4Model(config)
model.build()

# make a trainer to train the model on self-play data
model.load(filename = 'best-1.tar') # keep going from the previous state of the art
trainer = Trainer(model)
trainer.setup()
trainer.train_epoch(games = 100, training_epochs = 50, generations = 10)

# evaluate the trained model against a random player
#model.load(filename = 'best-1.tar')
#wins, draws = Evaluator.combat_random(model, 40)
#print("wins = " + str(wins))

# evaluate the trained model against a human player
#model.load(filename = 'best-1.tar')
#Evaluator.combat_human(model, 1)
