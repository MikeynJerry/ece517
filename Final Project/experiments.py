import os
from itertools import product

models = ['ChannelDQNModel', 'ImageDQNModel']
grids = ['smallGrid', 'mediumGrid']
replays = ['basic', 'priority-proportional', 'prioritized-rank']
losses = ['mse', 'huber']

for model, grid, replay, loss in product(models, grids, replays, losses):
    print(model, grid, replay, loss)
