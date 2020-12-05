import os
from itertools import product

models = ["ChannelDQNModel", "ImageDQNModel"]
grids = ["smallGrid", "mediumGrid", "mediumClassic"]
replays = ["basic", "priority-proportional", "priority-rank"]
losses = ["mse", "huber"]

for model, grid, replay, loss in product(models, grids, replays, losses):
    model_type = "small" if model == "ChannelDQNAgent" else "stanford"
    save_dir = f"grid_{grid}_model_{model_type}_replay_{replay}_loss_{loss}"
    command = (
        f"python play.py -p {model} -l {grid} -n 3000 -x 3000 -f -q"
        f" --train-start 100 --loss-type {loss} --replay-type {replay}"
        f" --model-save-rate 100 --model-dir {save_dir} --log-dir {save_dir}"
        f" --model-type {model_type} --eps-decay 3000"
    )
    os.system(command)

    command = f"python play.py --from-experiment {save_dir} --nb-testing-episodes 1000"
    os.system(command)
