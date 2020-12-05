import os
from itertools import product

models = ["ChannelDQNModel", "ImageDQNModel"]
grids = ["smallGrid", "mediumGrid", "mediumClassic"]
replays = ["basic", "priority-proportional", "priority-rank"]
losses = ["mse", "huber"]

for model, grid, replay, loss in product(models, grids, replays, losses):
    import time

    start = time.time()
    save_dir = f"grid_{grid}_model_{model}_replay_{replay}_loss_{loss}"
    model_type = "small" if model == "ChannelDQNModel" else "stanford"
    command = (
        f"python play.py -p {model} -l smallGrid -n 3000 -x 3000 -f -q"
        f" --train-start 100 --loss-type {loss} --replay-type {replay}"
        f" --model-save-rate 100 --model-dir {save_dir} --log-dir {save_dir}"
        f" --model-type {model_type} --eps-decay 3000"
    )
    os.system(command)
    command = f"python play.py --from-config {save_dir} --nb-testing-epsiodes 1000"
    os.system(command)
    print(time.time() - start)
    break
