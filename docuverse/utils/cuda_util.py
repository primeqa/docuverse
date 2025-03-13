import torch
from docuverse.utils.timer import timer
from time import sleep
from docuverse.utils.ticker import Ticker
import argparse

parser = argparse.ArgumentParser(description="Monitor GPU utilization.")
parser.add_argument("--window", type=int, default=10, help="Moving average window size for utilization (default: 10)")
args = parser.parse_args()

util = 0
n = 0
tm = timer()
tk = Ticker("Utilization: ", step=100)
vals = []
prev_sum = 0
window_size = args.window-1
while True:
    ut = torch.cuda.utilization()
    prev_sum += ut
    vals.append(ut)
    if len(vals) >= window_size:
        prev_sum -= vals.pop(0)
    # util += ut
    sleep(0.1)
    if n % args.window == 0:
        tk.tick(force=True, new_value=f"{prev_sum / len(vals) / 100:.1%}")
