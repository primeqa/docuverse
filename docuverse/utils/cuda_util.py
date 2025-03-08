import torch
from docuverse.utils.timer import timer
from time import sleep
from docuverse.utils.ticker import Ticker

util=0
n=0
tm=timer()
tk = Ticker("Utilization: ", step=100)
while True:
    ut=torch.cuda.utilization()
    util += ut
    n += 1
    sleep(0.1)
    if n%10==0:
        # print(f"Time elapsed: {tm.time_since_beginning()}; Utilization: {util/n/100:.1%}")
        tk.tick(force=True, new_value=f"{util/n/100:.1%}")
