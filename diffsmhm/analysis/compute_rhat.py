import numpy as np
import pandas as pd

from blackjax.diagnostics import potential_scale_reduction


# define files to work over
prefix = "/home/jwick/branches_diffsmhm/opt_wprp/diffsmhm/scripts/output/"
position_files = [
    prefix+"positions_14.6_8.0_9.0_01.csv",
    prefix+"positions_14.6_8.0_9.0_02.csv",
    prefix+"positions_14.6_8.0_9.0_03.csv",
    prefix+"positions_14.6_8.0_9.0_04.csv"
]

# load files iteratively to build input array
start_step = -1000


input_arr = []
for f in position_files:
    df = pd.read_csv(f)
    stack_list = []
    for k in df.keys():
        stack_list.append(df[k][start_step:].to_numpy())

    pos_np = np.vstack(stack_list).T

    input_arr.append(pos_np)

rhat = potential_scale_reduction(np.array(input_arr), chain_axis=0, sample_axis=1)

print(rhat)
