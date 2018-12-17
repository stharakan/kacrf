import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool
import IPython

# read in data
data = pd.read_csv('./../data/data_623.csv')

# accuracy filters
got_better = data.dice_2 > data.dice_0
got_worse = data.dice_0 > data.dice_2

# message weight filters
omw = (data.app_weight + 1.0) * data.pair_weight
heavy_message = omw > 2.5
equal_message = (omw <= 2.5 ) & (omw >= 0.4)
light_message = omw < 0.4

# improvement
imp = data.dice_2 - data.dice_0
sort_worst = np.argsort(imp)
sort_best = np.flip(sort_worst)

# finally create Ipython session to play around
IPython.embed()
