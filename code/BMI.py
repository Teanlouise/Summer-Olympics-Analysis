import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

import numpy as np
from plots import *


################ HISTOGRAM ###############################
plt.figure()
olympic_hist('BMI')
#plt.show()


################ BOXPLOT ###############################
plt.figure()
plt.subplot(1,2,1)
season_boxplot('BMI', 'Winner', 'Athletes')
plt.subplot(1,2,2)
season_boxplot('BMI', 'Medal', 'Medal Winners')
#plt.show()


############### QQ PLOT ###################

olympic_qqplot()
plt.show()
    



