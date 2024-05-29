import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
import seaborn as sns
warnings.filterwarnings('ignore')
def smooth(data, sm=1):
    smooth_data = []
    if sm > 1:
        for d in data:
            z = np.ones(len(d))
            y = np.ones(sm)*1.0
            d = np.convolve(y, d, "same")/np.convolve(y, z, "same")
            smooth_data.append(d)
    return smooth_data

def make_data(taskname):
    filename='./data/'+taskname
    Steps = []
    Rs = []
    for i in [1,2,3]:
        steps = []
        rs = []
        with open(filename+str(i)+'.txt', 'r') as input_file:
            next(input_file)
            for k, line in enumerate(input_file):
                row = line.strip().split()
                steps.append(int(row[1]))
                rs.append(float(row[2]))
        Steps.append(np.array(steps))
        Rs.append(rs)

    Rs = np.array(Rs)[:,:-1]
    y_data = np.array(smooth(Rs, 19))
    mean = y_data.mean(axis=0)
    std = y_data.std(axis=0)
    lower, upper = scipy.stats.t.interval(0.95, y_data[0].shape[0], mean, std)
    return mean, std, lower, upper, Steps[0][:-1]

linestyle = ['-', '--', ':', '-.']
file = "./progress.csv"

data = pd.read_csv(file)

step = data['step']

nscore = data['normalized return mean']
nscore = smooth([nscore], 19)[0]
sns.set_theme(style="darkgrid")

fig, ax1 = plt.subplots()
plt.title("walker2d-mr")

plt.xlabel("Training Steps")
ax1.set_ylabel("Rewards")
ax1.plot(step,nscore,linewidth=1,label='sdv')

fig.legend(loc='lower right')
plt.show()
plt.savefig(fname="test.pdf", format="pdf",bbox_inches="tight")

