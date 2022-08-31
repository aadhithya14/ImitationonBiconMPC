import matplotlib.pyplot as plt

def prepare_plot(wide=True):
	# TODO: Set default color set for lines
	# TODO: Only call once at the beginning?

    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams['xtick.major.size'] = 7
    plt.rcParams['xtick.major.width'] = 2
    plt.rcParams['ytick.major.size'] = 7
    plt.rcParams['ytick.major.width'] = 2
    plt.rcParams['font.size'] = 14
    plt.rcParams['grid.linestyle'] = ':'

    #plt.grid(True)

    if wide:
    	plt.rcParams["figure.figsize"] = (15, 6)
    else:
    	plt.rcParams["figure.figsize"] = (10, 10)
