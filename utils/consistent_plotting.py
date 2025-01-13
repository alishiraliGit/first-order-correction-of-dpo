from matplotlib import pyplot as plt


def figure():
    fig = plt.figure(figsize=(4, 2.7))

    return fig


def subplots_adjust():
    plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.15)
