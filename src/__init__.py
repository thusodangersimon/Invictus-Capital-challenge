import numpy as np
import pylab as plt
from IPython.display import clear_output


def moving_average(a, n: int = 3) -> np.array:
    """
    Calculates the moving average over a window
    :param a: vector or list to average
    :param n: windo
    :return: moving average
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def watch_episode(logger, n_episodes: int = 1):
    """
    Shows images of a pong episode
    :param logger: logger used to record match
    :param n_episodes: number of episode to play
    :return:
    """

    # play match
    for episode in logger.observations:
        for steps in logger.observations[episode]:
            clear_output(wait=True)
            plt.imshow(steps)
            plt.show()
        if n_episodes - 1 >= episode:
            break
