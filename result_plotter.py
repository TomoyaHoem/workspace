import io
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from running_metric_ret import RunningMetricAnimation


# https://stackoverflow.com/questions/14708695/specify-figure-size-in-centimeter-in-matplotlib
def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def fig_to_im(size: tuple) -> io.BytesIO:
    """
    Writes current figure to memory for later access.

    Parameters:
        size: Plot size in cm.

    Returns:
        io.BytesIO: Resized plot in memory buffer.
    """
    plt.gcf().set_size_inches(cm2inch(size))
    imgdata = io.BytesIO()
    plt.gcf().savefig(imgdata, format="JPEG", dpi=100)
    plt.clf()
    return imgdata


def single_pc_plot(res: list, tasks: list[str]) -> io.BytesIO:
    """
    Function that returns a parallel coordinate plot of the
    final populations objective values as a memory buffer.

    Parameters:
        obj_vals: List containing objective value lists for each individual.
        tasks: List containing objective functions as strings.

    Returns:
        io.BytesIO: Plot in memory buffer.
    """
    df = pd.DataFrame(res.F, columns=tasks)
    df["Algorithm"] = res.algorithm.__class__.__name__

    pd.plotting.parallel_coordinates(df, class_column="Algorithm", cols=tasks)

    return fig_to_im((16, 10))


def multi_pc_plot(results: list, tasks: list[str]) -> io.BytesIO:
    """
    Function that returns parallel coordinate plots, contrasting all algorithms.

    Parameters:
        results: Pymoo results list.
        tasks: List containing objective functions as strings.

    Returns:
        io.BytesIO: Plot in memory buffer.
    """
    dfs = []
    for res in results:
        res_df = pd.DataFrame(res.F, columns=tasks)
        res_df["Algorithm"] = res.algorithm.__class__.__name__
        dfs.append(res_df)
    df = pd.concat(dfs)

    pd.plotting.parallel_coordinates(df, class_column="Algorithm", cols=tasks)
    return fig_to_im((16, 10))


def r_plot_data(num_iter):
    """Helper function to dynamically adjust running metric plots to number of generations."""
    delta, num_p = 0, 0

    if num_iter < 10:
        delta = num_iter
        num_p = 1
    elif num_iter <= 50:
        delta = 10
        num_p = num_iter / 10
    else:
        delta = num_iter / 5
        num_p = 5
    return delta, num_p


def single_running_plot(res: list, num_iter: int) -> tuple[io.BytesIO, io.BytesIO]:
    """
    Function that returns running plot for all
    generations and the last run separated.

    Parameters:
        res: Pymoo single algorithm result.
        num_iter: Number of EA generations.

    Returns:
        tuple: Memory buffer tuple contaning both plots.
    """
    # r-plot setup
    delta, num_p = r_plot_data(num_iter)
    running = RunningMetricAnimation(
        delta_gen=delta, n_plots=num_p, key_press=False, do_show=False
    )
    # process generations
    for algorithm in res.history[:num_iter]:
        running.update(algorithm)

    # plot
    fig, ax = plt.subplots()
    running.draw(running.data, ax)
    r_plot_f = fig_to_im((16, 10))
    fig, ax = plt.subplots()
    running.draw(running.data[-1:], ax)
    r_plot_l = fig_to_im((16, 10))

    return (r_plot_f, r_plot_l)
