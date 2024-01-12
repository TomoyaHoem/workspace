import io
import itertools
import numpy as np
import pandas as pd
import selfies as sf
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Draw
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from util import (
    r_plot_data,
    dominates,
    non_dominated,
    split_string_lines,
    transparent_colors,
    algorithm_names,
)
from running_metric_ret import RunningMetricAnimation
from extended_similarity import internal_similarity

from pymoo.indicators.igd import IGD
from pymoo.util.normalization import normalize

# https://davidmathlogic.com/colorblind/#%23D81B60-%231E88E5-%23FFC107-%23004D40
colors = ["#D81B60", "#004D40", "#1E88E5"]
colors_colb = [(216, 27, 96), (0, 77, 64), (30, 136, 229)]
colors_colb_rgb = [tuple(c / 255 for c in color) for color in colors_colb]
colors__slightly_transparent = [
    tuple(c / 255 for c in color) + (0.8,) for color in colors_colb
]
colors_transparent = [tuple(c / 255 for c in color) + (0.2,) for color in colors_colb]
markers = ["s", "o", "*"]
sizes = [80, 60, 40]


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
    plt.close()
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
    df["Algorithm"] = res.name

    ax = pd.plotting.parallel_coordinates(
        df, class_column="Algorithm", cols=tasks, color=colors_transparent[1]
    )
    ax.get_legend().remove()
    ax.set_ylim(0, 1.1)

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
        res_df["Algorithm"] = algorithm_names(res.name)
        dfs.append(res_df)
    df = pd.concat(dfs)

    ax = pd.plotting.parallel_coordinates(
        df, class_column="Algorithm", cols=tasks, color=colors__slightly_transparent
    )
    # for i, leg in enumerate(ax.get_legend().legend_handles):
    #     leg.set_color(colors_colb_rgb[i])
    ax.set_ylim(0, 1.1)
    return fig_to_im((16, 10))


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
    _, ax = plt.subplots()
    running.draw(running.data, ax)
    r_plot_f = fig_to_im((16, 10))
    _, ax = plt.subplots()
    running.draw(running.data[-1:], ax)
    r_plot_l = fig_to_im((16, 10))

    return (r_plot_f, r_plot_l)


def multi_alg_running(pareto_obj: list, histories: list) -> list:
    """
    Function that implements multi algorithm comparison using running metric.
    https://www.egr.msu.edu/~kdeb/papers/c2020003.pdf
    @inproceedings{blank2020running,
    title={A running performance metric and termination criterion for evaluating evolutionary multi-and many-objective optimization algorithms},
    author={Blank, Julian and Deb, Kalyanmoy},
    booktitle={2020 IEEE Congress on Evolutionary Computation (CEC)},
    pages={1--8},
    year={2020},
    organization={IEEE}
    }

    Parameters:
        pareto_obj: List containing the objective values of the pareto fronts.
        histories: List containing the histories of each algorithm run.

    Returns:
        igd_vals: Averaged running metric values for each iteration.
    """
    # calculate IGD values for each run
    igd_vals = [[] for _ in range(len(pareto_obj))]

    for i in range(len(pareto_obj[0])):
        # merge pareto sets and remove duplicates
        merged = list(itertools.chain(*[x[i].F.tolist() for x in pareto_obj]))
        merged.sort()
        merged = list(m for m, _ in itertools.groupby(merged))
        # remove non-dominated
        for a, b in itertools.combinations(merged, 2):
            # print(f"check if {a} dominates {b}")
            if dominates(a, b):
                # print("true")
                merged.remove(b)
        # reverse objectives and scale SA to [1-10] for IGD calc (denormalized_d = normalized_d * (max_d - min_d) + min_d)
        merged = [
            [x * -1 if i != 1 else 10 - (x * (10 - 1) + 1) for i, x in enumerate(inner)]
            for inner in merged
        ]
        merged = np.array(merged)
        # get ideal and nadir
        c_F, c_ideal, c_nadir = merged, merged.min(axis=0), merged.max(axis=0)
        # normalize
        # normalize the current objective space values
        c_N = normalize(c_F, c_ideal, c_nadir)
        # normalize all previous generations with respect to current ideal and nadir for each alg
        for j in range(len(pareto_obj)):
            N = [
                normalize(p.opt, c_ideal, c_nadir) for p in histories[j][i]
            ]  # p.opt.get("F")
            # calculate IGD
            delta_f = [IGD(c_N).do(N[k]) for k in range(len(N))]
            # append
            igd_vals[j].append(delta_f)
    return igd_vals


def multi_running_plot(container: dict) -> io.BytesIO:
    """
    Function to plot running metrics and deviation for multiple algorithms

    Parameters:
        container: List containing results for multiple runs.

    Returns:
        io.ByterIO: Plot in memory buffer.
    """
    alg_names, pareto_objectives, histories = [], [], []
    # stack results
    for key, value in container.items():
        alg_names.append(key)
        pareto_objectives.append(value[0])
        histories.append(value[1])
    # calculate IGD values
    igd = multi_alg_running(pareto_objectives, histories)
    iterations = range(len(igd[0][0]))
    # average and calculate std from respective iterations
    sorted_runs = []
    for i in range(len(igd)):
        sorted_run = []
        for j in iterations:
            sorted_run.append([inner[j] for inner in igd[i]])
        sorted_runs.append(sorted_run)
    average_and_std = [
        [(np.mean(x), np.std(x)) for x in inner] for inner in sorted_runs
    ]
    # create running plots
    fig, ax = plt.subplots()
    gens = [g for g in range(1, len(igd[0][0]) + 1, 1)]
    for i in range(len(average_and_std)):
        # avg
        ax.plot(
            gens,
            [x[0] for x in average_and_std[i]],
            label=algorithm_names(alg_names[i]),
            alpha=0.9,
            linewidth=3,
            color=colors[i],
        )
        # fill
        plt.fill_between(
            gens,
            [x[0] + x[1] for x in average_and_std[i]],
            [x[0] - x[1] for x in average_and_std[i]],
            color=colors[i],
            alpha=0.1,
        )

    ax.legend()

    ax.set_xlabel("Generation")
    ax.set_ylabel("$\\varnothing_{k,t}$", rotation=0)
    ax.yaxis.set_label_coords(-0.075, 0.5)

    return fig_to_im((15, 10))


def similarity_plot(res: list) -> io.BytesIO:
    """
    Function that plots the extended similarity over iterations.

    Parameters:
        res: Pymoo single algorithm result.

    Returns:
        io.BytesIO: Plot in memory buffer.
    """
    similarities = []
    for alg in res.history:
        mols = [sf.decoder(mol.X[0]) for mol in alg.pop]
        similarities.append(internal_similarity(mols))

    iterations = range(1, len(similarities) + 1, 1)

    plt.plot(iterations, similarities)
    plt.ylim(0.3, 1.0)

    return fig_to_im((16, 10))


def multi_similiarty_plot(container: dict) -> io.BytesIO:
    """
    Function to plot average internal similarities and deviation for multiple algorithms

    Parameters:
        container: List containing results for multiple runs.

    Returns:
        io.ByterIO: Plot in memory buffer.
    """
    alg_names, similarity_data = [], []
    for key, value in container.items():
        # calculate similarity values for each algorithm run
        alg_names.append(key)
        sim_runs = []
        for res in value[0]:
            sim_run = []
            for alg in res.history:
                mols = [sf.decoder(mol.X[0]) for mol in alg.pop]
                sim_run.append(internal_similarity(mols))
            sim_runs.append(sim_run)
        # bin same iteration values
        similarities = []
        for i in range(len(sim_runs[0])):
            similarities.append([x[i] for x in sim_runs])
        # calculate and append average and standard deviation
        similarity_data.append(
            [(np.round(np.mean(x), 2), np.round(np.std(x), 2)) for x in similarities]
        )

    # plot with deviation intervals
    itr = range(len(similarity_data[0]))

    _, ax = plt.subplots()

    for i in range(len(similarity_data)):
        ax.plot(
            itr,
            [x[0] for x in similarity_data[i]],
            label=algorithm_names(alg_names[i]),
            linewidth=2,
            color=colors[i],
        )
        ax.fill_between(
            itr,
            [x[0] + x[1] for x in similarity_data[i]],
            [x[0] - x[1] for x in similarity_data[i]],
            color=colors[i],
            alpha=0.1,
        )

    ax.set_ylim(0.3, 1.0)
    ax.legend()

    ax.set_xlabel("Generation")
    ax.set_ylabel("Similarity", rotation=0)
    ax.yaxis.set_label_coords(-0.075, 0.5)

    return fig_to_im((15, 10))


def single_pareto_plot(res: list) -> io.BytesIO:
    """
    Function to plot pareto plot of single algorithm result using two objectives as specified.
    Highlight pareto front given two objectives.

    Parameters:
        res: Pymoo single algorithm result.

    Returns:
        io.ByterIO: Plot in memory buffer.
    """
    # determine non-dominated set
    nds = non_dominated(res.F[:, :2])
    # plot population and non-dominated set
    _, ax = plt.subplots()
    split = []
    pareto = np.array([res.F[i] for i in nds])
    if len(nds) < 16:
        remaining = 16 - len(nds) if len(res.F) > 15 else len(res.F) - len(nds)
        sample_set = [i for i in range(len(res.F)) if i not in nds]
        split = list(np.random.choice(sample_set, remaining, replace=False))
        pop_split = np.array([res.F[i] for i in split])
        pop = np.array([idv for i, idv in enumerate(res.F) if i not in nds + split])
        if len(pop) > 0:
            ax.scatter(pop[:, 1], pop[:, 0], color=colors_colb_rgb[1])
        ax.scatter(
            pop_split[:, 1],
            pop_split[:, 0],
            color=colors_colb_rgb[2],
            edgecolors="black",
        )
    else:
        pop = np.array([idv for i, idv in enumerate(res.F) if i not in nds])
        ax.scatter(pop[:, 1], pop[:, 0], color=colors_colb_rgb[1])
    ax.scatter(pareto[:, 1], pareto[:, 0], color=colors_colb_rgb[0], edgecolor="black")
    ax.set_ylim(0, 1.01)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("SA score")
    ax.set_ylabel("QED", rotation=0)
    ax.yaxis.set_label_coords(-0.075, 0.5)

    # annotate non-dominated set with mol image and SMILES string
    # 1. Sort non-dominated set for QED in descending order

    pareto_set = [(res.X[i], res.F[i]) for i in nds]
    pareto_sorted = sorted(pareto_set, key=lambda x: x[1][0], reverse=False)
    for i in split:
        pareto_sorted.append((res.X[i], res.F[i]))
    # 2. Annotate pareto front with molecule representation
    plt.subplots_adjust(left=0.04, right=0.4)
    for i, nd in enumerate(pareto_sorted):
        mol_smiles = sf.decoder(nd[0][0])
        mol_img = Draw.MolToImage(Chem.MolFromSmiles(mol_smiles))
        x_offset = int(i / 4) * 0.4
        # datapoints
        offset = (3, 3) if i < len(nds) else (-12, -12)
        plt.annotate(
            text=str(i + 1),
            xy=(nd[1][1], nd[1][0]),
            xytext=offset,
            textcoords="offset points",
            annotation_clip=False,
        )
        # mol image
        image_arr = np.array(mol_img)
        imagebox = OffsetImage(image_arr, zoom=0.28)
        ab = AnnotationBbox(
            imagebox,
            xy=(nd[1][1], nd[1][0]),
            xybox=(1.16 + x_offset, 0.95 - (i % 4 * 0.3)),
        )
        ax.add_artist(ab)
        # numbering
        plt.annotate(
            text=str(i + 1) + ".",
            xy=(nd[1][1], nd[1][0]),
            xytext=(1.04 + x_offset, 0.95 - (i % 4 * 0.3)),
            annotation_clip=False,
        )
        # smiles
        plt.annotate(
            text=split_string_lines(mol_smiles, 12),
            xy=(nd[1][1], nd[1][0]),
            xytext=(1.26 + x_offset, 0.95 + 0.08 - (i % 4 * 0.3)),
            annotation_clip=False,
            va="top",
        )

    return fig_to_im((55, 15))


def multi_pareto_plot(results: list) -> io.BytesIO:
    """
    Function to plot pareto plot for multiple algorithm results using two objectives as specified.
    Highlight pareto front given two objectives.

    Parameters:
        results: Pymoo multi algorithm result.

    Returns:
        io.ByterIO: Plot in memory buffer.
    """
    _, ax = plt.subplots()

    for i, res in enumerate(results):
        nds = non_dominated(res.F[:, :2])
        # plot population and non-dominated set
        pareto = np.array([res.F[i] for i in nds])
        pop = np.array([idv for i, idv in enumerate(res.F) if i not in nds])
        ax.scatter(
            pop[:, 1],
            pop[:, 0],
            color=transparent_colors(0.5)[i],
            label=algorithm_names(res.name),
            marker=markers[i],
        )
        ax.scatter(
            pareto[:, 1],
            pareto[:, 0],
            color=transparent_colors(0.8)[i],
            edgecolor=(0, 0, 0, 0.5),
            # label=res.name + " pareto",
            s=sizes[i],
            marker=markers[i],
        )

    ax.legend(loc="lower left")
    # ax.set_ylim(0, 1.0)
    # ax.set_xlim(0, 1.0)
    ax.set_xlabel("SA score")
    ax.set_ylabel("QED", rotation=0)
    ax.yaxis.set_label_coords(-0.075, 0.5)

    return fig_to_im((20, 15))
