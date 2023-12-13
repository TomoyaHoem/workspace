import io
import numpy as np
from typing import Any
from guacamole_tasks import Task
from result_writer import write_multi_results
from result_printer import print_results
from result_plotter import multi_similiarty_plot, multi_running_plot


class AverageProceesor:
    """
    Class used to process experiment results before writing to Excel file.
    Processes N algorithm runs at a time to produce averages.
    Its functionality is devided into two main parts, first the results
    are preprocessed and appended in a data container.
    Secondly, the preprocessed results are organized once more
    to be then exported to an excel file.

    Parameters:
        algs: Algorithm names.
        task: Task object containing GuacaMol information.
    """

    def __init__(self, algs: list[str], guac: str) -> None:
        """
        self.res_container: Dictionary that contains appended data of N runs as shown below:
        { "NSGA-II": [results], "NSGA-III": [results], "MOEA/D": [results] }
        each of the [results] lists stores the fitness and SELFIES representation of each individual obtained as depicted below
        [ [res (run 1), res (run 2), ..., res (run N)], [history (run 1), history (run 2), ..., history (run N)] ]

        self.data: List that contains the processed and organized results to be written to the excel:
        [ [[NSGA-II Data], [NSGA-III Data], [MOEA/D Data]], running plot, internal similarity plot ]
        each inner list is organized as follows
        [ algorithm name, [Objective values (Min, Max, Average)], number of pareto members ]
        """
        self.task = Task(guac)
        self.res_container = {}
        for alg in algs:
            self.res_container[alg] = [[], []]
        self.data = []

    def __call__(self, store_print: str, filename: str, repeat: int) -> None:
        """
        Entry method that calls data preparation and storing and/or printing.

        Parameters:
            store_print: Whether to call storing or printing method.
            filename: Output excel filename.
        """
        if store_print in ["-s", "-p", "-sp", "-ps"]:
            self.process_results()
        else:
            return

        if "s" in store_print:
            write_multi_results(filename, self.data, self.task.objectives, repeat)

        if "p" in store_print:
            print_results()

    def append_results(self, results: list) -> None:
        """
        Method to preorganize and append result data.

        Parameters:
            results: Pymoo result object for each algorithm run.
        """
        for res in results:
            curList = self.res_container[res.name.lower()]
            # append results
            curList[0].append(res)
            # append histories
            curList[1].append(res.history)

    def objective_statistics(self, rez: list) -> np.array:
        """
        Helper function to summarize objective statistics
        into min, max, average, plus respective standard deviation.
        """
        all_stats = []
        for res in rez:
            vals, stats = [], []
            # get objective values from columns
            for i in range(len(res.F[0])):
                vals.append([j[i] for j in res.F])
            # get min, max, and mean for objectives
            for v in vals:
                stats.append([np.min(v), np.max(v), np.mean(v)])
            # append
            all_stats.append(stats)
        # accumulate min, max, average values of different iterations
        acc = []
        for i in range(len(all_stats[0])):
            acc.append([list(x) for x in zip(*[x[i] for x in all_stats])])
        # calculate means and stds
        all = [
            [
                (str(np.round(np.mean(y), 2)), " \u00B1 " + str(np.round(np.std(y), 2)))
                for y in x
            ]
            for x in acc
        ]
        return all

    def pareto_statistics(self, rez: list) -> list:
        """Function to calculate average number of pareto members and standard deviation"""
        pareto_count = []
        for res in rez:
            pareto_count.append(len(res.F[:, 0]))
        return (
            str(np.round(np.mean(pareto_count))),
            " \u00B1 " + str(np.round(np.std(pareto_count))),
        )

    def process_results(self) -> None:
        alg_data_all = []
        for key, value in self.res_container.items():
            algorithm_data = []
            # I. Alg name
            algorithm_data.append(key)
            # II. Objectives + |Pareto|
            algorithm_data.append(self.objective_statistics(value[0]))
            algorithm_data.append(self.pareto_statistics(value[0]))
            alg_data_all.append(algorithm_data)
        self.data.append(alg_data_all)
        # III. Running Plot
        self.data.append(multi_running_plot(self.res_container))
        # VI. Similarity Plot
        self.data.append(multi_similiarty_plot(self.res_container))
