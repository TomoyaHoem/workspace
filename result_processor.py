import random
import numpy as np
import pandas as pd
from guacamole_tasks import Task
from result_writer import write_single_results
from result_printer import print_results
from result_plotter import (
    single_pc_plot,
    multi_pc_plot,
    single_running_plot,
    similarity_plot,
    single_pareto_plot,
    multi_pareto_plot,
)
from util import hypervolume


class ResultProcessor:
    """
    Class used to process experiment results before writing to Excel file.
    Processes one algorithm run at a time.

    Parameters:
        results: Pymoo algorithm results.
        task: Task object containing GuacaMol information.
        settings: Dictionary of algorithm settings.
        filename: Output excel filename.
    """

    def __init__(
        self,
        molecules: pd.DataFrame,
        results: list,
        task: Task,
        settings: list,
        filename: str,
    ) -> None:
        """
        self.data: list containing organized algorithm results as depicted below
        [ [NSGA-II Data], [NSGA-III Data], [MOEA/D Data] ]
        each inner list is organized as follows
        [ algorithm name, initial population (SELFIE), top N pareto members (SELFIE), [settings], number of pareto members, parallel plot, [running_plots], hypervolume ]

        compare_data: list containig ogranized result comparisons as depicted below (lists contain one entry for each algorithm)
        [ parallel coordinate plot (for all algorithms), [algorithm names], [objective comparison], [number of pareto members], [hypervolume] ]
        """
        # TODO: add pareto plot (using 2-3 objectives) to compare_data
        # store algorithm run results
        self.molecules = molecules
        self.results = results
        self.task = task
        self.settings = settings
        self.filename = filename
        self.data = []
        self.compare_data = []

    def __call__(self, store_print: str, r_count: int) -> None:
        """
        Entry method that calls data preparation and storing and/or printing.

        Parameters:
            store_print: Whether to call storing or printing method.
        """
        if store_print in ["-s", "-p", "-sp", "-ps"]:
            self.process_data(r_count)
        else:
            return

        if "s" in store_print:
            write_single_results(
                self.filename, self.data, self.compare_data, self.task.objectives
            )

        if "p" in store_print:
            print_results()

    def sample_initial_population(
        self, results: list, molecules: pd.DataFrame
    ) -> list[str]:
        """
        Sample initial population that will be displayed next to top N solutions.

        Parameters:
            res: pymoo result of one algorithm run
            molecules: DataFrame containing subset from which initial population was sampled from

        Returns:
            List: SELFIES strings of initial population
        """
        initial_population = [x.X[0] for x in results[0].history[0].pop]
        mol_sample = molecules.loc[molecules["SELFIES"].isin(initial_population)]
        size = len(mol_sample) if len(mol_sample) < 100 else 100
        initial = np.random.choice(mol_sample["SELFIES"], size=size, replace=False)
        init = [[x] for x in initial]
        return init

    def top_n_individuals(self, res: list) -> list[str]:
        """Helper function that returns 100 or, if less, all availalbe best individuals"""
        top = res.X.tolist()
        n = len(top) if len(top) < 100 else 100
        topN = random.sample(top, n)
        return topN

    def process_data(self, r_count: int) -> None:
        """
        Method that summarizes algorithm results, e.g. plots or tables, and
        stores them in a list for easy access.
        """
        # I. process each algorithm and fill data
        # sample initial population for all (same for all)
        initial_pop = self.sample_initial_population(self.results, self.molecules)
        for res in self.results:
            algorithm_data = []
            # algorithm name
            algorithm_data.append(res.name + " Data")
            # initial population
            algorithm_data.append(initial_pop)
            # top N individuals
            topN = self.top_n_individuals(res)
            algorithm_data.append(topN)
            # settings
            algorithm_data.append(self.settings)
            # pareto set cardinality
            algorithm_data.append(len(res.F))
            # parallel coordinates objective plot
            algorithm_data.append(single_pc_plot(res, self.task.objectives))
            # running plots
            algorithm_data.append(single_running_plot(res, self.settings["N_Gen"]))
            # internal similarity plot
            algorithm_data.append(similarity_plot(res))
            # pareto plot
            algorithm_data.append(single_pareto_plot(res, r_count))
            # append to data
            self.data.append(algorithm_data)

        # II. compare algorithms and fill compare_data
        if len(self.results) > 1:
            self.compare_data = self.process_compare_data()

    def hypervolume(self, res: list) -> float:
        # create ref_point depending on number of objectives
        ref_point = np.ones(len(res.F[0])) * 1.2
        # hypervolume indicator
        ind = HV(ref_point=ref_point)
        # calculate hypervolume
        hyv = np.round(ind(res.F), 4)
        # print(f"Hypervolume is: {hyv}")
        return hyv

    def objective_statistics(self, res: list) -> np.array:
        """Helper function to summarize objective statistics"""
        vals, stats = [], []
        # get objective values from columns
        for i in range(len(res.F[0])):
            vals.append([j[i] for j in res.F])
        # get min, max, and mean for objectives
        for v in vals:
            stats.append([np.min(v), np.max(v), np.mean(v)])
        return np.round(np.array(stats), 2)

    def process_compare_data(self) -> list:
        """
        Data processing for comparison of different algorithms.
        Creates parallel coordinate plot and objective table data.
        """
        comp_data = []
        # PC plot
        comp_data.append(multi_pc_plot(self.results, self.task.objectives))
        # Objectives + |pareto_set|
        algs, obj, pareto, hv = [], [], [], []
        for res in self.results:
            algs.append(res.name)
            obj.append(self.objective_statistics(res))
            pareto.append(len(res.F[:, 0]))
            hv.append(hypervolume(res))
        comp_data.append(algs)
        comp_data.append(obj)
        comp_data.append(pareto)
        comp_data.append(multi_pareto_plot(self.results))
        comp_data.append(hv)
        return comp_data
