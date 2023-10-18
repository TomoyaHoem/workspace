import os
import xlsxwriter
import numpy as np

from algdata import AlgData
from running_metric_ret import RunningMetricAnimation


# Create and store formats in dictionary
def get_format_dict(workbook: xlsxwriter.workbook) -> dict:
    # filler box
    filler_cell_format = workbook.add_format()
    filler_cell_format.set_bg_color("gray")

    # title
    title_cell_format = workbook.add_format()
    title_cell_format.set_bold()
    title_cell_format.set_bg_color("gray")
    title_cell_format.set_align("center")
    title_cell_format.set_align("vcenter")
    title_cell_format.set_font_size(16)

    # img
    img_cell_format = workbook.add_format()
    img_cell_format.set_border(2)
    img_cell_format.set_align("center")
    img_cell_format.set_align("vcenter")

    # header
    header_cell_format = workbook.add_format()
    header_cell_format.set_bold()
    header_cell_format.set_align("center")
    header_cell_format.set_font_size(14)
    header_cell_format.set_bottom(2)

    # fit
    fit_cell_format = workbook.add_format()
    fit_cell_format.set_bold()
    fit_cell_format.set_italic()
    fit_cell_format.set_align("center")
    fit_cell_format.set_align("vcenter")

    formats = {
        "filler": filler_cell_format,
        "title": title_cell_format,
        "img": img_cell_format,
        "header": header_cell_format,
        "fit": fit_cell_format,
    }
    return formats


class AverageWriter:
    def __init__(self, algs) -> None:
        self.data = {}
        for alg in algs:
            self.data[alg] = AlgData(alg_name=alg)

    def append_results(self, results: list) -> None:
        delta, num_p = self.r_plot_data(len(results[0].history))
        for res in results:
            currData = self.data[res.algorithm.__class__.__name__.lower()]
            # pareto
            currData.pareto = len(res.X[np.argsort(res.F[:, 0])].tolist())
            # objectives
            self.append_objectives(res, currData)
            # running metrics
            self.append_running_data(delta, num_p, res, currData)

    def append_running_data(self, delta, num_p, res, currData):
        running = RunningMetricAnimation(
            delta_gen=delta, n_plots=num_p, key_press=False, do_show=False
        )
        for algorithm in res.history[: len(res.history)]:
            running.update(algorithm)
        currData.running_data = running.data[-1:]

    def r_plot_data(self, num_iter):
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

    def append_objectives(self, res: list, currData: AlgData) -> None:
        qed = [i[0] for i in res.F]
        logp = [i[1] for i in res.F]
        sa = [i[2] for i in res.F]

        currData.qed = [[np.min(qed)], [np.max(qed)], [np.mean(qed)]]
        currData.logp = [[np.min(logp)], [np.max(logp)], [np.mean(logp)]]
        currData.sa = [[np.min(sa)], [np.max(sa)], [np.mean(sa)]]

    def running_plot(self, data: list):
        pass

    def store_averages(self, filename, n) -> None:
        path = os.path.join("ResultWriter", filename)
        # for k, v in self.data.items():
        #     print(v.pareto)
        #     print(v.qed)
        #     print(v.running_data)
        workbook = xlsxwriter.Workbook(path)
        formats = get_format_dict(workbook)
        worksheet = workbook.add_worksheet("AVG_DATASHEET")
        # I. Objectives
        # Title
        worksheet.merge_range("A1:E1", f"AVERAGE DATA OF {n} RUNS", formats["title"])
        # Objectives Headers

        # Objective Values
        for key, value in self.data.items():
            pass
        # II. Pareto
        # Sideheader
        # Number
        # III. Runnning

        # filler

        workbook.close()
