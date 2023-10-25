import io
import os
import itertools
import xlsxwriter
import numpy as np
from PIL import Image

from algdata import AlgData
import matplotlib.pyplot as plt
from running_metric_ret import RunningMetricAnimation
from pymoo.indicators.igd import IGD
from pymoo.util.normalization import normalize

colors = ["#FF0000", "#00FF0A", "#383FFF"]


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

    # pareto
    pareto_cell_format = workbook.add_format()
    pareto_cell_format.set_text_wrap()
    pareto_cell_format.set_align("center")
    pareto_cell_format.set_align("vcenter")

    formats = {
        "filler": filler_cell_format,
        "title": title_cell_format,
        "img": img_cell_format,
        "header": header_cell_format,
        "fit": fit_cell_format,
        "pareto": pareto_cell_format,
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
            # pareto_len
            currData.pareto_len = len(res.F.tolist())
            # pareto
            currData.pareto = res.F.tolist()
            # objectives
            self.append_objectives(res, currData)
            # running metrics
            self.append_running_data(delta, num_p, res, currData)
            # history
            currData.histories = res.history

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

    def running_plot(self) -> Image:
        fig, ax = plt.subplots()
        fig.set_size_inches(12, 6.5)
        for i, (key, value) in enumerate(self.data.items()):
            # avg
            tau, x, f, v = value.running_data["AVG"]
            ax.plot(x, f, label=tau, alpha=0.9, linewidth=3, color=colors[i])
            # min & max
            tau, x, f1, v = value.running_data["MIN"]
            ax.plot(x, f1, alpha=0.2, linestyle="dotted", linewidth=1, color=colors[i])
            tau, x, f2, v = value.running_data["MAX"]
            ax.plot(x, f2, alpha=0.2, linestyle="dashed", linewidth=1, color=colors[i])
            # fill
            plt.fill_between(x, f1, f2, color=colors[i], alpha=0.1)

            for k in range(len(v)):
                if v[k]:
                    ax.plot(
                        [k + 1, k + 1],
                        [0, f[k]],
                        color="black",
                        linewidth=0.5,
                        alpha=0.5,
                    )
                    ax.plot(
                        [k + 1], [f[k]], "o", color="black", alpha=0.5, markersize=2
                    )
        ax.legend()

        ax.set_xlabel("Generation")
        ax.set_ylabel("$\Delta \, f$", rotation=0)
        ax.yaxis.set_label_coords(-0.075, 0.5)
        imgdata = io.BytesIO()
        fig.savefig(imgdata, format="JPEG")
        return imgdata

    def dominates(self, a: list, b: list) -> bool:
        dominate = False

        # check QED
        if b[0] > a[0]:
            return False
        if a[0] > b[0]:
            dominate = True

        # check LogP
        if b[1] > a[1]:
            return False
        if a[1] > b[1]:
            dominate = True

        # check SA
        if b[2] < a[2]:
            return False
        if a[2] < b[2]:
            dominate = True

        return dominate

    def running_comparison_plot(self) -> Image:
        data = list(self.data.values())
        igd_vals = [[], [], []]

        for i in range(len(data[0].pareto)):
            # merge pareto sets and remove duplicates
            merged = list(
                itertools.chain(data[0].pareto[i], data[1].pareto[i], data[2].pareto[i])
            )
            merged.sort()
            merged = list(m for m, _ in itertools.groupby(merged))
            # remove non-dominated
            for a, b in itertools.combinations(merged, 2):
                # print(f"check if {a} dominates {b}")
                if self.dominates(a, b):
                    # print("true")
                    merged.remove(b)
            # reverse objectives for IGD calc
            merged = [
                [i * -1 if i in inner[:2] else i for i in inner] for inner in merged
            ]
            merged = np.array(merged)
            # get ideal and nadir
            c_F, c_ideal, c_nadir = merged, merged.min(axis=0), merged.max(axis=0)
            # normalize
            # normalize the current objective space values
            c_N = normalize(c_F, c_ideal, c_nadir)
            # normalize all previous generations with respect to current ideal and nadir for each alg
            for j in range(len(data)):
                N = [
                    normalize(p.opt.get("F"), c_ideal, c_nadir)
                    for p in data[j].histories[i]
                ]
                # calculate IGD
                delta_f = [IGD(c_N).do(N[k]) for k in range(len(N))]
                # append
                igd_vals[j].append(delta_f)

        # plot min, max, avg of igd_vals
        fig, ax = plt.subplots()
        fig.set_size_inches(12, 6.5)
        gens = [g for g in range(1, len(data[0].histories[0]) + 1, 1)]
        for i in range(len(igd_vals)):
            # avg
            ax.plot(
                gens,
                np.mean(igd_vals[i], axis=0),
                label=data[i].alg_name,
                alpha=0.9,
                linewidth=3,
                color=colors[i],
            )
            # min & max
            f1 = np.min(igd_vals[i], axis=0)
            f2 = np.max(igd_vals[i], axis=0)
            ax.plot(
                gens, f1, alpha=0.2, linestyle="dotted", linewidth=1, color=colors[i]
            )
            ax.plot(
                gens, f2, alpha=0.2, linestyle="dashed", linewidth=1, color=colors[i]
            )
            # fill
            plt.fill_between(gens, f1, f2, color=colors[i], alpha=0.1)

        ax.legend()

        ax.set_xlabel("Generation")
        ax.set_ylabel("$\\varnothing_{k,t}$", rotation=0)
        ax.yaxis.set_label_coords(-0.075, 0.5)
        imgdata = io.BytesIO()
        fig.savefig(imgdata, format="JPEG")
        return imgdata

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
        worksheet.merge_range("B1:K1", f"AVERAGE DATA OF {n} RUNS", formats["title"])
        # Objectives Headers
        worksheet.merge_range("B3:K3", "FITNESS", formats["header"])
        worksheet.merge_range("B4:B5", "Objectives", formats["fit"])
        worksheet.write("B6", "QED")
        worksheet.write("B7", "LogP")
        worksheet.write("B8", "SA")
        # Objective Values
        for i, (key, value) in enumerate(self.data.items()):
            worksheet.merge_range(
                3, 2 + (i * 3), 3, 4 + (i * 3), key.upper(), formats["fit"]
            )
            worksheet.write(4, 2 + (i * 3), "MIN", formats["fit"])
            worksheet.write(4, 3 + (i * 3), "MAX", formats["fit"])
            worksheet.write(4, 4 + (i * 3), "AVG", formats["fit"])
            for j, (q, l, s) in enumerate(zip(value.qed, value.logp, value.sa)):
                worksheet.write(5, 2 + (i * 3) + j, q)
                worksheet.write(6, 2 + (i * 3) + j, l)
                worksheet.write(7, 2 + (i * 3) + j, s)
        # II. Pareto
        # Sideheader
        worksheet.merge_range("B9:B10", "#Pareto", formats["fit"])
        # Number
        for i, (key, value) in enumerate(self.data.items()):
            worksheet.merge_range(
                8, 2 + (i * 3), 9, 4 + (i * 3), value.pareto_len, formats["pareto"]
            )
        # III. Runnning
        worksheet.merge_range("B13:M13", "R-METRIC ALL", formats["header"])
        worksheet.merge_range("B14:M14", ".", formats["img"])
        # Add running all
        d = {
            "x_scale": 200 / 300,
            "y_scale": 200 / 300,
            "object_position": 1,
        }
        worksheet.insert_image("B14", "", {"image_data": self.running_plot(), **d})
        # IV. Runnning Comp
        worksheet.merge_range("B37:M37", "R-METRIC ALL COMPARISON", formats["header"])
        worksheet.merge_range("B38:M38", ".", formats["img"])
        # Add running all
        d = {
            "x_scale": 200 / 300,
            "y_scale": 200 / 300,
            "object_position": 1,
        }
        worksheet.insert_image(
            "B38", "", {"image_data": self.running_comparison_plot(), **d}
        )
        # filler
        worksheet.conditional_format(
            "A1:N60",
            {
                "type": "blanks",
                "format": formats["filler"],
            },
        )

        workbook.close()
