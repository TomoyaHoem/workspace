import xlsxwriter
import io
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import selfies as sf
from rdkit import Chem
from rdkit.Chem import Draw

from running_metric_ret import RunningMetricAnimation

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

    # selfie
    selfie_cell_format = workbook.add_format()
    selfie_cell_format.set_text_wrap()
    selfie_cell_format.set_align("top")

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

    # setting
    set_cell_format = workbook.add_format()
    set_cell_format.set_text_wrap()
    set_cell_format.set_align("left")

    # fit
    fit_cell_format = workbook.add_format()
    fit_cell_format.set_bold()
    fit_cell_format.set_italic()
    fit_cell_format.set_align("center")
    fit_cell_format.set_align("vcenter")

    formats = {
        "filler": filler_cell_format,
        "title": title_cell_format,
        "selfies": selfie_cell_format,
        "img": img_cell_format,
        "header": header_cell_format,
        "set": set_cell_format,
        "fit": fit_cell_format,
    }
    return formats


def pareto_plot(molecules: pd.DataFrame, res: list, plot=False) -> Image:
    res.X[np.argsort(res.F[:, 0])]

    initial_population = [x.X[0] for x in res.history[0].pop]
    mol_sample = molecules.loc[molecules["SELFIES"].isin(initial_population)]

    # add results to dataframe
    for mol, obj in zip(res.X, res.F):
        new_row = pd.DataFrame(
            {
                "Dir": "",
                "File": "",
                "Mol": "",
                "Smiles": "",
                "SELFIES": mol,
                "QED": obj[0],
                "LogP": obj[1],
                "SA": obj[2],
                "pareto": "#FF0022FF",
            }
        )
        mol_sample = pd.concat([mol_sample, new_row], axis=0, ignore_index=True)

    # Creating figure
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection="3d")

    # Creating plot
    ax.scatter(
        mol_sample["QED"],
        mol_sample["LogP"],
        mol_sample["SA"],
        color=mol_sample["pareto"].tolist(),
    )
    ax.set_xlabel("QED", fontweight="bold")
    ax.set_ylabel("LogP", fontweight="bold")
    ax.set_zlabel("SA", fontweight="bold")
    plt.title(f"MOP Result - {len(res.F[:, 0])} Pareto members")

    ax.view_init(elev=10.0, azim=-20.0)
    if plot:
        plt.show()

    imgdata = io.BytesIO()
    fig.savefig(imgdata, format="JPEG")

    return imgdata


def running_plots(res: list, num_iter):
    hist = res.history

    delta, num_p = r_plot_data(num_iter)

    running = RunningMetricAnimation(
        delta_gen=delta, n_plots=num_p, key_press=False, do_show=False
    )

    for algorithm in hist[:num_iter]:
        running.update(algorithm)

    # plot with only full iterations
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 6.5)
    running.draw(running.data, ax)
    imgdata_f = io.BytesIO()
    fig.savefig(imgdata_f, format="JPEG")
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 6.5)
    running.draw(running.data[-1:], ax)
    imgdata_l = io.BytesIO()
    fig.savefig(imgdata_l, format="JPEG")

    return imgdata_f, imgdata_l


def buffer_image(image: Image, format: str = "JPEG"):
    # Store image in buffer, so we don't have to write it to disk.
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer, image


def resize(image: Image, size: tuple[int, int], format="JPEG"):
    image = image.resize(size)
    return buffer_image(image, format)


def compare_data(molecules: pd.DataFrame, results: list, num_iter, plot=False):
    # I. Pareto

    initial_population = [x.X[0] for x in results[0].history[0].pop]
    mol_sample = molecules.loc[molecules["SELFIES"].isin(initial_population)]
    mol_sample["alg"] = "Initial Pop"

    for res, col in zip(results, colors):
        res.X[np.argsort(res.F[:, 0])]
        # add results to dataframe
        for mol, obj in zip(res.X, res.F):
            new_row = pd.DataFrame(
                {
                    "Dir": "",
                    "File": "",
                    "Mol": "",
                    "Smiles": "",
                    "SELFIES": mol,
                    "QED": obj[0],
                    "LogP": obj[1],
                    "SA": obj[2],
                    "pareto": col,
                    "alg": res.algorithm.__class__.__name__,
                }
            )
            mol_sample = pd.concat([mol_sample, new_row], axis=0, ignore_index=True)

    # Creating figure
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection="3d")

    # Creating plot
    for alg, d in mol_sample.groupby("alg"):
        ax.scatter(d["QED"], d["LogP"], d["SA"], color=d["pareto"].tolist(), label=alg)
    ax.set_xlabel("QED", fontweight="bold")
    ax.set_ylabel("LogP", fontweight="bold")
    ax.set_zlabel("SA", fontweight="bold")
    plt.legend()
    plt.title(f"MOP Result for different Algorithms")

    ax.view_init(elev=14.0, azim=55.0)

    if plot:
        plt.show()

    imgdata_p_1 = io.BytesIO()
    fig.savefig(imgdata_p_1, format="JPEG")

    ax.view_init(elev=10.0, azim=-20.0)
    imgdata_p_2 = io.BytesIO()
    fig.savefig(imgdata_p_2, format="JPEG")

    # II. Running
    delta, num_p = r_plot_data(num_iter)
    r_data = []
    r_alg = []
    for res in results:
        running = RunningMetricAnimation(
            delta_gen=delta, n_plots=num_p, key_press=False, do_show=False
        )
        for algorithm in res.history[:num_iter]:
            running.update(algorithm)
        r_data.append(running.data[-1:])
        r_alg.append(res.algorithm.__class__.__name__)

    fig, ax = plt.subplots()
    fig.set_size_inches(12, 6.5)
    for d, alg in zip(r_data, r_alg):
        temp = list(d[0])
        temp[0] = alg
        d = [tuple(temp)]
        running.draw_comp(d, ax)
    imgdata_r = io.BytesIO()
    fig.savefig(imgdata_r, format="JPEG")

    return [imgdata_p_1, imgdata_p_2, imgdata_r]


def r_plot_data(num_iter):
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


def calc_fitness_vals(fit: list):
    qed = [i[0] for i in fit]
    logp = [i[1] for i in fit]
    sa = [i[2] for i in fit]
    res = np.array(
        [
            [np.min(qed), np.max(qed), np.mean(qed)],
            [np.min(logp), np.max(logp), np.mean(logp)],
            [np.min(sa), np.max(sa), np.mean(sa)],
        ]
    )
    return np.round(res, 2)


class ResultWriter:
    def __init__(
        self, molecules: pd.DataFrame, results: list, sets: list, filename: str
    ) -> None:
        self.data = []
        self.comp = []
        self.filename = filename
        initial = self.initial_pop_sample(molecules, results)
        # result data setup
        for res, s in zip(results, sets):
            cur = []
            # store alg name
            cur.append(res.algorithm.__class__.__name__ + " Data")
            # store initial and top n individuals
            top = res.X[np.argsort(res.F[:, 0])].tolist()
            n = len(top) if len(top) < 100 else 100
            topN = random.sample(top, n)
            cur.append([initial, topN])
            # alg settings
            cur.append(s)
            # pareto plot
            cur.append(pareto_plot(molecules, res))
            # running metric plots
            cur.append(running_plots(res, s[3][1]))
            self.data.append(cur)
        if len(results) > 1:
            self.comp = compare_data(molecules, results, s[3][1])
            self.comp.append([])
            self.comp.append([])
            for res in results:
                self.comp[3].append(res.algorithm.__class__.__name__)
                self.comp[4].append(calc_fitness_vals(res.F))

    def initial_pop_sample(self, molecules, res):
        initial_population = [x.X[0] for x in res[0].history[0].pop]
        mol_sample = molecules.loc[molecules["SELFIES"].isin(initial_population)]
        size = len(mol_sample) if len(mol_sample) < 100 else 100
        initial = np.random.choice(mol_sample["SELFIES"], size=size, replace=False)
        init = [[x] for x in initial]
        return init

    def store_data(self):
        path = os.path.join("ResultWriter", self.filename)
        workbook = xlsxwriter.Workbook(path)
        formats = get_format_dict(workbook)

        for data in self.data:
            worksheet = workbook.add_worksheet(data[0])

            # I. Title
            worksheet.merge_range("A1:E1", data[0].upper(), formats["title"])

            # II. Initial vs Top members
            last = 0
            # Initial
            worksheet.merge_range("B2:F2", "INITIAL", formats["header"])
            for row, mol in enumerate(data[1][0]):
                row *= 10
                row += 2
                last = row
                worksheet.merge_range(row, 1, row + 9, 2, mol[0], formats["selfies"])

                # Add images
                img = Draw.MolToImage(Chem.MolFromSmiles(sf.decoder(mol[0])))
                image_buffer, image = resize(img, (300, 300), format="JPEG")

                d = {
                    "x_scale": 192 / image.width,
                    "y_scale": 200 / image.height,
                    "object_position": 1,
                }
                worksheet.merge_range(row, 3, row + 9, 5, ".", formats["img"])
                worksheet.insert_image(row, 3, "", {"image_data": image_buffer, **d})
            # Top
            worksheet.merge_range("G2:K2", "TOP INDIVIDUALS", formats["header"])
            for row, mol in enumerate(data[1][1]):
                row *= 10
                row += 2
                worksheet.merge_range(row, 6, row + 9, 7, mol[0], formats["selfies"])

                # Add images
                img = Draw.MolToImage(Chem.MolFromSmiles(sf.decoder(mol[0])))
                image_buffer, image = resize(img, (300, 300), format="JPEG")

                d = {
                    "x_scale": 192 / image.width,
                    "y_scale": 200 / image.height,
                    "object_position": 1,
                }
                worksheet.merge_range(row, 8, row + 9, 10, ".", formats["img"])
                worksheet.insert_image(row, 8, "", {"image_data": image_buffer, **d})

            # III. Settings
            worksheet.merge_range("M2:P2", "SETTINGS", formats["header"])
            for row, s in enumerate(data[2]):
                row *= 2
                row += 2
                worksheet.merge_range(row, 12, row + 1, 13, s[0], formats["set"])
                worksheet.merge_range(row, 14, row + 1, 15, s[1], formats["set"])

            # IV. Pareto
            worksheet.merge_range("R2:AA2", "PARETO FRONT", formats["header"])
            # Add pareto image
            d = {
                "x_scale": 200 / image.width,
                "y_scale": 200 / image.height,
                "object_position": 1,
            }
            worksheet.merge_range("R3:AA21", ".", formats["img"])
            worksheet.insert_image("R3", "", {"image_data": data[3], **d})

            # V. Running Metric
            # a)
            worksheet.merge_range("M25:X25", "R-METRIC ALL", formats["header"])
            worksheet.merge_range("M26:X46", ".", formats["img"])
            # Add running all
            d = {
                "x_scale": 200 / image.width,
                "y_scale": 202 / image.height,
                "object_position": 1,
            }
            worksheet.insert_image("M26", "", {"image_data": data[4][0], **d})
            # b)
            worksheet.merge_range("M50:X50", "R-METRIC LAST", formats["header"])
            worksheet.merge_range("M51:X71", ".", formats["img"])
            # Add running last
            d = {
                "x_scale": 200 / image.width,
                "y_scale": 202 / image.height,
                "object_position": 1,
            }
            worksheet.insert_image("M51", "", {"image_data": data[4][1], **d})

            # VI. Filler
            last += 11
            worksheet.conditional_format(
                "A1:AB" + str(last),
                {
                    "type": "blanks",
                    "format": formats["filler"],
                },
            )

        if self.comp:
            worksheet = workbook.add_worksheet("Comparison")
            # I. Pareto comparison a)
            d = {
                "x_scale": 200 / image.width,
                "y_scale": 202 / image.height,
                "object_position": 1,
            }
            worksheet.insert_image("B2", "", {"image_data": self.comp[0], **d})
            # I. Pareto comparison b)
            d = {
                "x_scale": 200 / image.width,
                "y_scale": 202 / image.height,
                "object_position": 1,
            }
            worksheet.insert_image("B23", "", {"image_data": self.comp[1], **d})

            # II. Running Metric comparison
            d = {
                "x_scale": 200 / image.width,
                "y_scale": 202 / image.height,
                "object_position": 1,
            }
            worksheet.insert_image("M21", "", {"image_data": self.comp[2], **d})
            worksheet.conditional_format(
                "A1:Y44",
                {
                    "type": "blanks",
                    "format": formats["filler"],
                },
            )

            # III. Pareto Values
            worksheet.merge_range("M2:S2", "FITNESS", formats["header"])
            worksheet.merge_range("M3:M4", "Objectives", formats["fit"])
            worksheet.write("M5", "QED")
            worksheet.write("M6", "LogP")
            worksheet.write("M7", "SA")
            for i in range(2):
                worksheet.merge_range(
                    2, 13 + (i * 3), 2, 15 + (i * 3), self.comp[3][i], formats["fit"]
                )
                worksheet.write(3, 13 + (i * 3), "MIN", formats["fit"])
                worksheet.write(3, 14 + (i * 3), "MAX", formats["fit"])
                worksheet.write(3, 15 + (i * 3), "AVG", formats["fit"])
                for j in range(3):
                    worksheet.write(4, 13 + (i * 3) + j, self.comp[4][i][0][j])
                    worksheet.write(5, 13 + (i * 3) + j, self.comp[4][i][1][j])
                    worksheet.write(6, 13 + (i * 3) + j, self.comp[4][i][2][j])

        workbook.close()

    def print_data(self, molecules: pd.DataFrame, results: list, num_iter):
        for res in results:
            pareto_plot(molecules, res, plot=True)
        if len(results) > 1:
            compare_data(molecules, results, num_iter, plot=True)
