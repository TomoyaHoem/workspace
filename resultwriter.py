import xlsxwriter
import io
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import selfies as sf
from rdkit import Chem
from rdkit.Chem import Draw

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

    formats = {
        "filler": filler_cell_format,
        "title": title_cell_format,
        "selfies": selfie_cell_format,
        "img": img_cell_format,
        "header": header_cell_format,
        "set": set_cell_format,
    }
    return formats


def pareto_plot(molecules: pd.DataFrame, res: list) -> Image:
    res.X[np.argsort(res.F[:, 0])]

    # maximize objectives
    for obj_vals in res.F:
        obj_vals[0] *= -1
        obj_vals[1] *= -1

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
        color=mol_sample["pareto"],
    )
    ax.set_xlabel("QED", fontweight="bold")
    ax.set_ylabel("LogP", fontweight="bold")
    ax.set_zlabel("SA", fontweight="bold")
    plt.title(f"MOP Result - {len(res.F[:, 0])} Pareto members")

    ax.view_init(elev=10.0, azim=-20.0)
    # plt.show()  # TODO for testing purposes, extract later to print method

    imgdata = io.BytesIO()
    fig.savefig(imgdata, format="JPEG")
    return imgdata


def running_plots(res: list):
    hist = res.history

    running = RunningMetricAnimation(
        delta_gen=10, n_plots=3, key_press=True, do_show=True
    )

    for algorithm in hist[:30]:
        running.update(algorithm)

    # plot with only full iterations
    fig, ax = plt.subplots()
    running.draw(running.data[-1:], ax)

    imgdata = io.BytesIO()
    fig.set_size_inches(12, 6.5)
    fig.savefig(imgdata, format="JPEG")

    return running.plots[-1], imgdata


def buffer_image(image: Image, format: str = "JPEG"):
    # Store image in buffer, so we don't have to write it to disk.
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer, image


def resize(image: Image, size: tuple[int, int], format="JPEG"):
    image = image.resize(size)
    return buffer_image(image, format)


class ResultWriter:
    def __init__(
        self, molecules: pd.DataFrame, results: list, sets: list, filename: str
    ) -> None:
        self.data = []
        self.filename = filename
        # result data setup
        for res in results:
            cur = []
            # store alg name
            cur.append(res.algorithm.__class__.__name__ + " Data")
            # store top n individuals
            topN = res.X[np.argsort(res.F[:, 0])][:10]
            cur.append(topN)
            # alg settings
            cur.append(sets)
            # pareto plot
            cur.append(pareto_plot(molecules, res))
            # running metric plots
            cur.append(running_plots(res))
            self.data.append(cur)

    def store_data(self):
        workbook = xlsxwriter.Workbook(self.filename)
        formats = get_format_dict(workbook)

        for data in self.data:
            worksheet = workbook.add_worksheet(data[0])

            # I. Title
            worksheet.merge_range("B1:F1", data[0].upper(), formats["title"])

            # II. Top members
            last = 0
            worksheet.merge_range("B2:F2", "TOP INDIVIDUALS", formats["header"])
            for row, mol in enumerate(data[1]):
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

            # III. Settings
            worksheet.merge_range("H2:K2", "SETTINGS", formats["header"])
            for row, s in enumerate(data[2]):
                row *= 2
                row += 2
                worksheet.merge_range(row, 7, row + 1, 8, s[0], formats["set"])
                worksheet.merge_range(row, 9, row + 1, 10, s[1], formats["set"])

            # IV. Pareto
            worksheet.merge_range("M2:V2", "PARETO FRONT", formats["header"])
            # Add pareto image
            d = {
                "x_scale": 200 / image.width,
                "y_scale": 200 / image.height,
                "object_position": 1,
            }
            worksheet.merge_range("M3:V21", ".", formats["img"])
            worksheet.insert_image("M3", "", {"image_data": data[3], **d})

            # V. Running Metric
            # a)
            worksheet.merge_range("H25:S25", "R-METRIC ALL", formats["header"])
            worksheet.merge_range("H26:S46", ".", formats["img"])
            # Add running all
            d = {
                "x_scale": 200 / image.width,
                "y_scale": 202 / image.height,
                "object_position": 1,
            }
            worksheet.insert_image("H26", "", {"image_data": data[4][0], **d})
            # b)
            worksheet.merge_range("H50:S50", "R-METRIC LAST", formats["header"])
            worksheet.merge_range("H51:S71", ".", formats["img"])
            # Add running last
            d = {
                "x_scale": 200 / image.width,
                "y_scale": 202 / image.height,
                "object_position": 1,
            }
            worksheet.insert_image("H51", "", {"image_data": data[4][1], **d})

            # VI. Filler
            last += 11
            worksheet.conditional_format(
                "A1:W" + str(last),
                {
                    "type": "blanks",
                    "format": formats["filler"],
                },
            )

        workbook.close()

    def print_data(self):
        print("print res")
