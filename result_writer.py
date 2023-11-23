import io
import os
import xlsxwriter
import selfies as sf
from PIL import Image
from typing import Any
from rdkit import Chem
from rdkit.Chem import Draw


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

    # pareto
    pareto_cell_format = workbook.add_format()
    pareto_cell_format.set_text_wrap()
    pareto_cell_format.set_align("center")
    pareto_cell_format.set_align("vcenter")

    formats = {
        "filler": filler_cell_format,
        "title": title_cell_format,
        "selfies": selfie_cell_format,
        "img": img_cell_format,
        "header": header_cell_format,
        "set": set_cell_format,
        "fit": fit_cell_format,
        "pareto": pareto_cell_format,
    }
    return formats


def insert_mol_image(mol: list) -> tuple[io.BytesIO, dict[str, Any]]:
    img = Draw.MolToImage(Chem.MolFromSmiles(sf.decoder(mol[0])))
    image_buffer, image = resize(img, (300, 300), format="JPEG")

    d = {
        "x_scale": 192 / image.width,
        "y_scale": 200 / image.height,
        "object_position": 1,
    }
    return image_buffer, d


def buffer_image(image: Image, format: str = "JPEG"):
    # Store image in buffer, so we don't have to write it to disk.
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer, image


def resize(image: Image, size: tuple[int, int], format="JPEG"):
    image = image.resize(size)
    return buffer_image(image, format)


def write_results(filename: str, data: list, comp_data: list, tasks: list) -> None:
    """
    Write experiment results to excel file.

    Parameters:
        filename: Excel filename.
        data: Single algorithm statistics.
        comp_data: Algorithm comparison statistics.
        tasks: Objectives as a string list.
    """
    path = os.path.join("ResultWriter", "GuacaMol", filename)
    workbook = xlsxwriter.Workbook(path)
    formats = get_format_dict(workbook)

    for alg_data in data:
        worksheet = workbook.add_worksheet(alg_data[0])

        # I. Title
        worksheet.merge_range("A1:E1", alg_data[0].upper(), formats["title"])

        # II. Initial vs Top members
        last = 0
        # Initial
        worksheet.merge_range("B2:F2", "INITIAL", formats["header"])
        for row, mol in enumerate(alg_data[1]):
            row *= 10
            row += 2
            last = row
            worksheet.merge_range(row, 1, row + 9, 2, mol[0], formats["selfies"])

            # Add images
            image_buffer, d = insert_mol_image(mol)

            worksheet.merge_range(row, 3, row + 9, 5, ".", formats["img"])
            worksheet.insert_image(row, 3, "", {"image_data": image_buffer, **d})
        # Top
        worksheet.merge_range("G2:K2", "TOP INDIVIDUALS", formats["header"])
        for row, mol in enumerate(alg_data[2]):
            row *= 10
            row += 2
            worksheet.merge_range(row, 6, row + 9, 7, mol[0], formats["selfies"])

            # Add images
            image_buffer, d = insert_mol_image(mol)

            worksheet.merge_range(row, 8, row + 9, 10, ".", formats["img"])
            worksheet.insert_image(row, 8, "", {"image_data": image_buffer, **d})

        # III. Settings
        worksheet.merge_range("M2:P2", "SETTINGS", formats["header"])
        for row, (key, value) in enumerate(alg_data[3].items()):
            row *= 2
            row += 2
            worksheet.merge_range(row, 12, row + 1, 13, key, formats["set"])
            worksheet.merge_range(row, 14, row + 1, 15, value, formats["set"])

        # IV. Parallel Coordinates
        worksheet.merge_range(
            "R2:AA2", "PARETO SET - " + str(alg_data[4]), formats["header"]
        )
        # Add PC plot
        worksheet.merge_range("R3:AA21", ".", formats["img"])
        worksheet.insert_image("R3", "", {"image_data": alg_data[5]})

        # V. Running Metric
        # a)
        worksheet.merge_range("M25:X25", "R-METRIC ALL", formats["header"])
        worksheet.merge_range("M26:X46", ".", formats["img"])
        # Add running all
        worksheet.insert_image("M26", "", {"image_data": alg_data[6][0]})
        # b)
        worksheet.merge_range("M50:X50", "R-METRIC LAST", formats["header"])
        worksheet.merge_range("M51:X71", ".", formats["img"])
        # Add running last
        worksheet.insert_image("M51", "", {"image_data": alg_data[6][1]})

        # VI. Filler
        last += 11
        worksheet.conditional_format(
            "A1:AB" + str(last),
            {
                "type": "blanks",
                "format": formats["filler"],
            },
        )

    # VII. Comparison
    if comp_data:
        worksheet = workbook.add_worksheet("Comparison")
        # 1. Multi PC plot
        worksheet.insert_image("B2", "", {"image_data": comp_data[0]})

        # 2. Pareto Values and Number
        # text
        worksheet.merge_range("M2:V2", "FITNESS", formats["header"])
        worksheet.merge_range("M3:M4", "Objectives", formats["fit"])
        for i, a in enumerate(comp_data[1]):
            worksheet.merge_range(2, 13 + (i * 3), 2, 15 + (i * 3), a, formats["fit"])
            worksheet.write(3, 13 + (i * 3), "MIN", formats["fit"])
            worksheet.write(3, 14 + (i * 3), "MAX", formats["fit"])
            worksheet.write(3, 15 + (i * 3), "AVG", formats["fit"])
        for i, t in enumerate(tasks):
            worksheet.write(4 + i, 12, t)
        # vals
        for i, b in enumerate(comp_data[2]):
            for j, v in enumerate(b):
                for k, n in enumerate(v):
                    worksheet.write(4 + j, 13 + (i * 3) + k, n)
        # pareto
        worksheet.merge_range("M12:M13", "#Pareto", formats["fit"])
        for i, p in enumerate(comp_data[3]):
            worksheet.merge_range(
                11, 13 + (i * 3), 12, 15 + (i * 3), p, formats["pareto"]
            )

        worksheet.conditional_format(
            "A1:Y44",
            {
                "type": "blanks",
                "format": formats["filler"],
            },
        )

    workbook.close()
