import xlsxwriter


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


def write_results() -> None:
    pass
