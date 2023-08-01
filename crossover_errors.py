import numpy as np
from matplotlib import pyplot as plt


def count_errors(errs: list) -> list:
    keys = [
        "syntax error",
        "parentheses",
        "Unkekulized atoms",
        "unclosed ring",
        "duplicate",
        "valence",
        "aromatic",
    ]
    counts = [0, 0, 0, 0, 0, 0, 0]

    for line in errs:
        for i, key in enumerate(keys):
            if key in line:
                counts[i] += 1

    return counts


def main() -> None:
    infile = r"error.log"

    with open(infile) as errs:
        errs = errs.readlines()

    counts = count_errors(errs=errs)

    keys = [
        "Syntax error",
        "Extra parentheses",
        "Unkekulized atoms",
        "Unclosed ring",
        "Duplicate ring",
        "Valence greater",
        "Non-ring aromatic",
    ]

    print(sum(counts))
    print(10000 - 740)

    plt.bar(keys, counts)
    plt.show()


if __name__ == "__main__":
    main()
