import gc


def r_plot_data(num_iter):
    """Helper function to dynamically adjust running metric plots to number of generations."""
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


def dominates(a: list, b: list) -> bool:
    """Helper function to check whether a dominates b"""
    dominate = False

    for i in range(len(a)):
        # print(f"check if {a[i]} dominates {b[i]}")
        if b[i] > a[i]:
            return False
        if a[i] > b[i]:
            dominate = True

    return dominate


def dump_garbage():
    """
    show us what the garbage is about
    """
    # Force collection
    print("\nGARBAGE:")
    gc_count = gc.collect()
    print(gc_count)

    print("\nGARBAGE OBJECTS:")
    for x in gc.garbage:
        s = str(x)
        if len(s) > 80:
            s = s[:77] + "..."
        print(type(x), "\n  ", s)


def normalize_sa(fitness: list) -> list:
    for inner in fitness:
        inner[1] = 1 - ((inner[1] - 1) / (10 - 1))
    return fitness


def split_string_lines(string: str, num_char: int):
    """Split string into multiple lines, each line being num_char characters long."""
    split_string = [string[i : i + num_char] for i in range(0, len(string), num_char)]
    return "\n".join(split_string)


def non_dominated(fitness: list) -> list:
    """Returns indices of non-dominated individuals for 2 obj."""
    idx = list(range(len(fitness)))
    remove = set()
    for i, inner in enumerate(fitness):
        for j, other in enumerate(fitness):
            if i != j:
                if dominates(other[:2], inner[:2]):
                    remove.add(i)

    for index in sorted(remove, reverse=True):
        del idx[index]

    return idx


def transparent_colors(alpha: float):
    colors_colb = [(216, 27, 96), (0, 77, 64), (30, 136, 229)]
    return [tuple(c / 255 for c in color) + (alpha,) for color in colors_colb]
