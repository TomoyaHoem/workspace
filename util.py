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
        if i == 1:
            # SA
            if b[i] < a[i]:
                return False
            if a[i] < b[i]:
                dominate = True
        else:
            # other objectives
            if b[i] > a[i]:
                return False
            if a[i] > b[i]:
                dominate = True

    return dominate
