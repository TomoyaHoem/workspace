import pandas as pd
import numpy as np


def main() -> None:
    df = pd.DataFrame(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=["a", "b", "c"]
    )

    df_drop = df.drop(index=[0, 1])

    print(df)
    print(df_drop)


if __name__ == "__main__":
    main()

    # 0 3 - 0 6 - 3 6
    # 1 4 - 1 7 - 4 7
    # 2 5 - 2 8 - 5 8

    # for i in range(3):
    #     for j in range(3):
    #         if j > 1:
    #             k = 3 + i
    #             l = 6 + i
    #         else:
    #             k = i
    #             l = 3 + i + 3 * j
    #         print(k, l)

    # a = [1, 2, 3, 4]
    # b = [5, 6, 7, 8]

    # c = [a, b]
    # end = False
    # currentList = a
    # index = 1

    # while not end:
    #     for i, x in enumerate(currentList):
    #         print(f"i: {i}, x: {x}")
    #         if i == len(a) - 1:
    #             if index >= len(c):
    #                 print("end")
    #                 end = True
    #                 break
    #             currentList = c[index]
    #             index += 1
