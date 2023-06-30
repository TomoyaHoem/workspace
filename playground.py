def main() -> None:
    a = [1, 2, 3, 4]
    b = [5, 6, 7, 8]

    c = [a, b]
    end = False
    currentList = a
    index = 1

    while not end:
        for i, x in enumerate(currentList):
            print(f"i: {i}, x: {x}")
            if i == len(a) - 1:
                if index >= len(c):
                    print("end")
                    end = True
                    break
                currentList = c[index]
                index += 1


if __name__ == "__main__":
    main()
