import xlsxwriter
import numpy as np
import matplotlib.pyplot as plt

import selfies as sf
from rdkit import Chem
from rdkit.Chem import Draw


class ResultWriter:
    def __init__(self, results: list, filename: str) -> None:
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
            self.data.append(cur)

    def store_data(self):
        workbook = xlsxwriter.Workbook(self.filename)

        for data in self.data:
            worksheet = workbook.add_worksheet(data[0])
            for i, mol in enumerate(data[1]):
                worksheet.write(i, 0, mol[0])
                img = Draw.MolToImage(Chem.MolFromSmiles(sf.decoder(mol[0])))
                img.save("mol.png")
                worksheet.insert_image(i, 1, "mol.png")

        workbook.close()

    def print_data(self):
        print("print res")

    # results = res.X[np.argsort(res.F[:, 0])]
    # # print(np.column_stack(results))

    # # maximize objectives
    # for obj_vals in res.F:
    #     obj_vals[0] *= -1
    #     obj_vals[1] *= -1

    # mol_sample = molecules.sample(n=2500, random_state=1)

    # # add results to dataframe
    # for mol, obj in zip(res.X, res.F):
    #     new_row = pd.DataFrame(
    #         {
    #             "Dir": "",
    #             "File": "",
    #             "Mol": "",
    #             "Smiles": "",
    #             "SELFIES": mol,
    #             "QED": obj[0],
    #             "LogP": obj[1],
    #             "SA": obj[2],
    #             "pareto": "#FF0022FF",
    #         }
    #     )
    #     mol_sample = pd.concat([mol_sample, new_row], axis=0, ignore_index=True)

    # # plot data
    # # Creating figure
    # fig = plt.figure(figsize=(10, 6))
    # ax = plt.axes(projection="3d")

    # # Creating plot
    # ax.scatter(
    #     mol_sample["QED"],
    #     mol_sample["LogP"],
    #     mol_sample["SA"],
    #     color=mol_sample["pareto"],
    # )
    # ax.set_xlabel("QED", fontweight="bold")
    # ax.set_ylabel("LogP", fontweight="bold")
    # ax.set_zlabel("SA", fontweight="bold")
    # plt.title("MOP Result")

    # # show plot
    # plt.show()

    # # Scatter().add(res.F).show()

    # # Evaluation using Running Metric

    # hist = res.history

    # running = RunningMetricAnimation(
    #     delta_gen=10, n_plots=10, key_press=True, do_show=True
    # )

    # for algorithm in hist[:100]:
    #     running.update(algorithm)
