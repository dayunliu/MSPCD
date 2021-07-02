import pandas as pd


def megre():
    path_prefix = "../data/circFunbase/"
    file = []
    for i in range(1,14):
        path = path_prefix + f"circRNA_sim_{i}.csv"
        f1 = pd.read_csv(path)
        file.append(f1)
    circRNA_sim = pd.concat(file)
    circRNA_sim.to_csv("../data/circFunbase/circRNA_sim1.csv", index=0, sep=',')


if __name__ == "__main__":
    megre()
