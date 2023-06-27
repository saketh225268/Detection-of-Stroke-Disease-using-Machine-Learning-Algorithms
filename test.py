import pandas as pd
import numpy as np

dataset = pd.read_csv("Dataset/dataset.csv")
unique, count = np.unique(dataset["stroke"],return_counts=True)
print(unique)
print(count)

