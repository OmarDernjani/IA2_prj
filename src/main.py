import pandas as pd
from EDA_helper_fun import generate_plot

if __name__ == '__main__':
    dataset = pd.read_csv('C:/Users/dernj/Desktop/IA2_progetto/data/MathE.csv', sep = ';', encoding = 'latin-1')
    print(dataset.head())
    generate_plot(dataset)



