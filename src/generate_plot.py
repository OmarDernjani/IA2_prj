from helper_fun import generate_plot
import pandas as pd

data = pd.read_csv('./data/raw/MathE.csv', encoding = 'latin-1', sep = ';')
generate_plot(data)