# this code will be taking an image and extracting best colors out of it

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

def get_palette(
        n_colors, 
        palette_name='category20'
    ):

    try:
        palette = sns.color_palette(palette_name)
    except:
        print('Palette not found. Using default palette tab10')
        palette = sns.color_palette('category20')
    while len(palette) < n_colors:
        palette += palette
    
    return palette