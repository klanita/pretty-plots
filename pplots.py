import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from adjustText import adjust_text
from artpalettes import *

import matplotlib
matplotlib.use('Agg')

# default parameters for matlab
FONT_SIZE = 7
font = {'family': 'Arial', 'size': FONT_SIZE}
matplotlib.rc('font', **font)
matplotlib.rc('ytick', labelsize=FONT_SIZE)
matplotlib.rc('xtick', labelsize=FONT_SIZE)

from sklearn.metrics.pairwise import cosine_similarity

def save_to_file(fig, file_name, file_format):
    if file_format is None:
        if file_name.split(".")[-1] in ['png', 'pdf']:
            file_format = file_name.split(".")[-1]
            savename = file_name                
        else:
            file_format = 'pdf'
            savename = f'{file_name}.{file_format}'
        
        fig.savefig(savename, format=file_format)
        print(f"Saved file to: {savename}")


def plot_embedding(
        emb,
        labels=None,
        col_dict=None,
        title=None,
        show_lines=False,
        show_text=False,
        show_legend=True,
        axis_equal=True,
        circle_size=30,
        circe_transparency=1.0,
        line_transparency=0.8,
        line_width=0.8,
        fontsize=9,
        fig_width=4,
        fig_height=4,
        file_name=None,
        file_format=None,
        labels_name=None,
        width_ratios=[7, 1],
        bbox=(1.3, 0.7)
    ):
    
    # create data structure suitable for embedding
    df = pd.DataFrame(emb, columns=['dim1', 'dim2'])
    if not (labels is None):
        if labels_name is None:
            labels_name = 'labels'
        df[labels_name] = labels

    
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(1, 2, width_ratios=width_ratios)
    ax = plt.subplot(gs[0])
    
    sns.despine(left=False, bottom=False, right=True)
    
    # if col_dict is None:
    #     unique_labels = np.unique(labels)
    #     col_dict = dict(zip(unique_labels, ['#00000']*len(unique_labels)))
    
    if (col_dict is None) and not (labels is None):
        col_dict = get_colors(labels)

    
    sns.scatterplot(
        x="dim1", 
        y="dim2", 
        hue=labels_name, 
#       hue_order=labels_order,
        palette=col_dict,
        # palette='Set1',
        alpha=circe_transparency,                    
        edgecolor="none",
        s=circle_size,
        data=df, 
        ax=ax)
    
    # I remove seaborn default legend so it won't clash with the rest of the plot
    try:
        ax.legend_.remove()
    except:
        pass
    
    if show_lines:
        for i in range(len(emb)):
            if col_dict is None:
                ax.plot(
                    [0, emb[i, 0]],
                    [0, emb[i, 1]],
                    alpha=line_transparency,
                    linewidth=line_width,
                    c=None
                )
            else:
                ax.plot(
                    [0, emb[i, 0]],
                    [0, emb[i, 1]],
                    alpha=line_transparency,
                    linewidth=line_width,
                    c=col_dict[labels[i]]
                )

    if show_text:
        texts = []
        for i in range(len(emb)):
            texts.append(
                ax.text(
                    emb[i, 0], 
                    emb[i, 1], 
                    labels[i], 
                    fontsize=fontsize
                )
            )
    
        adjust_text(
            texts,
            arrowprops=dict(arrowstyle='-', color='black', lw=0.1),
            ax=ax
        )

    if axis_equal:
        ax.axis('equal')
        ax.axis('square')

    if title:
        ax.set_title(title, fontsize=fontsize, fontweight="bold")

    ax.set_xlabel('dim1', fontsize=fontsize)
    ax.set_ylabel('dim2', fontsize=fontsize)
    plt.tight_layout()

    if file_name:
        save_to_file(fig, file_name, file_format)        

    return plt


def get_colors(labels, palette=None, palette_name=None):
    n_colors = len(labels)
    if palette is None:
        palette = get_palette(n_colors, palette_name)    
    col_dict = dict(zip(labels, palette[:n_colors]))
    return col_dict


def plot_similarity(
        emb,
        labels=None,
        col_dict=None,
        fig_width=4,
        fig_height=4,
        cmap='coolwarm',
        fmt='png',
        fontsize=7,
        file_format=None,
        file_name=None
    ):

    # first we take construct similarity matrix
    # add another similarity
    similarity_matrix = cosine_similarity(emb)

    df = pd.DataFrame(
        similarity_matrix,
        columns=labels,
        index=labels,
    )

    if col_dict is None:
        col_dict = get_colors(labels)

    network_colors = pd.Series(df.columns, index=df.columns).map(col_dict)

    sns_plot = sns.clustermap(
        df,
        cmap=cmap, 
        center=0, 
        row_colors=network_colors,
        col_colors=network_colors,
        mask=False,
        metric='euclidean',
        figsize=(fig_height, fig_width),
        vmin=-1, vmax=1,
        fmt=file_format
    )
    
    sns_plot.ax_heatmap.xaxis.set_tick_params(labelsize=fontsize)
    sns_plot.ax_heatmap.yaxis.set_tick_params(labelsize=fontsize)
    sns_plot.ax_heatmap.axis('equal')
    sns_plot.cax.yaxis.set_tick_params(labelsize=fontsize)

    if file_name:
        save_to_file(sns_plot, file_name, file_format)

