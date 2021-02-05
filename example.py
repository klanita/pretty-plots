import numpy as np
# import pplots
from pplots import *

if __name__ == "__main__":
    n = 5
    emb = np.random.normal(0, 1, [n, 2])
    labels = ['point'+str(i) for i in range(n)]

    plot_embedding(
        emb, 
        labels,
        show_lines=True,
        show_text=True,
        title='My random example',
        file_name='example.png'
        )

    plot_similarity(
        emb, 
        labels,
        # title='My random example',
        file_name='example_similarity.png'
        )
