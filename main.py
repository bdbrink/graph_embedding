# /usr/bin/env python3

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from karateclub.node_embedding.neighbourhood.deepwalk import DeepWalk

G = nx.random_tree(40)

deepwalk = DeepWalk(dimensions=3)
deepwalk.fit(G)

embedding = deepwalk.get_embedding()
print(embedding)

figure = plt.figure(figsize=(15, 8))
axis = figure.add_subplot(projection="3d")

# add all dimensions to graph
plt.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2])

# generate graph in matplot
nx.draw_spring(G)
plt.show()