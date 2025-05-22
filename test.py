from pyvis.network import Network
import networkx as nx

G = nx.Graph()
G.add_edge("A", "B")
G.add_edge("A", "C")

net = Network(notebook=False)
net.from_nx(G)
net.show("test_graph.html")
