import networkx as nx
import matplotlib.pyplot as plt
from graph.graph_data import graph

plt.ion()

# Cria o grafo e adiciona as arestas
G = nx.Graph()
for node, edges in graph.items():
    for adjacent, weight in edges:
        G.add_edge(node, adjacent, weight=weight)

""" y_mapping_reverse = {
    'A': 0, 'B': 0, 'C': 0, 'D': 0,
    'E': 2715, 'F': 2715, 'G': 2715, 'H': 2715, 'I': 2715,
    'J': 4840, 'K': 4840, 'L': 4840, 'M': 4840,
    'N': 6340, 'O': 6340, 'P': 6340, 'Q': 6340, 'R': 6340, 'S': 6340, 'T': 6340,
    'U': 7840, 'V': 7840, 'X': 7840, 'Y': 7840,
    'Z': 9965, 'W': 9965, 'AA': 9965, 'AB': 9965, 'AC': 9965,
    'AD': 12680, 'AE': 12680, 'AF': 12680, 'AG': 12680,
    'ST1': 9965,
    'ST2': 2715,
} """

y_mapping = {
    'A': 12680, 'B': 12680, 'C': 12680, 'D': 12680,
    'E': 9965, 'F': 9965, 'G': 9965, 'H': 9965, 'I': 9965, 'ST2': 9965,
    'J': 7840, 'K': 7840, 'L': 7840, 'M': 7840,
    'N': 6340, 'O': 6340, 'P': 6340, 'Q': 6340, 'R': 6340, 'S': 6340, 'T': 6340,
    'U': 4840, 'V': 4840, 'X': 4840, 'Y': 4840,
    'Z': 2715, 'W': 2715, 'AA': 2715, 'AB': 2715, 'AC': 2715, 'ST1': 2715,
    'AD': 0, 'AE': 0, 'AF': 0, 'AG': 0}

x_mapping = {
    'A': 0, 'E': 0, 'N': 0, 'U': 0,
    'B': 1500, 'F': 1500,
    'O': 2600, 'V': 2600,
    'C': 3000, 'G': 3000,
    'D': 4500, 'H': 4500,
    'P': 6925 - 2600, 'X': 6925 - 2600,  # 4325
    'I': 6925, 'J': 6925, 'Q': 6925, 'Y': 6925, 'Z': 6925,
    'K': 6925 + 2600, 'R': 6925 + 2600,       # 9525
    'L': 6925 - 2600 + 6925, 'S': 6925 - 2600 + 6925,  # 11250
    'M': 6925 + 6925, 'T': 6925 + 6925, 'AC': 6925 + 6925, 'AG': 6925 + 6925,  # 13850
    'AB': 6925 + 6925 - 1500, 'AF': 6925 + 6925 - 1500,  # 12350
    'AA': 6925 + 6925 - 1500 - 1500, 'AE': 6925 + 6925 - 1500 - 1500,  # 10850
    'W': 6925 + 6925 - 1500 - 1500 - 1500, 'AD': 6925 + 6925 - 1500 - 1500 - 1500,  # 9350
    'ST1': 0,
    'ST2': 13850
}

# Converter para inteiros (opcional)
y_mapping = {node: int(value) for node, value in y_mapping.items()}
x_mapping = {node: int(value) for node, value in x_mapping.items()}

# Criação do dicionário de posições para os nós existentes
pos = {}
default_x = 6925  # valor default para nós sem mapeamento horizontal
for node in G.nodes():
    x = x_mapping.get(node, default_x)
    y = y_mapping.get(node, 0)
    pos[node] = (x, y)

# ---------------------------
# 6. Desenhar o grafo atualizado
# ---------------------------
plt.figure(figsize=(13, 10))

# Desenha todas as arestas
nx.draw_networkx_edges(G, pos)

# Desenha os nós e os rótulos
nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
nx.draw_networkx_labels(G, pos, font_size=10)

# (Opcional) Rótulos das arestas
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

plt.title("Grafo Atualizado com Nós de Mudança de Direção (ST1 e ST2)")
plt.axis('off')
plt.show()

fh = open("edgelist.txt", "wb")
nx.write_edgelist(G, fh)


def render_grafo(current_node, carrying=False):
    plt.clf()  # limpa o painel atual, sem fechar a janela

    # Desenha arestas
    nx.draw_networkx_edges(G, pos)

    # Cores dos nós
    node_colors = []
    for node in G.nodes():
        if node == current_node:
            node_colors.append('red' if carrying else 'orange')
        else:
            node_colors.append('lightblue')

    nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors)
    nx.draw_networkx_labels(G, pos, font_size=10)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title(f"Posição atual: {current_node} | {'Com caixa' if carrying else 'Sem caixa'}")
    plt.axis('off')
    plt.pause(0.5)

