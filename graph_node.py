from graphviz import Source

# Load the .dot file
graph = Source.from_file('genome_graph.dot')

# Render it as an image
graph.render(format='png', filename='genome_graph', cleanup=True)  # Saves genome_graph.png
