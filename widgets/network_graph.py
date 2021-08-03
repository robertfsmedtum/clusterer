import streamlit as st
import networkx as nx
from bokeh.plotting import figure, from_networkx
from bokeh.models import Range1d, Circle, MultiLine, NodesAndLinkedEdges, ColumnDataSource, Label

def create_network_graph_widget(
    G,
    nodeweights,
    edgeweights,
    cluster,
    clustercolors,
):

    for nodekey in nodeweights:
        G.add_node(nodekey, size=nodeweights[nodekey])
    for edgekey in edgeweights:
        if edgekey[0] in nodeweights and edgekey[1] in nodeweights:
            G.add_edge(edgekey[0], edgekey[1], weight=edgeweights[edgekey])

    degrees = dict(nx.degree(G))
    nx.set_node_attributes(G, name='degree', values=degrees)

    number_to_adjust_by = 5 # 5

    # adjusted_node_size = dict([(node, degree+number_to_adjust_by) for node, degree in nx.degree(G)])
    adjusted_node_size = dict([(node, d['size']+number_to_adjust_by) for node, d in G.nodes.data()])

    for _,_,d in G.edges(data=True):
        d['weight'] = d['weight'] * 0.1

    min_node_size = min(adjusted_node_size.values())
    max_node_size = max(adjusted_node_size.values())
    target_min_size = 8 # 8
    target_max_size = 14 # 14

    if max_node_size > min_node_size:
        adjusted_node_size = {k:target_min_size + ((v-min_node_size)/(max_node_size-min_node_size))*target_max_size for k, v in adjusted_node_size.items()}

    nx.set_node_attributes(G, name='adjusted_node_size', values=adjusted_node_size)

    size_by_this_attribute = 'adjusted_node_size'

    node_highlight_color = 'white'
    edge_highlight_color = 'black'

    if int(cluster) != -1:
        title = 'Network graph of cluster {}'.format(cluster)
    else:
        title = 'Network graph of unclustered data'

    HOVER_TOOLTIPS = [
        ("Value", "@index"),
        ("Degree", "@degree")
    ]

    plot = figure(tooltips = HOVER_TOOLTIPS, sizing_mode = 'scale_height',
                tools="pan,wheel_zoom,save,reset", active_scroll='wheel_zoom',
                x_range=Range1d(-10.1, 10.1), y_range=Range1d(-10.1, 10.1), title=title) # , plot_width=600, plot_height=600)

    if int(cluster) in clustercolors:
        plot.title.text_color = clustercolors[int(cluster)]


    #Create a network graph object
    # https://networkx.github.io/documentation/networkx-1.9/reference/generated/networkx.drawing.layout.spring_layout.html
    network_graph = from_networkx(G, nx.spring_layout, scale=8, center=(0, 0))

    #Set node sizes and colors according to node degree (color as category from attribute) clustercolors[int(cluster)]
    if int(cluster) in clustercolors:
        network_graph.node_renderer.glyph = Circle(size=size_by_this_attribute, fill_color=clustercolors[int(cluster)], line_width=0)

    #Set node highlight colors
    network_graph.node_renderer.hover_glyph = Circle(size=size_by_this_attribute, fill_color=node_highlight_color, line_width=2)
    network_graph.node_renderer.selection_glyph = Circle(size=size_by_this_attribute, fill_color=node_highlight_color, line_width=2)

    #Set edge opacity and width
    network_graph.edge_renderer.glyph = MultiLine(line_alpha=0.05, line_width="weight")
    #Set edge highlight colors
    network_graph.edge_renderer.selection_glyph = MultiLine(line_color=edge_highlight_color, line_width=2)
    network_graph.edge_renderer.hover_glyph = MultiLine(line_color=edge_highlight_color, line_width=2)

    #Highlight nodes and edges
    network_graph.selection_policy = NodesAndLinkedEdges()
    network_graph.inspection_policy = NodesAndLinkedEdges()

    plot.renderers.append(network_graph)

    #Add Labels
    x, y = zip(*network_graph.layout_provider.graph_layout.values())
    node_labels = list(G.nodes())

    node_labels_size = [int(x[1]["adjusted_node_size"]) for x in list(G.nodes.data())]
    max_node_label_size = max(node_labels_size)
    min_node_label_size = min(node_labels_size)

    node_labels_size = [str(x) + 'px' for x in node_labels_size]
    if max_node_label_size > min_node_label_size:
        node_labels_alpha = [0.5 + ((int(x[1]["adjusted_node_size"])-min_node_label_size)/(max_node_label_size-min_node_label_size))*0.5 for x in list(G.nodes.data())]
    else:
        node_labels_alpha = [0.8 for x in list(G.nodes.data())]        
    source = ColumnDataSource({'x': x, 'y': y, 'name': [node_labels[i] for i in range(len(x))], 'fontsize': [node_labels_size[i] for i in range(len(x))], 'alpha': [node_labels_alpha[i] for i in range(len(x))]})

    graph_labels = []
    for x, y, name, fontsize, alpha in zip(source.data['x'], source.data['y'], source.data['name'], source.data['fontsize'], source.data['alpha']):
        graph_labels.append(Label(x=x, y=y, text=name, text_alpha=alpha, text_align ='center', text_font_size=fontsize, background_fill_color='white', background_fill_alpha=.7))
        plot.add_layout(graph_labels[-1])

    st.bokeh_chart(plot, use_container_width=True)