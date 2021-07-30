import streamlit as st
from bokeh.plotting import figure

def create_value_counts_widget(
    cluster, 
    labels, 
    maximum_number_of_nodes,
    counts,
    clustercolors):

    if int(cluster) != -1:
        mytitle = "Value counts in cluster {}".format(cluster)
    else:
        mytitle = "Value counts of unclustered data"

    barcolor = clustercolors[int(cluster)] if int(cluster) in clustercolors else 'darkblue'
    p_node = figure(y_range=labels, tools=('pan, wheel_zoom, reset, save'), 
        title=mytitle, plot_height=maximum_number_of_nodes*20)
    p_node.hbar(y=labels, right=counts, color=barcolor, height=0.618)
    p_node.title.text_color = barcolor

    st.bokeh_chart(p_node, use_container_width=True)