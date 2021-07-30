import streamlit as st
import matplotlib.pyplot as plt

def create_dendrogram_widget(clusterer, clustercolors):
    def create_fig(clusterer, clustercolors):
        fig, ax = plt.subplots()
        clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=list(clustercolors.values())[1:], axis=ax)            
        plt.title('Cluster hierarchy dendrogram')     
    st.pyplot(create_fig(clusterer, clustercolors))