## TODO Add probabilities
## TODO Better network graph
## TODO Nodes to export to networkx

from numpy.ma.core import minimum


def main():
    import os
    import pandas as pd
    import numpy as np
    import base64
    import hdbscan
    import copy
    import scipy
    import streamlit as st
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics.pairwise import pairwise_distances
    from bokeh.transform import linear_cmap
    from bokeh.plotting import figure, from_networkx
    from bokeh.models import HoverTool, ColumnDataSource, Label, NodesAndLinkedEdges, MultiLine, Title, Circle, ColorBar, BasicTicker, LinearColorMapper, Range1d
    from bokeh.palettes import Turbo256
    from collections import Counter
    import umap
    import itertools
    import networkx as nx
    import matplotlib.pyplot as plt
    from networkx.algorithms import community
    from wordcloud import WordCloud

    results_max_n = 3
    results_text = ''

    default_number_of_neighbors = 20 # 15 # 20 # 5
    default_minimum_samples = 5 # 5 # 10 # 5
    default_minimum_cluster_size = 20 # 20 # 30 # 50
    default_selection_epsilon = 0.0
    standard_scalar_default = 0 # 1 = Yes, 0 = No
    pairwisedistance_default = 0 # 1 = Yes, 0 = No
    default_maximum_number_of_nodes = 20

    remaining_dimensions = 2
    minimum_distance = 0.0

    random_seed = 42

    np.random.seed()

    if 'started' not in st.session_state:
        st.session_state.started = False

    DEBUG_OPTIONS = {
        "DEBUG": True,
        "input": "./data.csv",
        "save_graph": True,
        # "options": {
        #     'data': [1]
        # }
    }

    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.set_page_config(
        page_title="UMAP and HDBSCAN for network analysis",
        page_icon="🕸",
        layout="centered",
        initial_sidebar_state="auto",
    )

    @st.cache
    def convert_dataframe(df):
        df = df.dropna()
        conversion_dicts = {}
        for col in df.columns:
            if df[col].dtype != 'Int64' and df[col].dtype != 'float64':
                previous_keydict = {}
                for c in conversion_dicts:
                    for k in conversion_dicts[c]:
                        previous_keydict[k] = conversion_dicts[c][k]
                try:
                    df[col] = df[col].astype('Int64')
                except:
                    try:
                        df[col] = df[col].astype(float)
                    except:
                        try:
                            my_different_values = sorted(list(set(df[col])))
                            mycatdict = { str(v):int(i) for i, v in enumerate(my_different_values)}
                        except:
                            mycatdict = { str(v):int(i) for i, v in enumerate(list(set(df[col])))}
                        for k in previous_keydict:
                            if k in mycatdict:
                                mycatdict[k] = int(previous_keydict[k])
                        conversion_dicts[col] = mycatdict
                        df[col] = df[col].replace(mycatdict).astype('Int64')
        cdf = pd.DataFrame(conversion_dicts)
        return df, cdf

    @st.cache
    def mean_confidence_interval(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
        return m, h

    @st.cache
    def download_link(object_to_download, download_filename, download_link_text):
        if isinstance(object_to_download,pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False)
        b64 = base64.b64encode(object_to_download.encode()).decode()
        return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

    st.write("""
    # 🕸 UHC - Network analysis v1.0
    **for higher dimensional data using UMAP¹ and HDBSCAN²**
    """)

    uploaded_file = st.sidebar.file_uploader("📂 Select a file (csv or excel)")

    if (uploaded_file is None and not DEBUG_OPTIONS["DEBUG"]) and 'df_raw' not in st.session_state:
        st.write("""
            ## **Instructions**
            With the UMAP-HDBSCAN-Clusterer (UHC) you can choose a csv or excel file containing columns of binary,
            categorical or numerical data. The analysis starts automatically after a file was chosen.\n
            *Note that this is a local application which works offline when using with a localhost adress.*
            *The provided datasets do **not** get uploaded and you can stay offline after the application first loaded.*\n
            When you use the streamlit hosted version, see https://streamlit.io/privacy-policy for details: \n
            *"We also want to assure you that the Streamlit open-source software does not — and never will — see or store any of the data you put into any app that you develop with it. That data belongs to you and only you."*

        """)
    else:
        if DEBUG_OPTIONS["DEBUG"]:
            df_raw = pd.read_csv(DEBUG_OPTIONS["input"])
        else:
            if 'df_raw' in st.session_state:
                df_raw = st.session_state.df_raw
            else:
                fileending = uploaded_file.name.split('.')[-1]
                if fileending == 'csv':
                    df_raw = pd.read_csv(uploaded_file)
                elif fileending == 'xlsx' or fileending == 'xls':
                    df_raw = pd.read_excel(uploaded_file, engine='openpyxl')
                else:
                    st.error('Please select a valid csv, xls or xlsx file.')
                    st.stop()
                if 'df_raw' not in st.session_state:
                    st.session_state.df_raw = df_raw

        df_raw, dict_df = convert_dataframe(df_raw)

        df_cols = list(df_raw.columns)
        col1, col2, col3, col4 = st.beta_columns([1, 1, 1, 2])

        col1.write('__Data__')
        col2.write('__Dimensions__')
        col3.write('__Rows__')
        col4.write('__Row value mean__')

        col1.write('**Original**')
        col2.write('_{:,}_'.format(len(df_cols)))
        col3.write('_{:,}_'.format(len(df_raw)))
        original_mean_ci = mean_confidence_interval(df_raw.T.sum())
        col4.write('_{:.2f} +/- {:.2f} [95% CI]_'.format(original_mean_ci[0], original_mean_ci[1]))

        with st.sidebar.beta_expander("Filtering data (optional)"):
            options_col_to_exclude = st.multiselect(
                'Excluding selected columns.',
                df_cols,
                )

            options_col_to_analyze = st.multiselect(
                'Select column(s) to separate data',
                df_cols,
                )

            options = {}
            if len(options_col_to_analyze) > 0:
                for selected_col in options_col_to_analyze:
                    groups_to_analyze = list(set(list(df_raw[selected_col])))

                    options[selected_col] = st.multiselect(
                        'Analyze selected values of {}'.format(selected_col),
                        groups_to_analyze,
                        default=groups_to_analyze
                        )

            if DEBUG_OPTIONS["DEBUG"]:
                if 'options' in DEBUG_OPTIONS:
                    options = DEBUG_OPTIONS["options"]


        with st.sidebar.beta_expander("Show additional settings"):
            metric = st.selectbox(
                "What metric do you want to use in UMAP?",
                (
                    "manhattan",
                    "euclidean",
                    "chebyshev",
                    "canberra",
                    "braycurtis",
                    "haversine",
                    "hamming",
                    "jaccard",
                    "dice",
                    "russellrao",
                    "kulsinski",
                    "rogerstanimoto",
                    "sokalmichener",
                    "sokalsneath",
                    "yule"
                ))

            number_of_neighbors = st.slider('Minimum number of neighbors (UMAP)', 2, 100, default_number_of_neighbors, step=1)
            minimum_samples = st.slider('Minimum number of samples (HDBSCAN)', 1, 100, default_minimum_samples, step=1)
            minimum_cluster_size = st.slider('Minimum cluster size (HDBSCAN)', 2, 100, default_minimum_cluster_size, step=1)
            cluster_selection_epsilon = st.slider('Cluster selection minimum (HDBSCAN)', 0.0, 2.0, default_selection_epsilon, step=0.1, format=f"%1f")
            standardscalar = st.radio(
                'Use standardscalar transformation',
                ['Yes', 'No'],
                1-standard_scalar_default
            )
            pairwisedistance = st.radio(
                'Use pairwise distance',
                ['Yes', 'No'],
                1-pairwisedistance_default
            )

            color_this_col = st.selectbox(
                'Select column to color',
                ['None'] + df_cols,
            )

        if st.sidebar.button('▶️ Start analysis') or DEBUG_OPTIONS["DEBUG"] or st.session_state.started:
            new_values = st.sidebar.multiselect(
                'Select values for new prediction.',
                df_cols
            )

            st.session_state['started'] = True
            df = df_raw.copy()

            if len(options) > 0:
                for key in options:
                    df = df[df[key].isin(options[key])].reset_index(drop=True)
                    df = df.drop(key, axis=1)

            for col in options_col_to_exclude:
                if col in df:
                    df = df.drop(col, axis=1)

            if len(df) == 0:
                st.warning('Empty dataframe. Please change parameters or upload another dataset.')
                st.stop()

            col1.write('**Filtered**')
            col2.write('_{:,}_'.format(len(df.columns)))
            col3.write('_{:,}_'.format(len(df)))
            filtered_mean_ci = mean_confidence_interval(df.T.sum())
            col4.write('_{:.2f} +/- {:.2f} [95% CI]_'.format(filtered_mean_ci[0], filtered_mean_ci[1]))

            scaled_df = df
            if standardscalar == 'Yes':
                scaled_df = StandardScaler().fit_transform(scaled_df)
            if pairwisedistance == 'Yes':
                scaled_df = pairwise_distances(scaled_df)

            @st.cache
            def calculate_umap(n_o_n, input_df, minimum_distance):
                return umap.UMAP(
                    n_neighbors=n_o_n,
                    min_dist=minimum_distance,
                    n_components=remaining_dimensions,
                    random_state=random_seed
                    ).fit(input_df)

            trans = calculate_umap(number_of_neighbors, scaled_df, minimum_distance)
            standard_embedding = trans.transform(scaled_df)

            @st.cache
            def calculate_hdbscan(input_metric, input_min_samples, input_min_cluster_size, input_standard_embedding, cluster_selection_epsilon):
                return hdbscan.HDBSCAN(
                    metric=input_metric,
                    min_samples=input_min_samples, 
                    min_cluster_size=input_min_cluster_size,
                    cluster_selection_epsilon=cluster_selection_epsilon,
                    prediction_data=True
                    ).fit(input_standard_embedding)
            
            new_input_metric = 'precomputed' if pairwisedistance == 'Yes' else metric
            clusterer = copy.deepcopy(calculate_hdbscan(new_input_metric, minimum_samples, minimum_cluster_size, standard_embedding, cluster_selection_epsilon))
            hdbscan_labels = clusterer.labels_

            clustered = (hdbscan_labels >= 0)

            if True not in clustered:
                st.warning('Did not find any clusters. Please adjust the parameters or choose a different metric.')

            values = list(df.columns)

            @st.cache
            def return_value_list(x):
                values_to_return = []
                for value in values:
                    if x[value] == 1:
                        values_to_return.append(value)
                return values_to_return  
                    
            cluster_df = pd.DataFrame(standard_embedding, columns=('x', 'y'))
            cluster_df['cluster'] = [str(x) for x in hdbscan_labels]
            if color_this_col != 'None':
                cluster_df['color by ' + color_this_col] = scaled_df[color_this_col]
                different_labels = list(set(scaled_df[color_this_col]))
            else:
                different_labels = list(set(hdbscan_labels))
            cluster_df['values'] = df.apply(lambda x: return_value_list(x), axis=1)
            datasource = ColumnDataSource(cluster_df)

            plot_figure = figure(
                tools=('pan, wheel_zoom, reset, save')
            )

            plot_figure.add_tools(HoverTool(tooltips="""
            <div>
                <div>
                    <span style='font-size: 16px; color: #224499'>Cluster:</span>
                    <span style='font-size: 18px'>@cluster</span><br>
                    <span style='font-size: 14px'>@values</span>
                </div>
            </div>
            """))

            if color_this_col != 'None':
                mapper = linear_cmap(field_name='color by ' + color_this_col, palette=Turbo256, low=min(different_labels), high=max(different_labels)+0.8)
            else:
                mapper = linear_cmap(field_name='cluster', palette=Turbo256, low=min(different_labels), high=max(different_labels))

            different_labels_sorted = sorted(different_labels)
            total_color_length = len(Turbo256)
            clustercolors = {}

            # st.pyplot(fig=axes, use_container_width=True)

            if len(different_labels_sorted) > 1:
                for i in range(len(different_labels_sorted)):
                    colorindex = i * int(total_color_length/(len(different_labels_sorted)-1))
                    if colorindex > 255:
                        colorindex = 255
                    clustercolors[int(different_labels_sorted[i])] = Turbo256[colorindex]
            else:
                clustercolors[-1] = Turbo256[0]

            plot_figure.circle(
                'x',
                'y',
                source=datasource,
                line_color=mapper,
                color=mapper,
                line_alpha=0.6,
                fill_alpha=0.6,
                size=4,
            )

            prediction_to_cluster = -2

            if len(new_values) > 0:
                testdf = pd.DataFrame(columns=scaled_df.columns)
                testdf.loc[0] = [1 if x in new_values else 0 for x in testdf.columns]
                test_embedding = trans.transform(testdf)
                new_predictions = hdbscan.approximate_predict(clusterer, test_embedding)
                cluster_df_new = pd.DataFrame(test_embedding, columns=('x', 'y'))
                cluster_df_new['cluster'] = pd.Series(new_predictions[0])
                prediction_to_cluster = new_predictions[0][0]
                cluster_df_new['values'] = testdf.apply(lambda x: return_value_list(x), axis=1)
                datasource_new = ColumnDataSource(cluster_df_new)
                plot_figure.circle(
                    'x',
                    'y',
                    source=datasource_new,
                    line_color='black',
                    color=mapper,
                    line_alpha=0.6,
                    fill_alpha=0.6,
                    size=40
                )

            if True in clustered:
                mapper = LinearColorMapper(palette=Turbo256, low=min(different_labels), high=max(different_labels))
                color_bar = ColorBar(
                    ticker=BasicTicker(desired_num_ticks=len(different_labels)),
                    color_mapper=mapper,
                    label_standoff = 12,
                    location = (0,0)
                )
                plot_figure.add_layout(color_bar, 'right')

            plot_figure.add_layout(Title(text='metric: {}, number of neighbors: {}, minimum sample size: {}, minimum cluster size: {}'.format(
                    new_input_metric, number_of_neighbors, minimum_samples, minimum_cluster_size
                ), text_font_style="italic"), 'above')

            if color_this_col != 'None':
                    plot_figure.add_layout(Title(text="UMAP projection colored in {} different {} values".format(len(different_labels), color_this_col), text_font_size="16pt"), 'above')
            else:
                if True in clustered:
                    plot_figure.add_layout(Title(text="UMAP projection with {} color-separated clusters".format(len(different_labels)-1), text_font_size="16pt"), 'above')
                else:
                    plot_figure.add_layout(Title(text="UMAP projection with no separated clusters", text_font_size="16pt"), 'above')

            st.bokeh_chart(plot_figure, use_container_width=True)

            countdf = pd.DataFrame(dtype=str)

            all_clusters = sorted(list(set(list(cluster_df['cluster']))))
            all_counts = {}

            items_to_add = []

            for cluster in all_clusters:
                # if int(cluster) >= 0:
                sdf = cluster_df[cluster_df['cluster'] == cluster]
                counts = Counter([item for inner in list(sdf['values']) for item in inner])
                all_counts[cluster] = counts
                sorted_counts = counts.most_common(20)

                add_to_results = ''
                if cluster != '-1':
                    add_to_results += 'In cluster {}, the most common diseases were '.format(cluster)
                    for i in range(results_max_n):
                        if i == max(range(results_max_n)):
                            add_to_results = add_to_results[:-2]
                            add_to_results += ' and '
                        add_to_results += "'{}' (n={}), ".format(sorted_counts[i][0], sorted_counts[i][1])
                else:
                    add_to_results += 'The most common diseases for the unclassified patients were '.format(cluster)
                    for i in range(results_max_n):
                        if i == max(range(results_max_n)):
                            add_to_results = add_to_results[:-2]
                            add_to_results += ' and '
                        add_to_results += "'{}' (n={}), ".format(sorted_counts[i][0], sorted_counts[i][1])

                items_to_add.append(add_to_results)

                toplist = ['{:03d}'.format(item[1]) + ': ' + item[0] for item in sorted_counts]
                countdf['Cluster ' + str(cluster)] = pd.Series(toplist, dtype=str)

            items_to_add = items_to_add[1:] + [items_to_add[0]]

            for i, item in enumerate(items_to_add):
                item = item[:-2]
                results_text += item + '. '

            # # cluster = 1
            if len(all_clusters) > 1:
                if prediction_to_cluster == -2:
                    cluster = st.sidebar.selectbox(
                        "Select a cluster for network analysis",
                        sorted(tuple(all_clusters), key=lambda x: int(x)),
                        1
                    )
                else:
                    cluster = prediction_to_cluster
            else:
                cluster = -1

            maximum_number_of_nodes = st.sidebar.slider(
                "Choose a limit for number of nodes", 1, 200, default_maximum_number_of_nodes, step=1
            )

            G = nx.Graph(cluster=str(cluster))
            nodeweights = {}
            edgeweights = {}

            def create_network(x):
                l = x
                for node in l:
                    if node in nodeweights:
                        nodeweights[node] += 1
                    else:
                        nodeweights[node] = 1
                combs = [(t[0], t[1]) if t[0] < t[1] else (t[1], t[0]) for t in list(itertools.combinations(l, 2))]
                # G.add_edges_from(combs)
                for comb in combs:
                    if comb in edgeweights:
                        edgeweights[comb] += 1
                    else:
                        edgeweights[comb] = 1
            
            sdf = cluster_df[cluster_df['cluster'] == str(cluster)]
            sdf['values'].apply(lambda x: create_network(x))

            nodeweights = {k: v for k, v in sorted(nodeweights.items(), key=lambda item: item[1], reverse=True)[:maximum_number_of_nodes]}
            labels, counts = zip(*list(nodeweights.items())[::-1])
            if int(cluster) != -1:
                mytitle = "Value counts in cluster {}".format(cluster)
            else:
                mytitle = "Value counts of unclustered data"
            p_node = figure(y_range=labels, title=mytitle,
                toolbar_location=None, plot_height=maximum_number_of_nodes*20)
            p_node.hbar(y=labels, right=counts, color=clustercolors[int(cluster)], height=0.618)
            p_node.title.text_color = clustercolors[int(cluster)]

            for nodekey in nodeweights:
                G.add_node(nodekey, size=nodeweights[nodekey])
            for edgekey in edgeweights:
                if edgekey[0] in nodeweights and edgekey[1] in nodeweights:
                    G.add_edge(edgekey[0], edgekey[1], weight=edgeweights[edgekey])
                # G.add_edge(1, 2, weight=3)
                # if edgekey[0] in nodeweights and edgekey[1] in nodeweights:
                #     G.edges[edgekey[0], edgekey[1]]['weight'] = edgeweights[edgekey]

            ### Keep only largest component
            # small_components = sorted(nx.connected_components(G), key=len)[:-1]
            # G.remove_nodes_from(itertools.chain.from_iterable(small_components))

            degrees = dict(nx.degree(G))
            nx.set_node_attributes(G, name='degree', values=degrees)

            number_to_adjust_by = 5
            adjusted_node_size = dict([(node, degree+number_to_adjust_by) for node, degree in nx.degree(G)])

            min_node_size = min(adjusted_node_size.values())
            max_node_size = max(adjusted_node_size.values())
            target_min_size = 8
            target_max_size = 14

            if max_node_size > min_node_size:
                adjusted_node_size = {k:target_min_size + ((v-min_node_size)/(max_node_size-min_node_size))*target_max_size for k, v in adjusted_node_size.items()}

            # node_labels_size = [8 + ((int(x[1]["adjusted_node_size"])-min_node_size)/(max_node_size-min_node_size))*14 for x in list(G.nodes.data())]

            nx.set_node_attributes(G, name='adjusted_node_size', values=adjusted_node_size)

            size_by_this_attribute = 'adjusted_node_size'
            # color_by_this_attribute = 'modularity_color'

            # try:
            #     communities = community.greedy_modularity_communities(G)
            # except:
            #     communities = None
            #     # color_by_this_attribute = clustercolors[int(cluster)]
            # else:
            #     # Create empty dictionaries
            #     modularity_class = {}
            #     modularity_color = {}

            #     communities_total = len(list(set(list(communities))))

            #     #Loop through each community in the network
            #     for community_number, community in enumerate(communities):
            #         #For each member of the community, add their community number and a distinct color
            #         for name in community: 
            #             modularity_class[name] = community_number
            #             modularity_color[name] = Turbo256[int(community_number*(1/communities_total)*len(Turbo256))]


            #     # Add modularity class and color as attributes from the network above
            #     nx.set_node_attributes(G, modularity_class, 'modularity_class')
            #     nx.set_node_attributes(G, modularity_color, 'modularity_color')

            #Choose colors for node and edge highlighting
            node_highlight_color = 'white'
            edge_highlight_color = 'black'

            #Choose a title!
            if int(cluster) != -1:
                title = 'Network graph of cluster {}'.format(cluster)
            else:
                title = 'Network graph of unclustered data'

            #Establish which categories will appear when hovering over each node
            # if communities:
            #     HOVER_TOOLTIPS = [
            #         ("Value", "@index"),
            #             ("Degree", "@degree"),
            #             ("Modularity Class", "@modularity_class"),
            #             ("Modularity Color", "$color[swatch]:modularity_color"),
            #     ]
            # else:
            HOVER_TOOLTIPS = [
                ("Value", "@index"),
                ("Degree", "@degree")
            ]

            #Create a plot — set dimensions, toolbar, and title
            plot = figure(tooltips = HOVER_TOOLTIPS, sizing_mode = 'scale_height',
                        tools="pan,wheel_zoom,save,reset", active_scroll='wheel_zoom',
                        x_range=Range1d(-10.1, 10.1), y_range=Range1d(-10.1, 10.1), title=title) # , plot_width=600, plot_height=600)

            plot.title.text_color = clustercolors[int(cluster)]
            
            # plot.background_fill_color = clustercolors[int(cluster)]
            # plot.background_fill_alpha = 0.1


            #Create a network graph object
            # https://networkx.github.io/documentation/networkx-1.9/reference/generated/networkx.drawing.layout.spring_layout.html
            network_graph = from_networkx(G, nx.spring_layout, scale=10, center=(0, 0))

            #Set node sizes and colors according to node degree (color as category from attribute) clustercolors[int(cluster)]
            network_graph.node_renderer.glyph = Circle(size=size_by_this_attribute, fill_color=clustercolors[int(cluster)], line_width=0)
            # network_graph.node_renderer.glyph = Circle(size=size_by_this_attribute, fill_color=color_by_this_attribute)
            #Set node highlight colors
            network_graph.node_renderer.hover_glyph = Circle(size=size_by_this_attribute, fill_color=node_highlight_color, line_width=2)
            network_graph.node_renderer.selection_glyph = Circle(size=size_by_this_attribute, fill_color=node_highlight_color, line_width=2)

            #Set edge opacity and width
            network_graph.edge_renderer.glyph = MultiLine(line_alpha=0.05, line_width=1)
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

            labels = []
            for x, y, name, fontsize, alpha in zip(source.data['x'], source.data['y'], source.data['name'], source.data['fontsize'], source.data['alpha']):
                labels.append(Label(x=x, y=y, text=name, text_alpha=alpha, text_align ='center', text_font_size=fontsize, background_fill_color='white', background_fill_alpha=.7))
                plot.add_layout(labels[-1])

            st.bokeh_chart(plot, use_container_width=True)
            st.bokeh_chart(p_node, use_container_width=True)

            if DEBUG_OPTIONS['save_graph']:
                gephi_cluster_df = cluster_df.copy()
                gephi_cluster_df['values'] = gephi_cluster_df['values'].apply(lambda x: '_'.join(x))
                for c in all_clusters:
                    os.makedirs('./graph_files', exist_ok=True)
                    gdf = gephi_cluster_df[gephi_cluster_df['cluster'] == c].reset_index(drop=True)
                    gdf.to_excel('./graph_files/graph_c' + str(c) + '.xlsx', index=False)

            # g = clusterer.condensed_tree_.to_networkx()
            # nx.write_gexf(g, "network.gexf")

            fig, ax = plt.subplots()
            clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=list(clustercolors.values())[1:], axis=ax)            
            plt.title('Cluster hierarchy as dendrogram')
            st.pyplot(fig)

            if cluster in all_counts:
                wordcloud = WordCloud(
                    background_color="white", 
                    max_words=200, 
                    contour_width=3, 
                    contour_color='steelblue',
                    width=int(1000*8/10),
                    height=int(618*8/10)
                    )
                wordcloud.generate_from_frequencies(dict(all_counts[cluster]))

                st.image(wordcloud.to_image(), width=None)

            with st.beta_expander("Results"):
                st.info('''
                    **Methods**\n

                    The raw dataset of contains {} different dimensions (= disease entities) with
                    binary values (0 = no disease, 1 = disease) and {} rows (= different patients).
                    The average number of diseases was {:.2f} +/- {:.2f} [95% CI].
                    
                    After filtering the {} column, the remaining dataset
                    consits of {} different dimensions and {} rows. The average number of 
                    diseases was {:.2f} +/- {:.2f} [95% CI].

                    After performing Uniform Manifold Approximation and Projection (UMAP)¹ with given parameters
                    (n_neighbors={}, min_dist={}) for dimension reduction, each row of the dataset is a {}-dimensional 
                    representation of the corresponding patients.

                    Next, a high performance implementation² of
                    Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN)³ 
                    was utilized, which uses unsupervised learning to find clusters (= dense regions) within
                    the dataset. After visual exploration of the {}-dimensional representation, we set the parameters
                    (min_samples={}, min_cluster_size={}, cluster_selection_epsilon={}) to get the best fit of the projected dense regions.

                    We used {} as a distance metric as suggested by Aggarwal, Hinneburg et Keim⁴ for higher dimensional data.

                    For reproducibility, all stochastic calculations were obtained with fixed random seed of {}.
                    
                    **Results**\n
                    After performing UMAP dimension reduction and HDBSCAN clustering, we found {} different clusters.
                    {} ({:.2f}%) patients remain unclassified.
                    
                    {}

                    **Citations**\n
                    ¹ McInnes, L., Healy, J., Saul, N. & Großberger, L. UMAP: uniform manifold approximation and projection. J. Open Source Softw. 3, 861 (2018).\n
                    ² L. McInnes, J. Healy, S. Astels, hdbscan: Hierarchical density based clustering In: Journal of Open Source Software, The Open Journal, volume 2, number 11. 2017.\n 
                    ³ Campello R.J.G.B., Moulavi D., Sander J. (2013) Density-Based Clustering Based on Hierarchical Density Estimates. In: Pei J., Tseng V.S., Cao L., Motoda H., Xu G. (eds) Advances in Knowledge Discovery and Data Mining. PAKDD 2013. Lecture Notes in Computer Science, vol 7819. Springer, Berlin, Heidelberg.\n
                    ⁴ Aggarwal C.C., Hinneburg A., Keim D.A. (2001) On the Surprising Behavior of Distance Metrics in High Dimensional Space. In: Van den Bussche J., Vianu V. (eds) Database Theory — ICDT 2001. ICDT 2001. Lecture Notes in Computer Science, vol 1973. Springer, Berlin, Heidelberg. https://doi.org/10.1007/3-540-44503-X_27

                '''.format(
                    len(df_cols), 
                    len(df_raw),
                    original_mean_ci[0], 
                    original_mean_ci[1],
                    ' '.join(options.keys()),
                    len(df.columns),
                    len(df),
                    filtered_mean_ci[0], 
                    filtered_mean_ci[1],
                    number_of_neighbors,
                    minimum_distance,
                    remaining_dimensions,
                    remaining_dimensions,
                    minimum_samples,
                    minimum_cluster_size,
                    cluster_selection_epsilon,
                    metric,
                    random_seed,
                    len(different_labels)-1,
                    len(cluster_df[cluster_df['cluster'] == '-1']),
                    100*len(cluster_df[cluster_df['cluster'] == '-1'])/len(df),
                    results_text
                    ))

            with st.beta_expander("Raw dataframe"):
                st.write(df)
                st.markdown(download_link(df, 'df_raw.csv', 'Download raw dataset'), unsafe_allow_html=True)

            with st.beta_expander("Clustered dataframe"):
                gephi_cluster_df = cluster_df.copy()
                gephi_cluster_df['values'] = gephi_cluster_df['values'].apply(lambda x: '_'.join(x))
                st.write(cluster_df)
                st.markdown(download_link(gephi_cluster_df, 'clusters.csv', 'Download clusters'), unsafe_allow_html=True)

            with st.beta_expander("Value counts dataframe"):
                st.write(countdf)
                st.markdown(download_link(countdf, 'counts.csv', 'Download counts'), unsafe_allow_html=True)

            if len(dict_df) > 0:
                with st.beta_expander("Dictionary for filetype conversion"):
                    st.write(dict_df)
                    st.markdown(download_link(dict_df, 'dictionary.csv', 'Download dictionary'), unsafe_allow_html=True)

        if (not st.session_state.started) and (not DEBUG_OPTIONS["DEBUG"]):
            st.write("""
                ## **Instructions**
                Next you can adjust the parameters in the left sidebar.
                Exclude columns you want to omit or select a column and filter
                your dataset by values in that specific column.
                If you are done, just press the **Start analysis** button at the
                bottom of the left sidebar.\n
            """)



    st.write("""
    ### **Citations**
    ¹ McInnes, L., Healy, J., Saul, N. & Großberger, L. UMAP: uniform manifold approximation and projection. J. Open Source Softw. 3, 861 (2018).\n
    ² L. McInnes, J. Healy, S. Astels, hdbscan: Hierarchical density based clustering In: Journal of Open Source Software, The Open Journal, volume 2, number 11. 2017
    """)


if __name__ == "__main__":
    main()