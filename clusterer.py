def main():
    import pandas as pd
    import numpy as np
    import hdbscan
    import copy
    import scipy
    import pacmap
    import streamlit as st
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics.pairwise import pairwise_distances
    from bokeh.palettes import Turbo256
    from collections import Counter
    import umap
    import pynndescent
    import itertools
    import networkx as nx

    from widgets.value_counts import create_value_counts_widget
    from widgets.dendrogram import create_dendrogram_widget
    from widgets.wordcloud import create_wordcloud_widget
    from widgets.results_text import create_results_text_widget
    from widgets.results_table import create_results_table_widget
    from widgets.network_graph import create_network_graph_widget
    from widgets.column_analysis import create_column_analysis_widget
    from widgets.cluster_graph import create_cluster_graph_widget

    DEBUG_OPTIONS = {
        "DEBUG": True,
        "input": "./data/data.csv",
        "save_graph": False,
        # "options": {
        #     'psopat': [1]
        # }
    }

    results_max_n = 3
    results_text = ''
    drop_incomplete_rows_default = 0

    n_categories_for_float = 4
    use_bars = True

    default_disabled_modules = ['Column analysis']
    # default_disabled_modules = ['Wordcloud', 'Dendrogram', 'Results']
    # default_disabled_modules = ['Network graph', 'Wordcloud', 'Dendrogram', 'Results']
    # default_disabled_modules = ['Column analysis', 'Network graph', 'Value counts', 'Dendrogram', 'Wordcloud', 'Results (tables)', 'Results (text)']
    module_list = ['Column analysis', 'Network graph', 'Value counts', 'Dendrogram', 'Wordcloud', 'Results (tables)', 'Results (text)']
    
    if n_categories_for_float == 4 and use_bars:
        quartile_bars = ['▂', '▄', '▆', '█']

    ### Reduction of dimensions and parameters
    dimension_reduction_method_default = 0 # 1 = PacMAP, 0 = UMAP
    remaining_dimensions = 2

    ### PacMAP and UMAP
    default_number_of_neighbors = 20 # 20 # 15 # 20 # 5

    ### PaCMAP
    mn_ratio_default=5.0
    fp_ratio_default=10.0

    ### UMAP
    minimum_distance = 0.0

    ### Transformation of dataset
    standard_scalar_default = 0 # 1 = Yes, 0 = No
    pairwisedistance_default = 0 # 1 = Yes, 0 = No

    ### Cluster settings for network and value counts
    default_maximum_number_of_nodes = 20
    densemap_default = 0

    random_seed = 42

    np.random.seed()

    if 'started' not in st.session_state:
        st.session_state.started = False

    ## TODO progress like: ▁ ▂ ▃ ▄ ▅ ▆ ▇ █
    ## TODO progress like: ▂ ▄ ▆ █

    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.set_page_config(
        page_title="High dimension clusterer",
        page_icon="🕸",
        layout="centered",
        initial_sidebar_state="auto",
    )

    @st.cache
    def convert_dataframe(df, drop_incomplete_rows):
        if drop_incomplete_rows == 'Yes':
            df.replace('', np.nan, inplace=True)
            df.dropna(inplace=True)
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

    st.write("""
    # 🕸 High Dimension Clusterer v0.1
    **for higher dimensional data using UMAP¹/PaCMAP² and HDBSCAN³𝄒⁴**
    """)

    drop_incomplete_rows = st.sidebar.radio(
        'Drop incomplete rows?',
        ['Yes', 'No'],
        drop_incomplete_rows_default
    )

    uploaded_file = st.sidebar.file_uploader("📂 Select a file (csv or excel)")

    if (uploaded_file is None and not DEBUG_OPTIONS["DEBUG"]) and 'df_raw' not in st.session_state:
        st.write("""
            ## **Instructions**
            The High Dimension Clusterer (HDC) lets you analyze higher dimensional datasets in the form of
            csv or excel files containing columns of binary, categorical or numerical data.\n
            After you haven chosen your dataset, you can tune the parameters or start the analysis right away.\n
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

        df_raw, dict_df = convert_dataframe(df_raw, drop_incomplete_rows)

        df_cols = list(df_raw.columns)

        col1, col2, col3, col4 = st.beta_columns([1, 1, 1, 2])
        col1.write('__Data__')
        col2.write('__Dimensions__')
        col3.write('__Rows__')
        col4.write('__Row value mean__')

        column_analysis = st.empty()

        dimension_reduction_method = st.sidebar.radio(
            'Use the following dimension reduction method',
            ['UMAP', 'PaCMAP'],
            dimension_reduction_method_default
        )

        if dimension_reduction_method == 'UMAP':
            ### HDBSCAN
            default_minimum_samples = 5 # 5 # 5 # 10 # 5
            default_minimum_cluster_size = 15 # 20 # 20 # 30 # 50
            default_selection_epsilon = 0.0
        else:
            ### HDBSCAN
            default_minimum_samples = 5 # 5 # 5 # 10 # 5
            default_minimum_cluster_size = 30 # 30 # 20 # 20 # 30 # 50
            default_selection_epsilon = 0.0

        with st.sidebar.beta_expander("Filtering + coloring data"):
            options_col_to_exclude = st.multiselect(
                'Excluding selected columns.',
                df_cols,
                )

            options_col_to_analyze = st.multiselect(
                'Select column(s) to separate data',
                df_cols,
                )
            
            color_this_col = st.selectbox(
                'Select column to color',
                ['by cluster'] + df_cols,
            )

            disabled_modules = st.multiselect(
                'Disable modules',
                module_list,
                default = default_disabled_modules
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
            if dimension_reduction_method == 'UMAP':
                metric = st.selectbox(
                    "What metric do you want to use in UMAP?",
                    (
                        "euclidean",
                        "manhattan",
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
            else:
                metric = "euclidean"

            if dimension_reduction_method == 'UMAP':
                number_of_neighbors = st.slider('Minimum number of neighbors (UMAP)', 2, 100, default_number_of_neighbors, step=1)
                mn_ratio = 0
                fp_ratio = 0
            else:
                mn_ratio = st.slider('Mid-near pairs to neighbors ratio (PaCMAP)', 0.0, 20.0, mn_ratio_default, step=0.1, format=f"%1f")
                fp_ratio = st.slider('Further pairs to neighbors ratio (PaCMAP)', 0.0, 20.0, fp_ratio_default, step=0.1, format=f"%1f")
                number_of_neighbors = None
            minimum_samples = st.slider('Minimum number of samples (HDBSCAN)', 1, 100, default_minimum_samples, step=1)
            minimum_cluster_size = st.slider('Minimum cluster size (HDBSCAN)', 2, 100, default_minimum_cluster_size, step=1)
            cluster_selection_epsilon = st.slider('Cluster selection minimum (HDBSCAN)', 0.0, 2.0, default_selection_epsilon, step=0.1, format=f"%1f")
            standardscalar = st.radio(
                'Use standardscalar transformation',
                ['Yes', 'No'],
                1-standard_scalar_default
            )
            densemap = st.radio(
                'Use densmap',
                ['Yes', 'No'],
                1-densemap_default
            )
            pairwisedistance = st.radio(
                'Use pairwise distance',
                ['Yes', 'No'],
                1-pairwisedistance_default
            )

        if st.sidebar.button('▶️ Start analysis') or DEBUG_OPTIONS["DEBUG"] or st.session_state.started:
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

            if dimension_reduction_method == 'UMAP':
                # new_values = st.sidebar.multiselect(
                #     'Select values for new prediction.',
                #     df_cols
                # )
                new_values = {}
                new_cols = st.sidebar.multiselect(
                    'Select columns for new prediction.',
                    df_cols
                )
                for new_col in new_cols:
                    new_values[new_col] = st.sidebar.slider(
                        'Value for ' + new_col, 
                        min(df[new_col]), 
                        max(df[new_col]), 
                        min(df[new_col]), 
                        step=0.1 if max(df[new_col]) < 2 else 1.0, 
                        format=f"%1f" if max(df[new_col]) < 2 else f"%0f")
            else:
                new_values = ''

            col1.write('**Original**')
            col2.write('_{:,}_'.format(len(df_cols)))
            col3.write('_{:,}_'.format(len(df_raw)))
            original_mean_ci = mean_confidence_interval(df_raw.T.sum())
            col4.write('_{:.2f} +/- {:.2f} [95% CI]_'.format(original_mean_ci[0], original_mean_ci[1]))
            col1.write('**Filtered**')
            col2.write('_{:,}_'.format(len(df.columns)))
            col3.write('_{:,}_'.format(len(df)))
            filtered_mean_ci = mean_confidence_interval(df.T.sum())
            col4.write('_{:.2f} +/- {:.2f} [95% CI]_'.format(filtered_mean_ci[0], filtered_mean_ci[1]))

            if 'Column analysis' not in disabled_modules:
                create_column_analysis_widget(
                    df,
                    df_raw,
                    column_analysis,
                    mean_confidence_interval
                )

            scaled_df = df
            if standardscalar == 'Yes':
                scaled_df = StandardScaler().fit_transform(scaled_df)
            if pairwisedistance == 'Yes':
                scaled_df = pairwise_distances(scaled_df)

            @st.cache(hash_funcs={pynndescent.pynndescent_.NNDescent: lambda _: 1})
            def calculate_umap(n_o_n, input_df, minimum_distance, use_densemap):
                return umap.UMAP(
                    n_neighbors=n_o_n,
                    min_dist=minimum_distance,
                    n_components=remaining_dimensions,
                    random_state=random_seed,
                    densmap=use_densemap
                    ).fit(input_df)

            @st.cache
            def calculate_pacmap(input_df, mn_ratio, fp_ratio, number_of_neighbors):
                X = np.array(input_df)
                # X = X.reshape(X.shape[0], -1)
                embedding = pacmap.PaCMAP(n_dims=2, MN_ratio=mn_ratio, FP_ratio=fp_ratio, n_neighbors=number_of_neighbors) # , MN_ratio=0.5, FP_ratio=2.0) 
                return embedding.fit_transform(X, init="pca")

            if dimension_reduction_method == 'UMAP':
                use_densemap = True if densemap == 'Yes' else False
                trans = calculate_umap(number_of_neighbors, scaled_df, minimum_distance, use_densemap)
                standard_embedding = trans.transform(scaled_df)
            else:
                standard_embedding = calculate_pacmap(scaled_df, mn_ratio, fp_ratio, number_of_neighbors)

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
            hdbscan_probabilities = clusterer.probabilities_

            clustered = (hdbscan_labels >= 0)

            if True not in clustered:
                st.warning('Did not find any clusters. Please adjust the parameters or choose a different metric.')

            ### TODO: Float columns als kategorisch mit einteilung n_max_number_of_categories
            values = list(df.columns)

            @st.cache
            def create_categorical_df(df, n_cats, mode='linear'):
                def create_appendix(i, n_cats):
                    if use_bars and n_categories_for_float == 4:
                        return ' ' + quartile_bars[i-1]
                    else:
                        return ' (' + str(i+1) + '/' + str(n_cats) + ')'
                ndf = pd.DataFrame()
                for col in df.columns:
                    different_values = list(set(list(df[col])))
                    ser = df[col]
                    if len(different_values) > 2:
                        if mode == 'linear':
                            mi = min(ser)
                            ma = max(ser)
                            d = ma - mi
                            steps = d/n_cats
                            bins_mid = [np.round(steps*i+mi, 2) for i in range(n_cats)[1:]]
                        elif mode == 'percentile':
                            bins_mid = [np.round(np.percentile(ser, i*(100/n_cats)), 2) for i in range(n_cats)[1:]]
                        bins = [-np.inf] + bins_mid + [np.inf]
                        labels = []
                        for i, cutoff in enumerate(bins_mid):
                            if i == 0:
                                labels.append('{} <= {}'.format(col, cutoff) + create_appendix(i+1, n_cats))
                                labels.append('{} < {} <= {}'.format(cutoff, col, bins_mid[i+1]) + create_appendix(i+2, n_cats))
                            elif i == len(bins_mid) - 1:
                                labels.append('{} > {}'.format(col, cutoff) + create_appendix(i+2, n_cats))
                            else:
                                labels.append('{} < {} <= {}'.format(cutoff, col, bins_mid[i+1]) + create_appendix(i+2, n_cats))
                        ndf[col] = pd.cut(ser, bins=bins, labels=labels)
                    else:
                        if 0 in different_values and 1 in different_values:
                            ser = ser.replace(1, col)
                            ser = ser.replace(0, '')
                        ndf[col] = ser
                return ndf.reset_index(drop=True) 

            @st.cache
            def return_row_values(x):
                return [y for y in list(x) if y]
                    
            cluster_df = pd.DataFrame(standard_embedding, columns=('x', 'y'))
            cluster_df['cluster'] = [str(x) for x in hdbscan_labels]
            cluster_df['probabilities'] = hdbscan_probabilities
            if color_this_col != 'by cluster':
                cluster_df['color by ' + color_this_col] = df_raw[color_this_col]
                different_labels = list(set(df_raw[color_this_col]))
            else:
                different_labels = list(set(hdbscan_labels))

            categorical_df = create_categorical_df(df, n_categories_for_float, mode='linear')

            cluster_df['values'] = categorical_df.apply(lambda x: return_row_values(x), axis=1)

            different_labels_sorted = sorted(different_labels)
            total_color_length = len(Turbo256)
            clustercolors = {}

            if len(different_labels_sorted) > 1:
                for i in range(len(different_labels_sorted)):
                    colorindex = i * int(total_color_length/(len(different_labels_sorted)-1))
                    if colorindex > 255:
                        colorindex = 255
                    clustercolors[int(different_labels_sorted[i])] = Turbo256[colorindex]
            else:
                clustercolors[-1] = Turbo256[0]

            prediction_to_cluster = -2

            testdf = pd.DataFrame(columns=scaled_df.columns)
            if len(new_values) > 0:
                new_values = { k: (new_values[k] if k in new_values else min(scaled_df[k])) for k in testdf.columns }
                testdf.loc[0] = [new_values[key] if key in new_values else min(scaled_df[key]) for key in testdf.columns]
                test_embedding = trans.transform(testdf)
                new_predictions = hdbscan.approximate_predict(clusterer, test_embedding)

            create_cluster_graph_widget(
                color_this_col,
                cluster_df,
                clustered,
                different_labels,
                new_values,
                test_embedding,
                new_predictions,
                testdf,
                mn_ratio,
                fp_ratio,
                minimum_samples,
                minimum_cluster_size,
                dimension_reduction_method,
                number_of_neighbors,
                new_input_metric,
                values
            )

            if len(new_values) > 0:
                prediction_to_cluster = new_predictions[0][0]

            countdf = pd.DataFrame(dtype=str)

            all_clusters = sorted(list(set(list(cluster_df['cluster']))), key=lambda x: int(x))
            all_counts = {}

            items_to_add = []

            for cluster in all_clusters:
                sdf = cluster_df[cluster_df['cluster'] == cluster]
                counts = Counter([item for inner in list(sdf['values']) for item in inner])
                all_counts[cluster] = counts
                sorted_counts = counts.most_common(20)

                add_to_results = ''
                if len(sorted_counts) > 0:
                    if cluster != '-1':
                        add_to_results += 'In cluster {}, the most common diseases were '.format(cluster)
                        for i in range(results_max_n if len(sorted_counts) >= results_max_n else len(sorted_counts)):
                            if i == max(range(results_max_n)):
                                add_to_results = add_to_results[:-2]
                                add_to_results += ' and '
                            add_to_results += "'{}' (n={}), ".format(sorted_counts[i][0], sorted_counts[i][1])
                    else:
                        add_to_results += 'The most common diseases for the unclassified patients were '.format(cluster)
                        for i in range(results_max_n if len(sorted_counts) >= results_max_n else len(sorted_counts)):
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
                for comb in combs:
                    if comb in edgeweights:
                        edgeweights[comb] += 1
                    else:
                        edgeweights[comb] = 1
            
            sdf = cluster_df[cluster_df['cluster'] == str(cluster)].copy()

            if not any(list(sdf['values'])):
                sdf['values'] = [['None'] for _ in sdf.index]
            sdf['values'].apply(lambda x: create_network(x))

            nodeweights = {k: v for k, v in sorted(nodeweights.items(), key=lambda item: item[1], reverse=True)[:maximum_number_of_nodes]}
            labels, counts = zip(*list(nodeweights.items())[::-1])

            if 'Network graph' not in disabled_modules:
                create_network_graph_widget(
                    G,
                    nodeweights,
                    edgeweights,
                    cluster,
                    clustercolors,
                )

            if 'Value counts' not in disabled_modules:
                create_value_counts_widget(    
                    cluster, 
                    labels, 
                    maximum_number_of_nodes,
                    counts,
                    clustercolors
                )

            if 'Dendrogram' not in disabled_modules:
                create_dendrogram_widget(clusterer, clustercolors)

            if 'Wordcloud' not in disabled_modules:
                create_wordcloud_widget(cluster, all_counts)

            if 'Results (text)' not in disabled_modules:
                create_results_text_widget(
                    dimension_reduction_method,
                    number_of_neighbors,
                    minimum_distance,
                    metric,
                    mn_ratio,
                    fp_ratio,
                    df_cols, 
                    df_raw,
                    df,
                    original_mean_ci, 
                    filtered_mean_ci,
                    options,
                    remaining_dimensions,
                    minimum_samples,
                    minimum_cluster_size,
                    cluster_selection_epsilon,
                    random_seed,
                    different_labels,
                    cluster_df,
                    results_text
                )

            if 'Results (tables)' not in disabled_modules:
                create_results_table_widget(
                    df,
                    cluster_df,
                    countdf,
                    dict_df
                )

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
    ² Yingfan Wang, , Haiyang Huang, Cynthia Rudin, and Yaron Shaposhnik. "Understanding How Dimension Reduction Tools Work: An Empirical Approach to Deciphering t-SNE, UMAP, TriMAP, and PaCMAP for Data Visualization." (2020).\n
    ³ L. McInnes, J. Healy, S. Astels, hdbscan: Hierarchical density based clustering In: Journal of Open Source Software, The Open Journal, volume 2, number 11. 2017.\n
    ⁴ Campello R.J.G.B., Moulavi D., Sander J. (2013) Density-Based Clustering Based on Hierarchical Density Estimates. In: Pei J., Tseng V.S., Cao L., Motoda H., Xu G. (eds) Advances in Knowledge Discovery and Data Mining. PAKDD 2013. Lecture Notes in Computer Science, vol 7819. Springer, Berlin, Heidelberg.
    """)


if __name__ == "__main__":
    main()