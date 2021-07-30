import streamlit as st

def create_results_text_widget(
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
):
    @st.cache
    def create_results_text(
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
    ):
        if dimension_reduction_method == 'UMAP':
            dimension_reduction_text = 'Uniform Manifold Approximation and Projection (UMAP)'
            dimension_reduction_parameters = 'with given parameters (n_neighbors={}, min_dist={}) '.format(
                number_of_neighbors,
                int(minimum_distance) if float(minimum_distance).is_integer() else minimum_distance,
            )
            metric_text = 'We used {} as a distance metric as suggested by Aggarwal, Hinneburg et Keim⁴ for higher dimensional data.'.format(metric)
            dimension_reduction_citation = 'McInnes, L., Healy, J., Saul, N. & Großberger, L. UMAP: uniform manifold approximation and projection. J. Open Source Softw. 3, 861 (2018).'
        else:
            dimension_reduction_text = 'Pairwise Controlled Manifold Approximation (PaCMAP)'
            dimension_reduction_parameters = 'with given parameters (MN_ratio={}, FP_ratio={}) '.format(
                int(mn_ratio) if float(mn_ratio).is_integer() else mn_ratio,
                int(fp_ratio) if float(fp_ratio).is_integer() else fp_ratio,
            )                
            metric_text = ''
            dimension_reduction_citation = 'Yingfan Wang, , Haiyang Huang, Cynthia Rudin, and Yaron Shaposhnik. "Understanding How Dimension Reduction Tools Work: An Empirical Approach to Deciphering t-SNE, UMAP, TriMAP, and PaCMAP for Data Visualization." (2020).'
        return '''
            **Methods**\n

            The raw dataset of contains {} different dimensions (= disease entities) with
            binary values (0 = no disease, 1 = disease) and {} rows (= different patients).
            The average number of diseases was {:.2f} +/- {:.2f} [95% CI].
            
            After filtering the {} column, the remaining dataset
            consits of {} different dimensions and {} rows. The average number of 
            diseases was {:.2f} +/- {:.2f} [95% CI].

            After performing {}¹ {}for dimension reduction, each row of the dataset became a {}-dimensional 
            representation of the corresponding patient.

            Next, a high performance implementation² of
            Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN)³ 
            was utilized, which uses unsupervised learning to find clusters (= dense regions) within
            the dataset. After visual exploration of the {}-dimensional representation and the cluster hierarchy dendrogram (Supp. Fig. 1), 
            we set the parameters (min_samples={}, min_cluster_size={}, cluster_selection_epsilon={}) to get the best fit of the projected dense regions.
            {}
            For reproducibility a random seed of {} was set.
            
            **Results**\n
            After performing {} dimension reduction and HDBSCAN clustering, we found {} different clusters.
            {} ({:.2f}%) patients were not assigned to any cluster.
            
            {}

            **Citations**\n
            ¹ {}\n
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
            dimension_reduction_text,
            dimension_reduction_parameters,
            remaining_dimensions,
            remaining_dimensions,
            minimum_samples,
            minimum_cluster_size,
            int(cluster_selection_epsilon) if float(cluster_selection_epsilon).is_integer() else cluster_selection_epsilon,
            metric_text,
            random_seed,
            dimension_reduction_method,
            len(different_labels)-1,
            len(cluster_df[cluster_df['cluster'] == '-1']),
            100*len(cluster_df[cluster_df['cluster'] == '-1'])/len(df),
            results_text,
            dimension_reduction_citation
            )


    with st.beta_expander("Results"):
        st.info(create_results_text(
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
        ))