import streamlit as st
import base64
import pandas as pd

def create_results_table_widget(
    df,
    cluster_df,
    countdf,
    dict_df
):    

    @st.cache
    def download_link(object_to_download, download_filename, download_link_text):
        if isinstance(object_to_download,pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False)
        b64 = base64.b64encode(object_to_download.encode()).decode()
        return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'
        
    with st.beta_expander("Raw dataframe"):
        st.write(df)
        st.markdown(download_link(df, 'df_raw.csv', 'Download raw dataset'), unsafe_allow_html=True)

    with st.beta_expander("Clustered dataframe"):
        gephi_cluster_df = cluster_df.copy()
        gephi_cluster_df['values'] = gephi_cluster_df['values'].apply(lambda x: '_'.join([str(y) for y in x]) if not isinstance(x, int) else '')
        # print(cluster_df)
        st.write(gephi_cluster_df)
        st.markdown(download_link(gephi_cluster_df, 'clusters.csv', 'Download clusters'), unsafe_allow_html=True)

    with st.beta_expander("Value counts dataframe"):
        st.write(countdf)
        st.markdown(download_link(countdf, 'counts.csv', 'Download counts'), unsafe_allow_html=True)

    if len(dict_df) > 0:
        with st.beta_expander("Dictionary for filetype conversion"):
            st.write(dict_df)
            st.markdown(download_link(dict_df, 'dictionary.csv', 'Download dictionary'), unsafe_allow_html=True)