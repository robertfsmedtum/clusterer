import streamlit as st
from wordcloud import WordCloud

def create_wordcloud_widget(cluster, all_counts):
    if cluster in all_counts:
        if len(all_counts[cluster]) > 0:
            @st.cache
            def generate_wordcloud(cluster, all_counts):
                wordcloud = WordCloud(
                    background_color="white", 
                    max_words=200, 
                    contour_width=3, 
                    contour_color='steelblue',
                    width=int(1000*8/10),
                    height=int(618*8/10)
                    )
                wordcloud.generate_from_frequencies(dict(all_counts[cluster]))
                return wordcloud.to_image()

            st.image(generate_wordcloud(cluster, all_counts), width=None)