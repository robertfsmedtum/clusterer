import streamlit as st
import pandas as pd

def create_column_analysis_widget(
    df,
    df_raw,
    column_analysis,
    mean_confidence_interval,
):        

    if len(list(df_raw.columns)) < 8:

        col5, col6, col7 = column_analysis.beta_columns([1, 1, 1])
        col5.write('__Column__')
        col6.write('__Original mean__')
        col7.write('__Filtered mean__')

        for key in sorted(list(df_raw.columns)):
            col5.write('__' + key + '__')
            original_col_mean_ci = mean_confidence_interval(df_raw[key])
            col6.write('_{:.2f} +/- {:.2f} [95% CI]_'.format(original_col_mean_ci[0], original_col_mean_ci[1]))
            if key in df.columns:
                filtered_col_mean_ci = mean_confidence_interval(df[key])
                col7.write('_{:.2f} +/- {:.2f} [95% CI]_'.format(filtered_col_mean_ci[0], filtered_col_mean_ci[1]))
            else:
                col7.write('__-__')
                
    else:

        @st.cache 
        def create_record_df(df, df_raw, mean_confidence_interval):
            col_records = []
            for col in sorted(list(df_raw.columns)):
                original_col_mean_ci = mean_confidence_interval(df_raw[col])
                original_text = '{:.2f} +/- {:.2f}'.format(original_col_mean_ci[0], original_col_mean_ci[1])
                if col in df.columns:
                    filtered_col_mean_ci = mean_confidence_interval(df_raw[col])
                    filtered_text = '{:.2f} +/- {:.2f}'.format(filtered_col_mean_ci[0], filtered_col_mean_ci[1])
                else:
                    filtered_text = '-'

                col_records.append({
                    'Column': col,
                    'Original mean [95% CI]': original_text,
                    'Filtered mean [95% CI]': filtered_text
                })
            return pd.DataFrame.from_records(col_records)

        cadf = create_record_df(df, df_raw, mean_confidence_interval)
        column_analysis.write(cadf)