import streamlit as st
import pandas as pd
from bokeh.palettes import Turbo256, RdYlGn11
from bokeh.plotting import figure
from bokeh.transform import linear_cmap
from bokeh.models import ColumnDataSource, HoverTool, ColorBar, BasicTicker, Title, LinearColorMapper, FixedTicker, FuncTickFormatter

RdGn2 = ('#006837', '#a50026')

@st.cache
def return_value_list(x, values):
    values_to_return = []
    for value in values:
        if x[value] == 1:
            values_to_return.append(value)
    return values_to_return 

def create_cluster_graph_widget(
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
):
        datasource = ColumnDataSource(cluster_df)
        plot_figure = figure(
            tools=('pan, wheel_zoom, reset, save'),
            output_backend="svg"
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

        if color_this_col != 'by cluster':
            color_col = cluster_df['color by ' + color_this_col]
            if len(different_labels) == 2:
                color_palette = RdGn2
            else:
                color_palette = RdYlGn11
            mapper = linear_cmap(field_name='color by ' + color_this_col, palette=color_palette, low=min(color_col), high=max(color_col))
            # mapper = linear_cmap(field_name='color by ' + color_this_col, palette=Turbo256, low=min(different_labels), high=max(different_labels)+0.8)
        else:
            color_palette = Turbo256
            mapper = linear_cmap(field_name='cluster', palette=color_palette, low=min(different_labels), high=max(different_labels))

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

        if len(new_values) > 0:
            cluster_df_new = pd.DataFrame(test_embedding, columns=('x', 'y'))
            if color_this_col != 'by cluster':
                cluster_df_new['color by ' + color_this_col] = new_values[color_this_col]
            cluster_df_new['cluster'] = pd.Series(new_predictions[0])
            cluster_df_new['values'] = testdf.apply(lambda x: return_value_list(x, values), axis=1)
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
            if color_this_col != 'by cluster':
                mi = min(color_col)
                ma = max(color_col)
                mapper = LinearColorMapper(palette=color_palette, low=mi, high=ma)

                if color_palette == RdGn2:
                    desired_ticks = 2
                    step = ma - mi
                    smi = mi + step/4
                    sma = ma - step/4
                    mid = mi + step/2
                    ticks_list = [smi, sma]
                    ticks_list_dict = {smi: mi, sma: ma}
                    mylist = [mi, mid, ma]
                    ticker = FixedTicker(ticks=ticks_list)
                    formatter = FuncTickFormatter(code="""
                        var m = %s;  
                        if (tick < m[1]) {
                            return m[0]
                        } else {
                            return m[2]                  
                        };
                        """ % mylist)
                else:
                    desired_ticks = 11
                    dif = ma - mi
                    step = dif/desired_ticks
                    ticks_list = [mi] + [mi+i*step for i in list(range(desired_ticks))[1:]] + [ma]
                    ticks_list = [int(x) if float(x).is_integer() else round(x, 1) for x in ticks_list]
                    ticker = FixedTicker(ticks=ticks_list)
                    formatter = FuncTickFormatter(code="""
                        return tick
                        """)
                color_bar = ColorBar(
                    ticker=ticker,
                    formatter=formatter,
                    color_mapper=mapper,
                    label_standoff = 12,
                    location = (0,0)
                )
            else:
                mapper = LinearColorMapper(palette=color_palette, low=min(different_labels), high=max(different_labels))
                desired_ticks = len(different_labels)
                color_bar = ColorBar(
                    ticker=BasicTicker(desired_num_ticks=desired_ticks),
                    color_mapper=mapper,
                    label_standoff = 12,
                    location = (0,0)
                )

            plot_figure.add_layout(color_bar, 'right')

        if dimension_reduction_method == 'UMAP':
            plot_figure.add_layout(Title(text='metric: {}, number of neighbors: {}, minimum sample size: {}, minimum cluster size: {}'.format(
                    new_input_metric, number_of_neighbors, minimum_samples, minimum_cluster_size
                ), text_font_style="italic"), 'above')
        else:
            plot_figure.add_layout(Title(text='MN_ratio: {}, FP_ratio: {}, minimum sample size: {}, minimum cluster size: {}'.format(
                    mn_ratio, fp_ratio, minimum_samples, minimum_cluster_size
                ), text_font_style="italic"), 'above')

        if color_this_col != 'by cluster':
                plot_figure.add_layout(Title(text=dimension_reduction_method + " projection colored by {} [{}-{}]".format(
                    color_this_col, 
                    int(min(different_labels)) if float(min(different_labels)).is_integer() else min(different_labels), 
                    int(max(different_labels)) if float(max(different_labels)).is_integer() else max(different_labels), 
                    ), text_font_size="16pt"), 'above')
                # plot_figure.add_layout(Title(text=dimension_reduction_method + " projection colored in {} values".format(len(different_labels), color_this_col), text_font_size="16pt"), 'above')
        else:
            if True in clustered:
                plot_figure.add_layout(Title(text=dimension_reduction_method + " projection with {} color-separated clusters".format(len(different_labels)-1), text_font_size="16pt"), 'above')
            else:
                plot_figure.add_layout(Title(text=dimension_reduction_method + " projection with no separated clusters", text_font_size="16pt"), 'above')

        st.bokeh_chart(plot_figure, use_container_width=True)