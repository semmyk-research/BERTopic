##-----ExtendClass-----visualise_topics_over_time_area-----##
## Extend class BERTopic (bertopic.py) | plotting.visualise DTM area plot
## where get_docs() retrieves all documents for a topic and return dataFrame
## Assists from https://stackoverflow.com/questions/15526858/how-to-extend-a-class-in-python


##### Start _topics_over_time_area.py __version__ 0.12.0

import pandas as pd                ##SemmyK: DataFrame
from typing import List            ##SemmyK: topic list
import plotly.graph_objects as go  ##SemmyK: Plotly
from sklearn.preprocessing import normalize  ##SemmyK: normalise y

def visualize_topics_over_time_area(topic_model,
                               topics_over_time: pd.DataFrame,
                               top_n_topics: int = None,
                               topics: List[int] = None,
                               normalize_frequency: bool = False,
                               custom_labels: bool = False,
                               width: int = 900, #1250,
                               height: int = 450) -> go.Figure:  ##450
    """ Visualize topics over time | SemmyK: extends BERTopic visualize_topics_over_time for Area Plot | Aug 2022
    Bug: //TODO: some topics key terms not showing on hover
        //TODO: Extend BERTopic class, move visualize_topics_over_time_area() to extended class

    Arguments:
        topic_model: A fitted BERTopic instance.
        topics_over_time: The topics you would like to be visualized with the
                          corresponding topic representation
        top_n_topics: To visualize the most frequent topics instead of all
        topics: Select which topics you would like to be visualized
        normalize_frequency: Whether to normalize each topic's frequency individually
        custom_labels: Whether to use custom topic labels that were defined using 
                       `topic_model.set_topic_labels`.
        width: The width of the figure.
        height: The height of the figure.

    Returns:

        ##A plotly.graph_objects.Figure including all traces

    Examples:  ##Usage:
    To visualize the topics over time, simply run:

    ```python
    topics_over_time_area = topic_model.topics_over_time_area(docs, timestamps, topics[optional])
    topic_model.visualize_topics_over_time_area(topics_over_time)
    ```

    Or if you want to save the resulting figure:

    ```python
    fig = topic_model.visualize_topics_over_time_area(topics_over_time_area)
    fig.write_html("path/to/file.html")
    ```
    <iframe src="../../getting_started/visualization/trump.html"
    style="width:1000px; height: 680px; border: 0px;""></iframe>

    """

    ## SemmyK: added addition colors | Red:"FF0000" , Gold:"#ffd700" , Teal:"#333300" , Silver: "#C0C0C0"
    '''
    ## Define 11 colours
    colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#D55E00", "#0072B2", "#CC79A7",
             "#FF0000" , "#ffd700" , "#C0C0C0" , "#333300"]
    '''
    ## from - named CSS color:
    colors = ['gold','burlywood','cyan','violet','silver','black','royalblue','honeydew','chocolate','olive','beige',
             'blueviolet','cornsilk','darkmagenta','floralwhite','darkorchid','coral','indigo','gray',
             'plum','salmon','tan','whitesmoke','azure','fuchsia','black']
    '''
    NB:       - named CSS color:
            aliceblue, antiquewhite, aqua, aquamarine, azure, beige, bisque, black, blanchedalmond, blue,
            blueviolet, brown, burlywood, cadetblue, chartreuse, chocolate, coral, cornflowerblue,
            cornsilk, crimson, cyan, darkblue, darkcyan, darkgoldenrod, darkgray, darkgrey, darkgreen,
            darkkhaki, darkmagenta, darkolivegreen, darkorange, darkorchid, darkred, darksalmon, darkseagreen,
            darkslateblue, darkslategray, darkslategrey, darkturquoise, darkviolet, deeppink, deepskyblue,
            dimgray, dimgrey, dodgerblue, firebrick, floralwhite, forestgreen, fuchsia, gainsboro,
            ghostwhite, gold, goldenrod, gray, grey, green, greenyellow, honeydew, hotpink, indianred, indigo,
            ivory, khaki, lavender, lavenderblush, lawngreen, lemonchiffon, lightblue, lightcoral, lightcyan,
            lightgoldenrodyellow, lightgray, lightgrey, lightgreen, lightpink, lightsalmon, lightseagreen,
            lightskyblue, lightslategray, lightslategrey, lightsteelblue, lightyellow, lime, limegreen,
            linen, magenta, maroon, mediumaquamarine, mediumblue, mediumorchid, mediumpurple,
            mediumseagreen, mediumslateblue, mediumspringgreen, mediumturquoise, mediumvioletred, midnightblue,
            mintcream, mistyrose, moccasin, navajowhite, navy, oldlace, olive, olivedrab, orange, orangered,
            orchid, palegoldenrod, palegreen, paleturquoise, palevioletred, papayawhip, peachpuff, peru, pink,
            plum, powderblue, purple, red, rosybrown, royalblue, rebeccapurple, saddlebrown, salmon,
            sandybrown, seagreen, seashell, sienna, silver, skyblue, slateblue, slategray, slategrey, snow,
            springgreen, steelblue, tan, teal, thistle, tomato, turquoise, violet, wheat, white, whitesmoke,
            yellow, yellowgreen
    '''

    # Select topics based on top_n and topics args
    freq_df = topic_model.get_topic_freq()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]
    
    if topics is not None:
        selected_topics = list(topics)
    elif top_n_topics is not None:
        selected_topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
    else:
        selected_topics = sorted(freq_df.Topic.to_list())

    # Prepare data
    if topic_model.custom_labels_ is not None and custom_labels:
        topic_names = {key: topic_model.custom_labels_[key + topic_model._outliers] for key, _ in topic_model.topic_labels_.items()}
    else:
        topic_names = {key: value[:40] + "..." if len(value) > 40 else value
                       for key, value in topic_model.topic_labels_.items()}
    topics_over_time["Name"] = topics_over_time.Topic.map(topic_names)
    data = topics_over_time.loc[topics_over_time.Topic.isin(selected_topics), :].sort_values(["Topic", "Timestamp"])
    #data = topics_over_time.loc[topics_over_time.Topic.isin(selected_topics), :]

    #data.head(4)  ##debug SMY
    
    ## Plot
    ## import libraries
    #import plotly.graph_objects as go

    # Add traces
    fig = go.Figure()
    for index, topic in enumerate(data.Topic.unique()):
        trace_data = data.loc[data.Topic == topic, :]
        topic_name = trace_data.Name.values[0]
        words = trace_data.Words.values  #[:5]  ##SemmyK:   |PS: [:5] removed in 0.12.0    
        if normalize_frequency:
            y = normalize(trace_data.Frequency.values.reshape(1, -1))[0]
        else:
            y = trace_data.Frequency
        fig.add_trace(go.Scatter(x=trace_data.Timestamp, y=y,
                                 mode='lines',
                                 marker_color=colors[index % len(colors)], ##SemmyK: [index % 11], :for topics >7
                                 hoverinfo="text",
                                 name=topic_name,
                                 hovertext=[f'<b>Topic {topic}</b><br>Words: {word}' for word in words if len(words)>1],
                                 stackgroup='one'     ##SemmyK: Key to area plot
                                )) ##SemmyK:  if len(words)>1 | insert for safeguard

    # Styling of the visualization
    ##SemmyK: adjusted for 'boarderless' transparent hover
    ##NB: transparent hover: clue from https://stackoverflow.com/questions/67386595/plotly-hoverlabel-color-transparency
    ## Plotly hovermode: See https://plotly.com/python/hover-text-and-formatting/
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    fig.update_layout(
        yaxis_title="Normalized Frequency" if normalize_frequency else "Frequency",
        title={
            'text': "<b>Topics over Time: Area Plot",  ##SemmyK: Area Plot
            'y': .85, ##SemmyK: #.95,
            'x': 0.40,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=20, #22,          ##SemmyK: lower font
                color="Black")
        },
        template="simple_white",
        width=width,
        height=height,
        hovermode='x unified',             ##SemmyK: single hover label. Perhaps, I should use 'x'. 
        hoverlabel=dict(
            bgcolor="rgba(0,0,0,.05)",  #"white", ##SemmyK:adjusted transparency
            font_size=12, #16,             ##SemmyK: lower font
            font_family="Rockwell",
            bordercolor = "rgba(0,0,0,0)"  ##SemmyK:remove line border
        ),
        legend=dict(
            title="<b><i>Topic Representation",  #"<b>Global Topic Representation",
        )
    )

    #return data, fig  ##SemmyK: there is no need explicitly returning data. See model.topics_over_time()  
    return fig
