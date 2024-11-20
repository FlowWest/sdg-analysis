import streamlit as st
import pandas as pd
import altair as alt

# Title and description
st.title("Exploratory Data Visualizations for SDG Analysis")
st.write("Upload your data and explore interactive visualizations.")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    # Load the uploaded file into a DataFrame
    full_merged_df = pd.read_csv(uploaded_file, parse_dates=["datetime"])  # Ensure 'datetime' is parsed correctly

    # Display data preview
    st.write("### Data Preview")
    st.dataframe(full_merged_df.head())

    # Altair Visualization
    st.write("### Interactive Visualization")

    # Define the Altair charts
    interval = alt.selection_interval(encodings=["x"])
    
    line = alt.Chart(full_merged_df).mark_line(color="darkblue").encode(
        x=alt.X('yearmonthdatehoursminutes(datetime):T', title='Datetime', axis=alt.Axis(format='%b %d, %Y', labelAngle=-45)),
        y=alt.Y('GLC:Q', title='Velocity (ft/s)')
    ).properties(
        width=500,
        height=300,
        title='Velocity Through Fish Passage-Zoomed',
    ).add_selection(interval)

    bar_chart = alt.Chart(full_merged_df).mark_bar().encode(
        x=alt.X('Velocity_Category:N', title='Velocity Category'),
        y=alt.Y('sum(time_unit):Q', title='Total Time (Hours)'),
        color=alt.Color('DGL:N', title='Gate Status', scale=alt.Scale(scheme='dark2')),
        tooltip=alt.Tooltip('sum(time_unit):Q', title="Total Time (Hours)")
    ).properties(
        width=300,
        height=300,
        title="Comparison of Hours by Velocity Category and Gate Status"
    ).transform_filter(interval)

    area_dgl_true = alt.Chart(full_merged_df).mark_rect(color='orange').encode(
        x='gate_min_datetime:T',
        x2='gate_max_datetime:T',
        opacity=alt.value(0.008)
    ).transform_filter(
        alt.datum.DGL == "Open"
    )

    yrule = alt.Chart().mark_rule(strokeDash=[12, 6], size=2, color='red').encode(
        y=alt.datum(8)
    ).properties(
        width=500,
        height=300
    ).encode(
        tooltip=alt.TooltipValue('8 ft/s Threshold')
    )

    nearest = alt.selection_point(nearest=True, on="pointerover", fields=["datetime"], empty=False)
    points = line.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )
    
    rules = alt.Chart(full_merged_df).transform_calculate(
        FlowVelocityDuration="'Flow ' + datum.Velocity_Category + ' duration is ' + datum.streak_duration + ' hours'",
        GateStatusDuration="'Gate ' + datum.DGL + ' duration is ' + datum.gate_streak_duration + ' hours'"
    ).mark_rule(color="gray").encode(
        x="datetime:T",
        opacity=alt.condition(nearest, alt.value(0.3), alt.value(0)),
        tooltip=[
            alt.Tooltip('yearmonthdatehoursminutes(datetime):T', title='Datetime'),
            alt.Tooltip('GLC:Q', title="Velocity (ft/s)", format=".2f"),
            alt.Tooltip('FlowVelocityDuration:N', title="Flow Velocity Duration"),
            alt.Tooltip('GateStatusDuration:N', title="Gate Status Duration")
        ]
    ).add_params(nearest)

    combined_chart = alt.hconcat(
        alt.layer(line, points, yrule, rules, area_dgl_true),
        bar_chart
    ).configure_title(
        fontSize=17,
        color="black",
        font='Arial'
    )

    # Render the chart in Streamlit
    st.altair_chart(combined_chart, use_container_width=True)
else:
    st.write("Please upload a CSV file to see the visualization.")
