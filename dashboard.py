import streamlit as st
import pandas as pd
import altair as alt

# Title and description
# st.set_page_config(layout="wide")
st.title("Exploratory Data Visualizations for SDG Analysis")
st.write("Upload your data and explore interactive visualizations.")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    # Load the uploaded file into a DataFrame
    full_merged_df = pd.read_csv(uploaded_file, parse_dates=["datetime"])  # Ensure 'datetime' is parsed correctly
    full_merged_df['week'] = full_merged_df['datetime'].dt.isocalendar().week
    # full_merged_df['week'].unique()
    weekly_velocity = full_merged_df.groupby(["week", "Velocity_Category"])["time_unit"].sum().reset_index()
    
        # Define the Altair charts
    summary_stats_vel = (full_merged_df.groupby(["week", "date", "Velocity_Category"]).
        agg(
            total_velocity_duration = ("time_unit", "sum")
        )).reset_index()
    # print(summary_stats_vel.head())
    summary_stats_dgl = (full_merged_df.groupby(["week", "date", "DGL"]).
        agg(
            total_gate_duration = ("time_unit", "sum")
        )).reset_index()
    
    # Display data preview
    st.write("### Data Preview")
    st.dataframe(full_merged_df.head())

    # Altair Visualization
    st.write("### Interactive Visualization")
    
    brush = alt.selection_interval(encodings=['x'], mark=alt.BrushConfig(stroke="cyan", strokeOpacity=1))
    
    # Create the bar graph
    base_vel = alt.Chart(summary_stats_vel, width=800, height=400).mark_bar(color="green").encode(
            x=alt.X("date:T", title="Velocity Category"),
            y=alt.Y("total_velocity_duration:Q", title="Hours"),
            # color="Velocity_Category:N",
            color=alt.condition(brush, 'Velocity_Category:N', alt.value('lightgray')),
            # column="Velocity_Category:N",
            tooltip=["date:T", "Velocity_Category:N", "total_velocity_duration:Q"],
    # ).add_selection(week_selection
    ).properties(
        title="Daily Velocity Over/Under 8 ft/s Duration Summary"
    )
    upper_vel = base_vel.mark_bar(width=alt.RelativeBandSize(0.7)).encode(
        alt.X('date:T').scale(domain=brush)
    )
    lower_vel = base_vel.properties(
        height=90
    ).add_params(brush)
    vel_bar_chart = upper_vel & lower_vel
    # vel_chart_with_avg = vel_bar_chart + avg_line_velocity
    base_gate = alt.Chart(summary_stats_dgl, width=800, height=400).mark_bar(
        color="steelblue",
        # width=alt.RelativeBandSize(0.7)
    ).encode(
        x=alt.X("date:T", title="Gate Status"), 
        y=alt.Y("total_gate_duration:Q", title="Hours"),
        color=alt.condition(brush, 'DGL:N', alt.value('lightgray')),
        tooltip=["date:T","DGL:N", "total_gate_duration:Q"]
    ).properties(
        title="Daily Gate Status Duration Summary"
    )
    # gate_bar_chart = alt.Chart(summary_stats_dgl).mark_bar(color="steelblue").encode(
    #         x=alt.X("date", title="Gate Status",axis=alt.Axis()),
    #         y=alt.Y("total_gate_duration:Q", title="Hours"),
    #         color="DGL:N",
    #         # column="DGL:N",
    #         tooltip=["DGL:N", "total_gate_duration:Q"],
    # # ).add_selection(week_selection
    # ).properties(
    #     title="Weekly Gate Status Duration Summary", width=200, height=400
    # )
    
    upper_gate = base_gate.mark_bar(width=alt.RelativeBandSize(0.7)).encode(
        alt.X('date:T').scale(domain=brush)
    )
    lower_gate = base_gate.properties(
        height=90
    ).add_params(brush)
    
    gate_bar_chart = upper_gate & lower_gate
    # gate_chart_with_avg = gate_bar_chart + avg_line_gate
    # combined_bar_charts = alt.hconcat(
    #     gate_bar_chart,
    #     vel_bar_chart 
    # )
    # combined_bar_charts
    combined_bar_charts = alt.vconcat(
        gate_bar_chart,
        vel_bar_chart
        )

    # Render the chart in Streamlit
    # st.set_page_config(layout="wide")
    # st.write("Summary Stats for Velocity")
    # st.dataframe(summary_stats_vel)

    # st.write("Summary Stats for Gate Status")
    # st.dataframe(summary_stats_dgl)
    
    st.altair_chart(combined_bar_charts, use_container_width=True, theme=None)

    # with st.container():
    #     st.altair_chart(combined_bar_charts)
    # st.altair_chart(vel_bar_chart, use_container_width=True)

# Render the gate status chart
    # st.altair_chart(gate_bar_chart, use_container_width=True)
else:
    st.write("Please upload a CSV file to see the visualization.")
