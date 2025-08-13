import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set up the Streamlit app
st.set_page_config(page_title="Fintech Sentiment Dashboard", layout="wide")
st.title(" Smart Fintech Sentiment Explorer")
st.markdown("Gain insights from app store reviews for major fintech platforms.")

# Load the data
data_path = "data/fintech_reviews_sentiment.csv"
if not os.path.exists(data_path):
    st.error("Data file not found. Please run sentiment analysis script first.")
    st.stop()

@st.cache_data
def load_data():
    return pd.read_csv(data_path)

df = load_data()

# Sidebar filters
st.sidebar.header("📁 Filters")
app_names = df['app_name'].unique().tolist()
selected_app = st.sidebar.selectbox("Select an App", app_names)

filtered_df = df[df['app_name'] == selected_app]

# Filter by version
versions = sorted(filtered_df['reviewCreatedVersion'].dropna().unique().tolist())
selected_version = st.sidebar.selectbox("Select App Version (optional)", ["All"] + versions)
if selected_version != "All":
    filtered_df = filtered_df[filtered_df['reviewCreatedVersion'] == selected_version]

# Chart type selection
st.sidebar.markdown("---")
st.sidebar.header(" Visualization Selector")
chart_choice = st.sidebar.selectbox(
    "Choose a Chart to Display",
    options=[
        "Pie Chart",
        "Bar Chart",
        "Box Plot",
        "Line Chart",
        "Heatmap",
        "All Visualizations"
    ]
)

# Sidebar QnA block
st.sidebar.markdown("---")
st.sidebar.header(" Ask a Question")
question = st.sidebar.text_input("E.g., How many negative reviews for Google Pay?")

if question:
    st.subheader("🔍 Answer to your Question:")
    if "negative" in question.lower() and "google pay" in question.lower():
        count = df[(df['app_name'] == "Google Pay") & (df['sentiment_label'] == "negative")].shape[0]
        st.markdown(f" **Google Pay has {count} negative reviews**.")
    else:
        st.markdown("Question not recognized. Try a different query.")

# ------------------------------------------
# Chart Data Preparation
# ------------------------------------------
sentiment_counts = filtered_df['sentiment_label'].value_counts()
sentiment_df = sentiment_counts.reset_index()
sentiment_df.columns = ['sentiment_label', 'count']
filtered_df['at'] = pd.to_datetime(filtered_df['at'])

# ------------------------------------------
# Conditional Chart Rendering
# ------------------------------------------

st.header(f" Visualization for {selected_app}")

if chart_choice == "Pie Chart":
    st.subheader("Sentiment Distribution (Pie Chart)")
    fig = px.pie(
        sentiment_df,
        names='sentiment_label',
        values='count',
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3,
        title="Sentiment Share"
    )
    st.plotly_chart(fig, use_container_width=True)

elif chart_choice == "Bar Chart":
    st.subheader("Bar Chart of Sentiment Counts")
    fig = px.bar(
        sentiment_df,
        x='sentiment_label',
        y='count',
        labels={'sentiment_label': 'Sentiment', 'count': 'Count'},
        color='sentiment_label',
        title=f"Sentiment Count for {selected_app}"
    )
    st.plotly_chart(fig, use_container_width=True)

elif chart_choice == "Box Plot":
    st.subheader("Box Plot of Sentiment Scores")
    fig = px.box(
        filtered_df,
        x="sentiment_label",
        y="sentiment_score",
        color="sentiment_label",
        title="Sentiment Score Distribution by Label"
    )
    st.plotly_chart(fig, use_container_width=True)

elif chart_choice == "Line Chart":
    st.subheader("Reviews Over Time")
    daily_counts = filtered_df.groupby(filtered_df['at'].dt.date).size().reset_index(name='count')
    daily_counts.rename(columns={'at': 'date'}, inplace=True)
    fig = px.line(
        daily_counts,
        x='date',
        y='count',
        title="Review Volume Over Time"
    )
    st.plotly_chart(fig, use_container_width=True)

elif chart_choice == "Heatmap":
    st.subheader("Heatmap of Versions and Sentiments")
    heatmap_df = filtered_df.groupby(['reviewCreatedVersion', 'sentiment_label']).size().unstack().fillna(0)
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_df.values,
        x=heatmap_df.columns,
        y=heatmap_df.index,
        colorscale='YlGnBu'
    ))
    fig.update_layout(
        title="Version-wise Sentiment Heatmap",
        xaxis_title="Sentiment",
        yaxis_title="App Version"
    )
    st.plotly_chart(fig, use_container_width=True)

elif chart_choice == "All Visualizations":
    st.subheader("1. Sentiment Distribution (Pie Chart)")
    fig_pie = px.pie(
        sentiment_df,
        names='sentiment_label',
        values='count',
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3,
        title="Sentiment Share"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("2. Bar Chart of Sentiment Counts")
    fig_bar = px.bar(
        sentiment_df,
        x='sentiment_label',
        y='count',
        labels={'sentiment_label': 'Sentiment', 'count': 'Count'},
        color='sentiment_label',
        title=f"Sentiment Count for {selected_app}"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("3. Box Plot of Sentiment Scores")
    fig_box = px.box(
        filtered_df,
        x="sentiment_label",
        y="sentiment_score",
        color="sentiment_label",
        title="Sentiment Score Distribution by Label"
    )
    st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("4. Reviews Over Time")
    daily_counts = filtered_df.groupby(filtered_df['at'].dt.date).size().reset_index(name='count')
    daily_counts.rename(columns={'at': 'date'}, inplace=True)
    fig_line = px.line(
        daily_counts,
        x='date',
        y='count',
        title="Review Volume Over Time"
    )
    st.plotly_chart(fig_line, use_container_width=True)

    st.subheader("5. Heatmap of Versions and Sentiments")
    heatmap_df = filtered_df.groupby(['reviewCreatedVersion', 'sentiment_label']).size().unstack().fillna(0)
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_df.values,
        x=heatmap_df.columns,
        y=heatmap_df.index,
        colorscale='YlGnBu'
    ))
    fig_heatmap.update_layout(
        title="Version-wise Sentiment Heatmap",
        xaxis_title="Sentiment",
        yaxis_title="App Version"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

# ------------------------------------------
# Footer
# ------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center;">
        Developed by <strong>Pragathi</strong> using <a href="https://streamlit.io/" target="_blank"><code>Streamlit</code></a> 🚀
    </div>
    """,
    unsafe_allow_html=True
)
