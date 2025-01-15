import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

uploaded_file= r"Project_Car_crashes/data/car_crashes.csv"



# colors are: red, green, blue,black, white, cyan,dark blue,navyblue,
# navygreen,purple,orange,
# color: #FF0000; /* Red */
# color: #00FF00; /* Green */
# color: #0000FF; /* Blue */
# color: #FFFF00; /* Yellow */
# color: #FFA500; /* Orange */
# color: #800080; /* Purple */

st.markdown(
    """
    <style>
    h1 {
        color:Purple;
        font-family: 'Arial', sans-serif;
        text-align: center;
    }
    h2 {
        color: cyan;
        font-family: 'Courier New', monospace;
    }
    h3 {
    color: Green;
    font-family: 'Arial', sans-serif;
    # text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    button[data-testid="stButton"] {
        background-color:White 
        color:#0000FF 
        padding: 10px 20px; /* Padding */
        font-size: 16px; /* Font size */
        border: #FF0000 
        border-radius: 8px; /* Rounded corners */
        cursor: pointer; /* Pointer cursor */
    }
    button[data-testid="stButton"]:hover {
        background-color: #45a049; /* Darker green on hover */
    }
    </style>
    """,
    unsafe_allow_html=True
)


#  streamlit run tips_dashboard.py 

if uploaded_file:
    data1 = pd.read_csv(uploaded_file)
    data = data1.copy()
    num_data = data.select_dtypes("number")
    cat_data = data.select_dtypes("object")
else:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()

st.title("Tips Data Dashboard")

st.sidebar.title("Data information dashboard")
option = st.sidebar.radio(
    "Choose an action:",
    ["DataFrame Info","Data Overview", "Summary Statistics", "Unique Values"],
)
if option == "DataFrame Info":
    st.header("*Data information*")
    st.write(f"Number of rows: {data.shape[0]}")
    st.write(f"Number of columns: {data.shape[1]}")
    st.write(f"Column names: {list(data.columns)}")
    st.write("Column data types:")
    st.write(data.dtypes)
elif option == "Data Overview":
    st.header("*Data overview*")
    st.dataframe(data.head())
elif option == "Summary Statistics":
    st.header("*Summary Statistics*")
    st.dataframe(data.describe())
elif option == "Unique Values":
    st.header("*Unique Values*")
    st.dataframe(data.nunique())

#Data Visualization
st.header("*Data Visualization*")

# Initialize session state for visualization type and user selections
if "visu1" not in st.session_state:
    st.session_state.visu1 = False
if "visu2" not in st.session_state:
    st.session_state.visu2 = False
if "visu3" not in st.session_state:
    st.session_state.visu3 = False
# Layout for Visualization Type Selection
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.button("Numeric Data Visualization"):
        st.session_state.visu1 = True
        st.session_state.visu2 = False
        st.session_state.visu3 = False
with col2:
    if st.button("Categorical Data Visualization"):
        st.session_state.visu1 = False
        st.session_state.visu2 = True
        st.session_state.visu3 = False
with col3:
    if st.button("Categorical VS Numerical"):
        st.session_state.visu1 = False
        st.session_state.visu2 = False
        st.session_state.visu3 = True
#Numerical Visualization
if st.session_state.visu1:
    with col1:
        st.subheader("Numeric Data Visualization")
        st.write("select coloums for visualizations")
        x_axis = st.selectbox("Select X-Axis (Numeric)", num_data.columns)
        y_axis = st.selectbox("Select Y-Axis (Numeric)", num_data.columns)
        plot_tybe=st.radio("select plot type: ",["Histogram plot","Line plot","Scatter plot","Box plot"])
    
    if plot_tybe=="Histogram plot":
        fig=plt.figure()
        st.title("Histogram plot")
        sns.histplot(data[x_axis],bins=5,color="blue",edgecolor="black")
        st.pyplot(fig)


    elif plot_tybe=="Line plot":
        fig=plt.figure()
        st.title("Line plot")
        hue_axis = st.selectbox("Select Hue (Categorical)", cat_data.columns)
        sns.lineplot(x=x_axis,y=y_axis,hue=hue_axis,data=data)
        st.pyplot(fig)

    elif plot_tybe=="Scatter plot":
        fig=plt.figure()
        st.title("Scatter plot")
        hue_axis = st.selectbox("Select Hue (Categorical)", cat_data.columns)
        sns.scatterplot(x=x_axis,y=y_axis,hue=hue_axis,data=data)
        st.pyplot(fig)

    elif plot_tybe=="Box plot":
        fig=plt.figure()
        st.title("Box plot")
        hue_axis = st.selectbox("Select Hue (Categorical)", cat_data.columns)
        sns.boxplot(y=x_axis,hue=hue_axis,data=data,palette="flare")
        plt.tight_layout()
        st.pyplot(fig)
# categorical data visualization
elif st.session_state.visu2:
    with col2:
        st.subheader("Categorical Data Visualization")
        st.write("select coloums for visualizations")
        x_axis = st.selectbox("Select X-Axis (Categorical)", cat_data.columns)
        y_axis = st.selectbox("Select Y-Axis (Categorical)", cat_data.columns)
        plot_tybe=st.radio("select plot type: ",["Histogram plot","Bar plot","Box plot","Pie plot","Count plot"])

    if plot_tybe=="Histogram plot":
        fig=plt.figure()
        st.title("Histogram plot")
        sns.histplot(data[x_axis],bins=5,color="blue",edgecolor="black")
        st.pyplot(fig)

    elif plot_tybe=="Bar plot":
        fig=plt.figure()
        st.title("Bar plot")
        hue_axis = st.selectbox("Select Hue (Categorical)", cat_data.columns)
        sns.barplot(x=x_axis,y=y_axis,hue=hue_axis,data=data)
        st.pyplot(fig)

    elif plot_tybe=="Box plot":
        fig=plt.figure()
        st.title("Box plot")
        hue_axis = st.selectbox("Select Hue (Categorical)", cat_data.columns)
        sns.boxplot(x=x_axis,y=y_axis,hue=hue_axis,data=data,palette="flare")
        plt.tight_layout()
        st.pyplot(fig)

    elif plot_tybe=="Pie plot":
        fig=plt.figure()
        st.title("Pie plot")
        plt.pie(data[x_axis].value_counts(),autopct='%1.2f%%',labels=data[x_axis].unique())
        st.pyplot(fig)

    elif plot_tybe=="Count plot":
        fig=plt.figure()
        st.title("Count plot")
        hue_axis = st.selectbox("Select Hue (Categorical)", cat_data.columns)
        sns.countplot(x=x_axis,data=data,hue=hue_axis)
        st.pyplot(fig)
# Categorical VS Numerical
elif st.session_state.visu3:
    with col3:
        st.subheader("Categorical VS Numerical")
        st.write("select coloums for visualizations")
        x_axis = st.selectbox("Select X-Axis (Categorical)", cat_data.columns)
        y_axis = st.selectbox("Select Y-Axis (Numeric)", num_data.columns)
        plot_tybe=st.radio("select plot type: ",["Bar plot","Box plot","Count plot"])

    if plot_tybe=="Bar plot":
        fig=plt.figure()
        st.title("Bar plot")
        hue_axis = st.selectbox("Select Hue (Categorical)", cat_data.columns)
        sns.barplot(x=x_axis,y=y_axis,hue=hue_axis,data=data)
        st.pyplot(fig)

    elif plot_tybe=="Box plot":
        fig=plt.figure()
        st.title("Box plot")
        hue_axis = st.selectbox("Select Hue (Categorical)", cat_data.columns)
        sns.boxplot(x=x_axis,y=y_axis,hue=hue_axis,data=data,palette="flare")
        plt.tight_layout()
        st.pyplot(fig)

    elif plot_tybe=="Pie plot":
        fig=plt.figure()
        st.title("Pie plot")
        plt.pie(data[x_axis].value_counts(),autopct='%1.2f%%',labels=data[x_axis].unique())

        st.pyplot(fig)

    elif plot_tybe=="Count plot":
        fig=plt.figure()
        st.title("Count plot")
        hue_axis = st.selectbox("Select Hue (Categorical)", cat_data.columns)
        sns.countplot(x=x_axis,data=data,hue=hue_axis)
        st.pyplot(fig)
    
# ******* others plot *******
st.header("*Other Plots*")


if "f1" not in st.session_state:
    st.session_state.f1 = False
if "f2" not in st.session_state:
    st.session_state.f2 = False

col1, col2 = st.columns([2,2])

with col1:
    if st.button("Joint plot"):
        st.session_state.f1 = True
        st.session_state.f2 = False
with col2:
    if st.button("Pair Plot"):
        st.session_state.f1 = False
        st.session_state.f2 = True

#jointplot
if st.session_state.f1:
    with col1:
        st.subheader("*Joint plot*")
        st.write("select coloums for joint plot")
        x_axis = st.selectbox("Select X-axis", num_data.columns)
        y_axis = st.selectbox("Select Y-axis", num_data.columns)
        st.write("select kind for joint plot")
        kind_axis = st.selectbox("Select kind", ["reg", "resid"])
    with sns.axes_style("white"):
        jointplot_fig = sns.jointplot(x=x_axis, y=y_axis, data=data, kind=kind_axis)
    st.pyplot(jointplot_fig)
# Pair Plot Section

elif st.session_state.f2:
    with col2:
        st.subheader("*Pair Plot*")
    if num_data.empty:
        st.warning("No numeric data available to generate a pair plot.")
    else:
        with sns.axes_style("white"):
            pairplot_fig = sns.pairplot(num_data) 
            st.pyplot(pairplot_fig)  




## **** Statistics Analysis ***
st.header("*Statistics Analysis*")

# Initialize session state for toggling analysis
if "stat1" not in st.session_state:
    st.session_state.stat1 = False
if "stat2" not in st.session_state:
    st.session_state.stat2 = False
if "stat3" not in st.session_state:
    st.session_state.stat3 = False

# Create three buttons in separate columns
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.button("Distribution Plot"):
        st.session_state.stat1 = True
        st.session_state.stat2 = False
        st.session_state.stat3 = False
with col2:
    if st.button("Grouped Statistics"):
        st.session_state.stat1 = False
        st.session_state.stat2 = True
        st.session_state.stat3 = False
with col3:
    if st.button("Correlation Heat Map"):
        st.session_state.stat1 = False
        st.session_state.stat2 = False
        st.session_state.stat3 = True

# **1. Distribution Plot**
if st.session_state.stat1:
    with col1:
        st.subheader("*Distribution Plot*")
        selected_column = st.selectbox("Select a Numeric Column", num_data.columns, key="dist_col")
        log_scale = st.checkbox("Log Scale", key="dist_log_scale")
        bin_size = st.slider("Number of Bins", min_value=5, max_value=50, value=20, key="dist_bins")
        show_kde = st.checkbox("Show KDE", value=True, key="dist_kde")
        plot_color = st.color_picker("Pick Plot Color", "#4CAF50", key="dist_color")
    if selected_column:
        # Calculate statistics
        column_data = data[selected_column].dropna()  # Remove NaNs
        mean_val = column_data.mean()
        median_val = column_data.median()
        std_val = column_data.std()
        min_val = column_data.min()
        max_val = column_data.max()

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(
            column_data,
            bins=bin_size,
            kde=show_kde,
            log_scale=log_scale,
            color=plot_color,
            ax=ax
        )
        ax.set_title(f"Distribution of {selected_column}", fontsize=16, fontweight="bold")
        ax.set_xlabel(f"{selected_column}", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.grid(visible=True, linestyle="--", alpha=0.7)

        # Add statistical annotations
        stats_text = (
            f"Mean: {mean_val:.2f}\n"
            f"Median: {median_val:.2f}\n"
            f"Std Dev: {std_val:.2f}\n"
            f"Min: {min_val:.2f}\n"
            f"Max: {max_val:.2f}"
        )
        ax.text(
            0.95, 0.95, stats_text,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

        # Display the plot
        st.pyplot(fig)
    else:
        st.warning("Please select a numeric column to generate the plot.")

# **2. Grouped Statistics**
elif st.session_state.stat2:
    with col2:
        st.subheader("*Grouped Statistics*")
        group_column = st.selectbox("Select a Categorical Column for Grouping", cat_data.columns, key="group_col")
        stat_column = st.selectbox("Select a Numeric Column for Analysis", num_data.columns, key="stat_col")

    grouped_stats = data.groupby(group_column)[stat_column].describe()
    st.dataframe(grouped_stats)

# **3. Correlation Heat Map**
elif st.session_state.stat3:
    with col3:
        st.subheader("*Correlation Heat Map*")
    if num_data.empty:
        st.warning("No numeric data available to generate a heatmap.")
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(num_data.corr(), annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap", fontsize=16, fontweight="bold")
        st.pyplot(fig)



st.header("Missing Data Analysis")

if st.button("Generate Missing Data Heatmap"):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data.isnull(), cbar=False, cmap="viridis", ax=ax)
    ax.set_title("Missing Data Heatmap")
    st.pyplot(fig)

