import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

with open("project_Tips/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

uploaded_file= r"project_Tips/data/tips.csv"

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
st.header("*Exploraty Data Analysis*")



st.subheader("**1 - Univariate Analysis**")

# Select Column
selected_column = st.selectbox("Select a Column for Analysis", data.columns)

if selected_column:
    st.markdown(f"#### Analysis for {selected_column}")  
    col1, col2 = st.columns(2)
    with col1:
        # Additional Stats for Numeric Columns
        if pd.api.types.is_numeric_dtype(data[selected_column]):
            # Calculate statistics
            column_data = data[selected_column].dropna()  # Remove NaNs
            mean_val = column_data.mean()
            median_val = column_data.median()
            std_val = column_data.std()
            min_val = column_data.min()
            max_val = column_data.max()
            mode_val = column_data.mode().values
            with col1:
                st.write("**Histogram plot parameters**")
                # Create the plot
                fig, ax = plt.subplots()
                ax.set_title(f"Distribution of {selected_column}", fontsize=16, fontweight="bold")
                ax.set_xlabel(f"{selected_column}", fontsize=12)
                ax.set_ylabel("Frequency", fontsize=12)
                ax.grid(visible=True, linestyle="--", alpha=0.7)
                log_scale = st.checkbox("Log Scale", key="dist_log_scale")
                show_kde = st.checkbox("Show KDE", value=True, key="dist_kde")
                plot_color = st.color_picker("Pick Plot Color", "#4CAF50", key="dist_color")
                bin_size = st.slider("Number of Bins", min_value=5, max_value=50, value=20, key="dist_bins")
                # Summary Statistics
                st.write("**Summary Statistics:**")
                st.write(data[selected_column].describe())
            with col2:
                st.write("Histogram plot")
                sns.histplot(data[selected_column],bins=bin_size,color=plot_color,kde=show_kde,log_scale=log_scale,edgecolor="black")
                # Add statistical annotations
                stats_text = (
                    f"Mean: {mean_val:.2f}\n"
                    f"Median: {median_val:.2f}\n"
                    f"Std Dev: {std_val:.2f}\n"
                    f"Min: {min_val:.2f}\n"
                    f"Max: {max_val:.2f}\n"
                    f"Mode:{mode_val}"
                )
                ax.text(
                    0.95, 0.95, stats_text,
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
                )
                st.pyplot(fig)


                # Box Plot
                st.write("**Box Plot:**")
                fig, ax = plt.subplots()
                sns.boxplot(x=data[selected_column], color="lightgreen")
                st.pyplot(fig)

                # line Plot
                st.write("**line Plot:**")
                fig, ax = plt.subplots()
                sns.lineplot(data[selected_column])
                st.pyplot(fig)

    # Analysis for Categorical Columns
        else:
            with col1:
                # Summary Statistics
                st.write("**Summary Statistics:**")
                st.write(data[selected_column].describe())
                st.write(f"Mode: {data[selected_column].mode().values}")
                st.write("**Value Counts:**")
                st.write(data[selected_column].value_counts())
            with col2:

                # Bar Plot
                st.write("**Bar Plot:**")
                fig, ax = plt.subplots()
                sns.countplot(x=selected_column, data=data, order=data[selected_column].value_counts().index)
                st.pyplot(fig)
                
                # Pie Chart
                st.write("**Pie Chart:**")
                fig, ax = plt.subplots()
                data[selected_column].value_counts().plot.pie(autopct='%1.2f%%')
                st.pyplot(fig)


st.subheader("2 - Bivariate Analysis")

# Initialize session state for visualization type and user selections
if "visu1" not in st.session_state:
    st.session_state.visu1 = False
if "visu2" not in st.session_state:
    st.session_state.visu2 = False
if "visu3" not in st.session_state:
    st.session_state.visu3 = False

# Create three columns for different bivariate analyses
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Numeric vs Numeric"):
        st.session_state.visu1 = True
        st.session_state.visu2 = False
        st.session_state.visu3 = False
with col2:
    if st.button("Categorical vs Categorical"):
        st.session_state.visu1 = False
        st.session_state.visu2 = True
        st.session_state.visu3 = False
with col3:
    if st.button("Categorical VS Numerical"):
        st.session_state.visu1 = False
        st.session_state.visu2 = False
        st.session_state.visu3 = True

# **1. Numeric vs Numeric**
if st.session_state.visu1:
    with col1:
    
        x_var = st.selectbox("Select X-axis Variable (Numeric)", num_data.columns, key="num_x")
        y_var = st.selectbox("Select Y-axis Variable (Numeric)", num_data.columns, key="num_y")

        if x_var and y_var:
            st.subheader(f"Analysis between {x_var} and {y_var}")
            correlation = data[[x_var, y_var]].corr().iloc[0, 1]
            st.write(f"**Correlation Coefficient:** {correlation:.2f}")

    fig, ax = plt.subplots()
    sns.scatterplot(x=x_var, y=y_var, data=data)
    st.pyplot(fig)

# **2. Categorical vs Categorical**
if st.session_state.visu2:
    with col2:
        x_var = st.selectbox("Select First Categorical Variable", cat_data.columns, key="cat_x")
        y_var = st.selectbox("Select Second Categorical Variable", cat_data.columns, key="cat_y")
    fig, ax = plt.subplots()
    st.write(f"Bar plot between {x_var} and {y_var}")
    with col2:
        hue_axis = st.selectbox("Select Hue (Categorical)", cat_data.columns)
    sns.barplot(x=x_var,y=y_var,hue=hue_axis,data=data)
    st.pyplot(fig)

    if x_var and y_var:
            st.write(f"Cross-tabulation between {x_var} and {y_var}")
            crosstab = pd.crosstab(data[x_var], data[y_var])
            st.write(crosstab)
    fig, ax = plt.subplots()
    sns.heatmap(crosstab, annot=True, cmap="coolwarm", fmt="d")
    st.pyplot(fig)
    
    

# **3. Categorical vs Numeric**
if st.session_state.visu3:
    with col3:
        cat_var = st.selectbox("Select Categorical Variable", cat_data.columns, key="cat_var")
        num_var = st.selectbox("Select Numeric Variable", num_data.columns, key="num_var")
        hue_axis = st.selectbox("Select Hue (Categorical)", cat_data.columns)


    if cat_var and num_var:
        st.write(f"Box Plot between {cat_var} and {num_var}")
    fig, ax = plt.subplots()
    sns.boxplot(x=cat_var, y=num_var,hue=hue_axis, data=data,palette="flare")
    st.pyplot(fig)
        
    st.write(f"Bar plot between {cat_var} and {num_var}")
    fig, ax = plt.subplots()
    sns.barplot(x=cat_var,y=num_var,hue=hue_axis,data=data)
    st.pyplot(fig)


st.subheader("3 - Multivariate Analysis")

if "f1" not in st.session_state:
    st.session_state.f1 = False
if "f2" not in st.session_state:
    st.session_state.f2 = False
if "f3" not in st.session_state:
    st.session_state.f3 = False

col1, col2,col3 = st.columns(3)
with col1:
    if st.button("Generate Correlation Heatmap"):
        st.session_state.f1 = True
        st.session_state.f2 = False
        st.session_state.f3 = False
with col2:
    if st.button("Generate Pair Plot"):
        st.session_state.f1 = False
        st.session_state.f2 = True
        st.session_state.f3 = False
with col3:
    if st.button("Generate Multivariate Box Plot"):
        st.session_state.f1 = False
        st.session_state.f2 = False
        st.session_state.f3 = True
if st.session_state.f1:
    # Correlation Heatmap for Numeric Data
    if num_data.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(num_data.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        st.pyplot(fig)
    else:
        st.warning("Not enough numeric columns for correlation heatmap.")

# Pair Plot for Multiple Variables
if st.session_state.f2:
    if num_data.shape[1] >= 2:
        fig = sns.pairplot(num_data, diag_kind="kde", markers="o", corner=True)
        st.pyplot(fig)
    else:
        st.warning("Not enough numeric columns for pair plot.")


        # Multivariate Box Plot for Categorical Influence
num_cols=num_data.columns
if st.session_state.f3:
    with col3:
        cat_col = st.selectbox("Select a Categorical Column (Hue)", cat_data.columns if not cat_data.empty else [None])
        num_cols = st.multiselect("Select Numeric Columns", num_data.columns)

if st.session_state.f3 and cat_col and len(num_cols) >= 2:
    fig, axs = plt.subplots(len(num_cols), 1, figsize=(10, 5 * len(num_cols)))
    for i, col in enumerate(num_cols):
        sns.boxplot(x=cat_col, y=col, data=data, ax=axs[i])
        axs[i].set_title(f"Box Plot of {col} by {cat_col}")
    st.pyplot(fig)
elif len(num_cols) < 2:
    st.warning("Please select at least two numeric columns for multivariate box plots.")







# ******* others plot *******
st.header("*linear regression *")
st.subheader("*Joint plot*")
st.write("select coloums for joint plot")
x_axis = st.selectbox("Select X-axis", num_data.columns)
y_axis = st.selectbox("Select Y-axis", num_data.columns)
st.write("select kind for joint plot")
kind_axis = st.selectbox("Select kind", ["reg", "resid"])
with sns.axes_style("white"):
    jointplot_fig = sns.jointplot(x=x_axis, y=y_axis, data=data, kind=kind_axis)
st.pyplot(jointplot_fig)





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
        selected_column = st.selectbox("Select a Numeric Column", num_data.columns, key="dist-col")
        log_scale1 = st.checkbox("Log Scale", key="dist-log_scale")
        bin_size1 = st.slider("Number of Bins", min_value=5, max_value=50, value=20, key="dist-bin_size")
        show_kde1 = st.checkbox("Show KDE", value=True, key="dist-kde")
        plot_color1 = st.color_picker("Pick Plot Color", "#4CAF50", key="dist-color")
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
            bins=bin_size1,
            kde=show_kde1,
            log_scale=log_scale1,
            color=plot_color1,
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



st.header("*Missing Data Analysis*")

if st.button("Generate Missing Data Heatmap"):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data.isnull(), cbar=False, cmap="viridis", ax=ax)
    ax.set_title("Missing Data Heatmap")
    st.pyplot(fig)


