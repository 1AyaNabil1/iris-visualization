import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

def load_data(filepath):
    """Load the Iris dataset from a CSV file."""
    df = pd.read_csv(data\iris.csv)
    return df

def plot_histograms(df):
    """Plot histograms for numerical features."""
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    colors = ['tomato', 'limegreen', 'deepskyblue', 'orange']
    
    for i, column in enumerate(df.columns[1:5]):
        sns.histplot(df[column], ax=ax[i], bins=20, color=colors[i])
    
    plt.show()

def plot_kde(df):
    """Plot KDE distributions for each feature by species."""
    iris = df.copy()
    iris['Species'] = pd.Categorical(iris['Species'])
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 6))
    for col, ax in zip(df.columns[1:5], axs.flat):
        sns.histplot(data=iris, x=col, kde=True, hue='Species', common_norm=False, legend=ax == axs[0,0], ax=ax)
    
    plt.tight_layout()
    plt.show()

def plot_scatter(df):
    """Plot scatter plots for sepal and petal dimensions."""
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    df.plot(kind="scatter", x="SepalLengthCm", y="SepalWidthCm", ax=ax[0], color='limegreen')
    df.plot(kind="scatter", x="PetalLengthCm", y="PetalWidthCm", ax=ax[1], color='skyblue')
    plt.show()

def plot_boxplots(df):
    """Plot boxplots for each feature grouped by species."""
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    colors = ['tomato', 'limegreen', 'deepskyblue', 'orange']
    
    for i, column in enumerate(df.columns[1:5]):
        sns.boxplot(x="Species", y=column, data=df, ax=ax[i], color=colors[i])
    
    plt.show()

def plot_pie_chart(df):
    """Plot a pie chart showing the species distribution."""
    plt.figure(figsize=(10, 8))
    df['Species'].value_counts().plot.pie(explode=[0.1, 0.1, 0.1], autopct='%1.1f%%', shadow=True)
    plt.title("Iris Species Distribution")
    plt.show()

def plot_bubble_chart():
    """Plot a bubble chart for species categories."""
    fig = go.Figure(data=[go.Scatter(
        x=["setosa", "virginica", "versicolor"],
        y=["PetalWidthCm", "PetalLengthCm", "SepalWidthCm", "SepalLengthCm"],
        mode='markers',
        marker=dict(color=[120, 125, 130, 135, 140, 145], size=[15, 30, 55, 70, 90, 110], showscale=True)
    )])
    fig.show()

def main():
    """Main function to load data and call visualization functions."""
    df = load_data("data/Iris.csv")
    plot_histograms(df)
    plot_kde(df)
    plot_scatter(df)
    plot_boxplots(df)
    plot_pie_chart(df)
    plot_bubble_chart()

if __name__ == "__main__":
    main()
