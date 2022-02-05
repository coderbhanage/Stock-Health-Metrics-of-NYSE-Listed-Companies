# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt


# %%
from sklearn.feature_selection import RFE
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

# for splitting the data into train and test samples
from sklearn.model_selection import train_test_split

# for model evaluation metrics
from sklearn.metrics import classification_report

# for encoding categorical features from strings to number arrays
from sklearn.preprocessing import LabelEncoder

import plotly.express as px  # for data visualization
import plotly.graph_objects as go  # for data visualization

# Differnt types of Naive Bayes Classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import BernoulliNB


# %%
df = pd.read_csv("./dataset/final_data.csv")
df = df.drop(columns=['Estimated Shares Outstanding'])


# %%
df.sample()


# %%
def mfunc(X, y, typ):

    # Create training and testing samples
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1)

    # Fit the model
    
    model = typ
    clf = model.fit(X_train, y_train)

    # Predict class labels on a test data
    pred_labels = model.predict(X_test)

    # Print model attributes
    print('Classes: ', clf.classes_)  # class labels known to the classifier
    if str(typ) == 'GaussianNB()':
        # prior probability of each class.
        print('Class Priors: ', clf.class_prior_)
    else:
        # log prior probability of each class.
        print('Class Log Priors: ', clf.class_log_prior_)

    # Use score method to get accuracy of the model
    print('--------------------------------------------------------')
    
    score = model.score(X_test, y_test)
    print('Accuracy Score: ', score)
    print('--------------------------------------------------------')

    # Look at classification report to evaluate the model
    print(classification_report(y_test, pred_labels))

    # Return relevant data for chart plotting
    return X_train, X_test, y_train, y_test, clf, pred_labels


# %%
X = df.loc[:, ["stock_val_10", "Gross Margin"]]
y = df.loc[:, df.columns=="rating"].values.ravel()

# sc = StandardScaler()
# X = sc.fit_transform(X)
# X = sc.transform(X)
# X = normalize(X, norm='l2')


# %%
X_train, X_test, y_train, y_test, clf, pred_labels, = mfunc(X, y, GaussianNB())


# %%
# X_train, X_test, y_train, y_test, clf, pred_labels, = mfunc(X, y, BernoulliNB())


# %%
# Specify a size of the mesh to be used
mesh_size = 5
margin = 1

# Create a mesh grid on which we will run our model
x_min, x_max = X.iloc[:, 0].fillna(X.mean()).min() - margin, X.iloc[:, 0].fillna(X.mean()).max() + margin
y_min, y_max = X.iloc[:, 1].fillna(X.mean()).min() - margin, X.iloc[:, 1].fillna(X.mean()).max() + margin
xrange = np.arange(x_min, x_max, mesh_size)
yrange = np.arange(y_min, y_max, mesh_size)
xx, yy = np.meshgrid(xrange, yrange)

# Create classifier, run predictions on grid
Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

# Specify traces 
trace_specs = [
    #[X_train, y_train, 0, 'Train', 'brown'],
    #[X_train, y_train, 1, 'Train', 'aqua'],
    [X_test, y_test, 0, 'Test', 'red'],
    [X_test, y_test, 1, 'Test', 'blue']
]

# Build the graph using trace_specs from above
fig = go.Figure(data=[
    go.Scatter(
        x=X[y==label].iloc[:, 0], y=X[y==label].iloc[:, 1],
        name=f'{split} data, Actual Class: {label}',
        mode='markers', marker_color=marker
    )
    for X, y, label, split, marker in trace_specs
])

# Update marker size
fig.update_traces(marker_size=2, marker_line_width=0)

# Update axis range
fig.update_xaxes(range=[-1600, 1500])
fig.update_yaxes(range=[0,345])

# Update chart title and legend placement
fig.update_layout(title_text="Decision Boundary for Naive Bayes Model", 
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

# Add contour graph
fig.add_trace(
    go.Contour(
        x=xrange,
        y=yrange,
        z=Z,
        showscale=True,
        colorscale='magma',
        opacity=1,
        name='Score',
        hoverinfo='skip'
    )
)

fig.show()

