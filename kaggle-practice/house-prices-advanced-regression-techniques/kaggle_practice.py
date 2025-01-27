# %%
# import packages
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
from IPython.display import display, HTML


# import data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
# %%
# Perform basic cleaning and feature explorations
# look at summary stats
from dataprep.eda import create_report

# create_report(train_df).show_browser()
# %%
# from the report above we can change some datatypes to categorical based on what was mentioned on the dataset's documentation. I also looked to see if there's a non-linear relationship between the target and the feature. Few unique values.

for col in (
    "MSSubClass",
    "MoSold",
    "YrSold",
    "OverallCond",
):
    train_df[col] = train_df[col].astype("str")
    test_df[col] = test_df[col].astype("str")


# %%
# checking features that highly correlated to target like GrLivArea, TotalBsmtSF, OverallQual, and 1stFlrSF
# 2 points stood out as outliers and so I will be removing them.

train_df = train_df.loc[
    ~((train_df["GrLivArea"] == 4676) & (train_df["SalePrice"] == 184750))
    & ~((train_df["GrLivArea"] == 5642) & (train_df["SalePrice"] == 160000))
].reset_index(drop=True)

# %%
# evaluate null values


def create_scrollable_table(df, table_id, title):
    html = f"<h3>{title}</h3>"
    html += f'<div id="{table_id}" style="height:200px; overflow:auto;">'
    html += df.to_html()
    html += "</div>"

    return html


null_value = train_df.isnull().sum()
html_null_values = create_scrollable_table(
    null_value[null_value > 0].to_frame().sort_values(0, ascending=False),
    "null_values",
    "Null Values in the Dataset",
)

# percentage of missing values for each feature
missing_per = train_df.isnull().sum() / len(train_df) * 100
html_missing_per = create_scrollable_table(
    missing_per[missing_per > 0].to_frame().sort_values(0, ascending=False),
    "missing_per",
    "Percentage of Missing Values for Each Feature",
)

display(HTML(html_null_values + html_missing_per))

# %%
# based on the documentation about the data we know that PoolQC NA means no pool so let's replace NA with None meaning no pool. The following features imputation comes from the documentation of the data.
values = {}
for col in (
    "PoolQC",
    "MiscFeature",
    "Alley",
    "Fence",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "MasVnrType",
):
    values[col] = "None"
# based on the above info pulled from the data's documentation we are able to assume the following:
for col in (
    "GarageYrBlt",
    "GarageArea",
    "GarageCars",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "BsmtFullBath",
    "BsmtHalfBath",
    "MasVnrArea",
):
    values[col] = 0

# this imputation is based on the dataset documentation
train_df["Functional"] = train_df["Functional"].fillna("Typ")
test_df["Functional"] = test_df["Functional"].fillna("Typ")


train_df = train_df.fillna(value=values)
test_df = test_df.fillna(value=values)


# %%
# for all other potential missing features will use the simple imputer and setup either mode for categorical variables or median because most features are skewed for numerical one

# this is based on an approach i saw on kaggle, the idea is the median within neighborhoods would be better at representing that neighborhood than the general overall median of the data.
impute_features = list(
    set(train_df.columns).difference(
        set(list(values.keys()) + ["SalePrice", "Functional", "Id"])
    )
)
numerical_impute = list(
    train_df[impute_features].select_dtypes(include=[np.number]).columns
)

cat_impute = list(train_df[impute_features].select_dtypes(exclude=[np.number]).columns)

# train_df[numerical_impute] = train_df.groupby("Neighborhood")[
#     numerical_impute
# ].transform(lambda x: x.fillna(x.median()))


# test_df[numerical_impute] = test_df.groupby("Neighborhood")[numerical_impute].transform(
#     lambda x: x.fillna(x.median())
# )

# categorical imputation
# train_df[cat_impute] = train_df.groupby("Neighborhood")[cat_impute].transform(
#     lambda x: x.fillna(x.mode())
# )

# test_df[cat_impute] = test_df.groupby("Neighborhood")[cat_impute].transform(
#     lambda x: x.fillna(x.mode())
# )

# this is the control group and will be compared the above.
from sklearn.impute import SimpleImputer


imputer = SimpleImputer(strategy="median")
train_df[numerical_impute] = imputer.fit_transform(
    train_df[numerical_impute]
)  # Fit on training data
test_df[numerical_impute] = imputer.transform(test_df[numerical_impute])

cat_imputer = SimpleImputer(strategy="most_frequent")
train_df[cat_impute] = cat_imputer.fit_transform(train_df[cat_impute])
test_df[cat_impute] = cat_imputer.transform(test_df[cat_impute])

# created bins for the years
bins = [1870, 1900, 1945, 1970, 2000, 2010]
labels = ["Pre-1900", "1900-1945", "1946-1970", "1971-2000", "2001-2010"]

# Create a new binned column
train_df["YearBuilt_Binned"] = pd.cut(train_df["YearBuilt"], bins=bins, labels=labels)
test_df["YearBuilt_Binned"] = pd.cut(test_df["YearBuilt"], bins=bins, labels=labels)

bins = [0, 1940, 1960, 1980, 2000, 2010]
labels = ["Pre-1940", "1941-1960", "1961-1980", "1981-2000", "2001-2010"]

train_df["GarageYrBlt_Binned"] = pd.cut(
    train_df["GarageYrBlt"].replace(0, np.nan), bins=bins, labels=labels
)
train_df["GarageYrBlt_Binned"] = train_df["GarageYrBlt_Binned"].cat.add_categories(
    "No Garage"
)
train_df["GarageYrBlt_Binned"].fillna("No Garage", inplace=True)

test_df["GarageYrBlt_Binned"] = pd.cut(
    test_df["GarageYrBlt"].replace(0, np.nan), bins=bins, labels=labels
)
test_df["GarageYrBlt_Binned"] = test_df["GarageYrBlt_Binned"].cat.add_categories(
    "No Garage"
)
test_df["GarageYrBlt_Binned"].fillna("No Garage", inplace=True)


bins = [1940, 1970, 1990, 2010]
labels = ["1950-1970", "1971-1990", "1991-2010"]

train_df["YearRemodAdd_Binned"] = pd.cut(
    train_df["YearRemodAdd"], bins=bins, labels=labels
)

test_df["YearRemodAdd_Binned"] = pd.cut(
    test_df["YearRemodAdd"], bins=bins, labels=labels
)
# %%
train_df.drop(["YearRemodAdd", "GarageYrBlt", "YearBuilt"], inplace=True, axis=1)
test_df.drop(["YearRemodAdd", "GarageYrBlt", "YearBuilt"], inplace=True, axis=1)
# create_report(train_df).show_browser()
# %%
# Exploring the dependent variable

mu, sigma = stats.norm.fit(train_df["SalePrice"])

hist = go.Histogram(
    x=train_df["SalePrice"],
    nbinsx=50,
    name="Histogram",
    opacity=0.75,
    histnorm="probability density",
    marker=dict(color="purple"),
)

x_norm = np.linspace(train_df["SalePrice"].min(), train_df["SalePrice"].max(), 100)
y_norm = stats.norm.pdf(x_norm, mu, sigma)

norm_data = go.Scatter(
    x=x_norm,
    y=y_norm,
    mode="lines",
    name=f"Normal Dist (mu = {mu:.2f}, sigma = {sigma: .2f})",
    line=dict(color="green"),
)

fig = go.Figure(data=[hist, norm_data])

fig.update_layout(
    title="SalePrice Distribution",
    xaxis_title="SalePrice",
    yaxis_title="Density",
    legend_title_text="Fitted Normal Distribution",
    plot_bgcolor="rgba(32, 32, 32, 1)",
    paper_bgcolor="rgba(32, 32, 32, 1)",
    font=dict(color="white"),
)

qq_data = stats.probplot(train_df["SalePrice"], dist="norm")
qq_fig = px.scatter(
    x=qq_data[0][0],
    y=qq_data[0][1],
    labels={"x": "Theoretical Quantiles", "y": "Ordered Values"},
    color_discrete_sequence=["purple"],
)

qq_fig.update_layout(
    title="Q-Q Plot",
    plot_bgcolor="rgba(32, 32, 32, 1)",
    paper_bgcolor="rgba(32, 32, 32, 1)",
    font=dict(color="white"),
)

slope, intercept, r_value, p_value, std_err = stats.linregress(
    qq_data[0][0], qq_data[0][1]
)
line_x = np.array(qq_data[0][0])
line_y = intercept + slope * line_x

line_data = go.Scatter(
    x=line_x, y=line_y, mode="lines", name="Normal Line", line=dict(color="green")
)

qq_fig.add_trace(line_data)

fig.show()
qq_fig.show()

# is it normalized?
# the dependent variable is not normalized based on the qq plot and histogram
# should it be normalized?
# we should normalize it if we plan to use a model that assumes normality.
# %%
from scipy.stats import skew, boxcox

print(skew(train_df["SalePrice"]))
skew_feat = train_df.select_dtypes(include=[np.number]).apply(
    lambda x: skew(x, bias=False)
)
skew_df = pd.DataFrame({"skew": skew_feat})
skew_df = skew_df[skew_df["skew"].abs() >= 0.5].sort_values("skew", ascending=False)
# %%
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

scaler = StandardScaler()
numerical_impute.remove("YearRemodAdd")
numerical_impute.remove("YearBuilt")
train_df_num = train_df[numerical_impute + ["SalePrice"]]
train_df_num["SalePrice"] = np.log(train_df_num["SalePrice"])
x_2 = scaler.fit_transform(train_df_num[numerical_impute])

# create_report(
#     pd.concat(
#         [pd.DataFrame(x_2, columns=numerical_impute), train_df_num["SalePrice"]], axis=1
#     )
# ).show_browser()
# %%
numeric_pipe = Pipeline(steps=[("scaler", StandardScaler())])
cat_pipe = Pipeline(
    steps=[("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False))]
)


# Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_pipe,
            train_df.select_dtypes(include=[np.number]).columns.drop(
                ["SalePrice", "Id"]
            ),
        ),
        ("cat", cat_pipe, train_df.select_dtypes(exclude=[np.number]).columns),
    ],
    remainder="passthrough",
)

# Create a pipeline with the preprocessor
pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

# Apply the pipeline to your dataset
X = train_df.drop(["SalePrice", "Id"], axis=1)
y = np.log(train_df["SalePrice"])  # normalize dependent variable
X_preprocessed = pipeline.fit_transform(X)
# %%


# for feat in skew_df.index:
#     if skew_df.loc[skew_df.index == feat]["skew"][0] > 1:
#         train_df[feat] = boxcox(train_df[feat], lmbda=0.15)
#         if feat == "SalePrice":
#             continue
#         test_df[feat] = boxcox(test_df[feat], lmbda=0.15)
#     else:
#         train_df[feat] = np.sqrt(train_df[feat])
#         if feat == "SalePrice":
#             continue
#         test_df[feat] = np.sqrt(test_df[feat])

# %%
# if skewness is >0.5 then transformation is needed
# we use a log transformation because the values for salesprice are large and strictly positive

print(skew(train_df["SalePrice"]))
mu, sigma = stats.norm.fit(y)

hist = go.Histogram(
    x=y,
    nbinsx=50,
    name="Histogram",
    opacity=0.75,
    histnorm="probability density",
    marker=dict(color="purple"),
)

x_norm = np.linspace(y.min(), y.max(), 100)
y_norm = stats.norm.pdf(x_norm, mu, sigma)

norm_data = go.Scatter(
    x=x_norm,
    y=y_norm,
    mode="lines",
    name=f"Normal Dist (mu = {mu:.2f}, sigma = {sigma: .2f})",
    line=dict(color="green"),
)

fig = go.Figure(data=[hist, norm_data])

fig.update_layout(
    title="SalePrice Distribution",
    xaxis_title="SalePrice",
    yaxis_title="Density",
    legend_title_text="Fitted Normal Distribution",
    plot_bgcolor="rgba(32, 32, 32, 1)",
    paper_bgcolor="rgba(32, 32, 32, 1)",
    font=dict(color="white"),
)

qq_data = stats.probplot(y, dist="norm")
qq_fig = px.scatter(
    x=qq_data[0][0],
    y=qq_data[0][1],
    labels={"x": "Theoretical Quantiles", "y": "Ordered Values"},
    color_discrete_sequence=["purple"],
)

qq_fig.update_layout(
    title="Q-Q Plot",
    plot_bgcolor="rgba(32, 32, 32, 1)",
    paper_bgcolor="rgba(32, 32, 32, 1)",
    font=dict(color="white"),
)

slope, intercept, r_value, p_value, std_err = stats.linregress(
    qq_data[0][0], qq_data[0][1]
)
line_x = np.array(qq_data[0][0])
line_y = intercept + slope * line_x

line_data = go.Scatter(
    x=line_x, y=line_y, mode="lines", name="Normal Line", line=dict(color="green")
)

qq_fig.add_trace(line_data)

fig.show()
qq_fig.show()

# %%
skew_feat = train_df.select_dtypes(include=[np.number]).apply(
    lambda x: skew(x, bias=False)
)
skew_df_2 = pd.DataFrame({"skew": skew_feat})
skew_df_2 = skew_df_2[skew_df_2["skew"].abs() >= 0.5].sort_values(
    "skew", ascending=False
)
# %% the model build
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_preprocessed, y, test_size=0.2, random_state=42
)
# %%
from sklearn.model_selection import StratifiedKFold

bins = np.digitize(y, bins=np.percentile(y, [25, 50, 75]))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_index, test_index in skf.split(X_preprocessed, bins):
    X_train2, X_test2 = X_preprocessed[train_index], X_preprocessed[test_index]
    y_train2, y_test2 = y[train_index], y[test_index]
# %%
# Define the models
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42),
}

# Define the hyperparameter grids for each model
param_grids = {
    "LinearRegression": {},
    "RandomForest": {
        "n_estimators": [100, 200, 500],
        "max_depth": [None, 10, 30],
        "min_samples_split": [2, 5, 10],
    },
    "XGBoost": {
        "n_estimators": [100, 200, 500],
        "learning_rate": [0.01, 0.1, 0.3],
        "max_depth": [3, 6, 10],
    },
}

# 3-fold cross-validation
cv = KFold(n_splits=3, shuffle=True, random_state=42)

# Train and tune the models
grids = {}
grids2 = {}
for model_name, model in models.items():
    grids[model_name] = GridSearchCV(
        estimator=model,
        param_grid=param_grids[model_name],
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=2,
    )
    grids2[model_name] = GridSearchCV(
        estimator=model,
        param_grid=param_grids[model_name],
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=2,
    )
    grids[model_name].fit(X_train, y_train)
    grids2[model_name].fit(X_train2, y_train2)
    best_params = grids[model_name].best_params_
    best_params2 = grids2[model_name].best_params_
    best_score = np.sqrt(-1 * grids[model_name].best_score_)
    best_score2 = np.sqrt(-1 * grids2[model_name].best_score_)

    print(f"Best parameters for {model_name}: {best_params}")
    print(f"Best RMSE for {model_name}: {best_score}\n")
    print(f"Best parameters for {model_name}: {best_params2} using train_df2")
    print(f"Best RMSE for {model_name}: {best_score2} using traing_df2\n")

# %%
from sklearn.neural_network import MLPRegressor

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

# Create an MLPRegressor instance
mlp = MLPRegressor(
    random_state=42, max_iter=10000, n_iter_no_change=3, learning_rate_init=0.001
)

# Define the parameter grid for tuning
param_grid = {
    "hidden_layer_sizes": [(10,), (10, 10), (10, 10, 10), (25)],
    "activation": ["relu", "tanh"],
    "solver": ["adam"],
    "alpha": [0.0001, 0.001, 0.01],
    "learning_rate": ["constant", "invscaling", "adaptive"],
}

# Create the GridSearchCV object
grid_search_mlp = GridSearchCV(
    mlp, param_grid, scoring="neg_mean_squared_error", cv=3, n_jobs=-1, verbose=1
)

# Fit the model on the training data
grid_search_mlp.fit(X_train_scaled, y_train)

# Print the best parameters found during the search
print("Best parameters found: ", grid_search_mlp.best_params_)

# Evaluate the model on the test data
best_score = np.sqrt(-1 * grid_search_mlp.best_score_)
print("Test score: ", best_score)
# %%
from sklearn.metrics import mean_squared_error

for i in grids.keys():
    print(i + ": " + str(np.sqrt(mean_squared_error(grids[i].predict(X_test), y_test))))

from sklearn.metrics import mean_squared_error

for i in grids.keys():
    print(
        i
        + ": "
        + str(np.sqrt(mean_squared_error(grids_pca[i].predict(X_test_pca), y_test)))
    )
# %% Perform EDA to get sense of the data and answer questions we have of the data
# Try out a few models and parameters to see which performs the best
# feature engineering
# Ensembling
# Submitting to the Competition
#
#

# %%


# %%
