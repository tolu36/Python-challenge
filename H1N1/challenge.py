# %% [markdown]
# This challenge was posted by Driven Data: https://www.drivendata.org/competitions/66/flu-shot-learning/page/211/
# Problem description
# Your goal is to predict how likely individuals are to receive their H1N1 and seasonal flu vaccines. Specifically, you'll be predicting two probabilities: one for h1n1_vaccine and one for seasonal_vaccine.
#
# Each row in the dataset represents one person who responded to the National 2009 H1N1 Flu Survey.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


# %%s
train_df = pd.read_csv("training_set_features.csv")
label_df = pd.read_csv("training_set_labels.csv")
test_df = pd.read_csv("test_set_features.csv")
print(train_df.shape, test_df.shape, label_df.shape)

# %% [markdown]
# looks like the respondent_id is the common link between the feature dataset and label dataset
#
# I will now perform some basic EDA

# %%
label_df[["h1n1_vaccine", "seasonal_vaccine"]] = label_df[
    ["h1n1_vaccine", "seasonal_vaccine"]
].astype("category")

# %%
train_df.isna().sum()

# %%
tmp = label_df.value_counts(["h1n1_vaccine", "seasonal_vaccine"]) / (label_df.shape[0])
tmp.reset_index()

# %%
tmp = label_df.value_counts(["h1n1_vaccine"]) / (label_df.shape[0])
fig = px.bar(
    tmp.reset_index(),
    x="h1n1_vaccine",
    y="count",
    barmode="group",
    color="h1n1_vaccine",
    text_auto=".2%",
)
fig.update_yaxes(tickformat=".2%")
fig.show()
tmp = label_df.value_counts(["seasonal_vaccine"]) / (label_df.shape[0])
fig = px.bar(
    tmp.reset_index(),
    x="seasonal_vaccine",
    y="count",
    barmode="group",
    color="seasonal_vaccine",
    text_auto=".2%",
)
fig.update_yaxes(tickformat=".2%")
fig.show()

# %% [markdown]
# We can see that almost half the people in the sample take seasonal vaccines yet only 21% of them took the h1n1 vaccine.
# the seasonal flu shot is balance, however there is quite an imbalance with the h1n1 vaccine shot.

# %%
pd.crosstab(
    label_df["h1n1_vaccine"], label_df["seasonal_vaccine"], margins=True, normalize=True
)
label_df[["h1n1_vaccine", "seasonal_vaccine"]].corr()

# %% From the above we that see that of the 21% who took the h1n1 shot 17% of them also took the seasonal shot. However, of the 46.6% who took the seasonal shot only 17% of them took the h1n1 shot. This implies that there is some correction between taking the h1n1 and seasonal flu shot. This correlation is moderate and positive one.

# %%
df = train_df.merge(label_df, how="right", on="respondent_id")
# %%


def pairplot(df):
    cols = list(df.columns)[1:-2]
    for col in cols:
        tmp = df.groupby(["h1n1_vaccine", col]).size().unstack("h1n1_vaccine")
        tmp["no"] = tmp[0] / (tmp[0] + tmp[1])
        tmp["yes"] = tmp[1] / (tmp[0] + tmp[1])

        fig = px.bar(
            tmp, x=tmp.index, y=["yes", "no"], text_auto=".2%", title=f"{col} plot"
        )
        fig.update_yaxes(tickformat=".2%")
        fig.update_layout(legend_title_text="h1n1_vaccine")
        fig.show()

    for col in cols:
        tmp = df.groupby(["seasonal_vaccine", col]).size().unstack("seasonal_vaccine")
        tmp["no"] = tmp[0] / (tmp[0] + tmp[1])
        tmp["yes"] = tmp[1] / (tmp[0] + tmp[1])

        fig = px.bar(
            tmp, x=tmp.index, y=["yes", "no"], text_auto=".2%", title=f"{col} plot"
        )
        fig.update_yaxes(tickformat=".2%")
        fig.update_layout(legend_title_text="seasonal_vaccine")
        fig.show()


pairplot(df)
# %%
# from the above we can see tha h1n1 concern level is definitely correlated to increase of h1n1 shots

# Key determining features to whether a person will take a shot in general seems to be their opinion on the matter and their knowledge about it. Overall we see that the more concern and more knowledge a person has about h1n1 they more likely they are to take the shot.
#
# If the person tends to be follow the suggested preventative measures and has positive outlook towards vaccines they will tend to take the shot as well.
#
# The biggest indicator we see is that if people's doctor's recommended they take h1n1 shot. More than 50% of people who took the shot had a doctor's recommendation
#
#  a similar trend can be seen for doctor recommended seasonal shots, however to less degree with about 34.8% of those who took the h1n1 shot having a doctor's recommendation for a seasonal shot.

# those 27% of those who took the h1n1 shot had a medical condition so those most at risk tended to take the shot more than their counter parts.
# We saw that those who have contact with young children were likely to take the shot more than their counter part
# health professionals were more likely to take the shot than non health professionals
# similarly we see that those with health insurance were more likely to get the shot than those without.
# Interestingly we see that those without any knowledge/opinion of the risk the h1n1 virus poses were more likely to not take the shot than those with an opinion that implied little to no risk at all. A similar story can be seen with regards to people's knowledge/opinion on the risk of the seasonal flu when it comes to getting the h1n1 shot.

# we that age 55-64 age group were likely to get the shot than the other age group likely cause they more at risk than the younger age group and yet young enough to combat any adverse effect the shot may have when compared to the older age group

# we see the more educational experience a person have more likely they are to get the shot. A similar corelation can be drawn with income

# When we look at the corelation between the features and the seasonal_vaccine shot we see that much of the same corelations stated above were the same for seasonal vaccine, however, we found that the older the person the more likely they are to take the seasonal flu.


df.isna().sum().sort_values(ascending=False) / (df.shape[0])
for i in list(df.isna().sum()[df.isna().sum() > 0].index):
    fig = px.histogram(df[i])
    fig.show()
# more than 45 percent of the column is missing values for employment_occupation, employment_industry and health_insurance
# health insurance is get indicator of whether a person will get the shots or not so I will try and perform an imputation on it.

# I wil try to impute the data using the 3 known methods present in sklearn simple/knn/iterative

# from the plots above we can see that the distribution of all of the features are skewed as well because these features are all categorical it best to imply median as the imputation value when using the simpleimputer in sklearn

# I will be initially removing the features that I believe do not do a great job for discriminating between those who got the shots and did not
# %%
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
import category_encoders as ce

# %%
np.random.seed(42)

df_knn = df.copy()
# Identify columns with missing values
df_knn = df_knn.drop(
    columns=[
        "hhs_geo_region",
        "census_msa",
    ]
)

cat_cols = [
    col
    for col in list(df_knn.dtypes[df_knn.dtypes == "object"].index)
    if col not in ["education", "income_poverty", "age_group"]
]
bi_encoder = ce.BinaryEncoder(
    cols=cat_cols,
    return_df=True,
    handle_missing="return_nan",
    handle_unknown="return_nan",
)
ed_encoder = ce.OrdinalEncoder(
    cols=["education"],
    mapping=[
        {
            "col": "education",
            "mapping": {
                "< 12 Years": 1,
                "12 Years": 2,
                "Some College": 3,
                "College Graduate": 4,
            },
        }
    ],
    return_df=True,
    handle_missing="return_nan",
    handle_unknown="return_nan",
)
inc_encoder = ce.OrdinalEncoder(
    cols=["income_poverty"],
    mapping=[
        {
            "col": "income_poverty",
            "mapping": {
                "Below Poverty": 1,
                "<= $75,000, Above Poverty": 2,
                "> $75,000": 3,
            },
        }
    ],
    return_df=True,
    handle_missing="return_nan",
    handle_unknown="return_nan",
)
age_encoder = ce.OrdinalEncoder(
    cols=["age_group"],
    mapping=[
        {
            "col": "age_group",
            "mapping": {
                "18 - 34 Years": 1,
                "35 - 44 Years": 2,
                "45 - 54 Years": 3,
                "55 - 64 Years": 4,
                "65+ Years": 5,
            },
        }
    ],
    return_df=True,
    handle_missing="return_nan",
    handle_unknown="return_nan",
)


data_knn_encoded = bi_encoder.fit_transform(df_knn)
data_knn_encoded = ed_encoder.fit_transform(data_knn_encoded)
data_knn_encoded = inc_encoder.fit_transform(data_knn_encoded)
data_knn_encoded = age_encoder.fit_transform(data_knn_encoded)
# %%
columns_with_missing = list(
    data_knn_encoded.isna().sum()[data_knn_encoded.isna().sum() > 0].index
)

# Separate the features with and without missing values

X_missing = data_knn_encoded[columns_with_missing]
X_complete = data_knn_encoded.drop(columns=columns_with_missing)

# Initialize the KNNImputer
imputer = KNNImputer(n_neighbors=9)  # You can adjust the number of neighbors

# Fit and transform the imputer on your data
X_imputed = imputer.fit_transform(X_missing)

data_knn_imputed = data_knn_encoded.copy()
# Replace the missing values in your original dataset
data_knn_imputed[columns_with_missing] = X_imputed

# %%
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge


# Create an instance of IterativeImputer with BayesianRidge estimator
imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0)

df_imputed = imputer.fit_transform(data_knn_encoded)

# Convert the imputed data back to a DataFrame
df_imputed = pd.DataFrame(df_imputed, columns=data_knn_encoded.columns)

# %%
df_sub = df.drop(
    columns=[
        "hhs_geo_region",
        "census_msa",
    ]
)

imputer = SimpleImputer(strategy="most_frequent")
df_filled = imputer.fit_transform(df_sub)
df_filled = pd.DataFrame(df_filled, columns=df_sub.columns)
# %%
df_filled_enc = bi_encoder.fit_transform(df_filled)
df_filled_enc = ed_encoder.fit_transform(df_filled_enc)
df_filled_enc = inc_encoder.fit_transform(df_filled_enc)
df_filled_enc = age_encoder.fit_transform(df_filled_enc)
# %%
df_y = label_df.iloc[:, 1:].astype("int")
# df_simp_mod = df_filled_enc.iloc[:, 1:-2]
# df_knn_mod = data_knn_imputed.iloc[:, 1:-2]
df_iter_mod = df_imputed.iloc[:, 1:-2]
# %%
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

classifier = MultiOutputClassifier(LogisticRegression())


param_grid = {
    "estimator__C": [0.001, 0.01, 0.1, 1, 10],
    "estimator__penalty": ["l1", "l2", "none"],
    "estimator__solver": ["lbfgs", "liblinear", "saga"],
    "estimator__class_weight": [None, "balanced"],
    "estimator__multi_class": ["ovr", "multinomial"],
    "estimator__tol": [1e-4, 1e-3, 1e-2],
    "estimator__max_iter": [100, 200, 300],
}
# %%
grid_search_sim = GridSearchCV(
    estimator=classifier, param_grid=param_grid, scoring="accuracy", cv=5, n_jobs=-1
)

X_train, X_test, y_train, y_test = train_test_split(
    df_simp_mod, df_y, random_state=0, shuffle=True, test_size=0.20, stratify=df_y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

grid_search_sim.fit(X_train_scaled, y_train)
best_logistic_regression_sim = grid_search_sim.best_estimator_
best_params_sim = grid_search_sim.best_params_

y_pred_sim = best_logistic_regression_sim.predict_proba(X_test_scaled)

# %%
from sklearn.metrics import roc_curve, roc_auc_score, classification_report

# y_pred_sim = pd.DataFrame(
#     {
#         "h1n1_vaccine": y_pred_sim[0][:, 1],
#         "seasonal_vaccine": y_pred_sim[1][:, 1],
#     },
#     index=y_test.index,
# )
# roc_sim = roc_auc_score(y_test, y_pred_sim)
# print(roc_sim)


def plot_roc(y_true, y_score, label_name, ax):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--")
    ax.set_ylabel("TPR")
    ax.set_xlabel("FPR")
    ax.set_title(f"{label_name}: AUC = {roc_auc_score(y_true, y_score):.4f}")


# fig, ax = plt.subplots(1, 2, figsize=(7, 3.5))

# plot_roc(y_test["h1n1_vaccine"], y_pred_sim["h1n1_vaccine"], "h1n1_vaccine", ax=ax[0])
# plot_roc(
#     y_test["seasonal_vaccine"],
#     y_pred_sim["seasonal_vaccine"],
#     "seasonal_vaccine",
#     ax=ax[1],
# )
# fig.tight_layout()
# report = classification_report(y_test, y_pred_sim)
# print(report)
# the model performance was a accuracy of 0.8466 which is better than the benchmark of 0.8185
# %%

grid_search_knn = GridSearchCV(
    estimator=classifier, param_grid=param_grid, scoring="accuracy", cv=5, n_jobs=-1
)
X_train, X_test, y_train, y_test = train_test_split(
    df_knn_mod, df_y, random_state=0, shuffle=True, test_size=0.20, stratify=df_y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

grid_search_knn.fit(X_train_scaled, y_train)
best_logistic_regression_knn = grid_search_knn.best_estimator_
best_params_knn = grid_search_knn.best_params_

y_pred_knn = best_logistic_regression_knn.predict_proba(X_test_scaled)

# %%
y_pred_knn = pd.DataFrame(
    {
        "h1n1_vaccine": y_pred_knn[0][:, 1],
        "seasonal_vaccine": y_pred_knn[1][:, 1],
    },
    index=y_test.index,
)
roc_knn = roc_auc_score(y_test, y_pred_knn)
print(roc_knn)
fig, ax = plt.subplots(1, 2, figsize=(7, 3.5))

plot_roc(y_test["h1n1_vaccine"], y_pred_knn["h1n1_vaccine"], "h1n1_vaccine", ax=ax[0])
plot_roc(
    y_test["seasonal_vaccine"],
    y_pred_knn["seasonal_vaccine"],
    "seasonal_vaccine",
    ax=ax[1],
)
fig.tight_layout()
# %%
grid_search_iter = GridSearchCV(
    estimator=classifier, param_grid=param_grid, scoring="accuracy", cv=5, n_jobs=-1
)
X_train, X_test, y_train, y_test = train_test_split(
    df_iter_mod, df_y, random_state=0, shuffle=True, test_size=0.20, stratify=df_y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

grid_search_iter.fit(X_train_scaled, y_train)
best_logistic_regression_iter = grid_search_iter.best_estimator_
best_params_iter = grid_search_iter.best_params_

y_pred_iter = best_logistic_regression_iter.predict_proba(X_test_scaled)

# %%
# y_pred_iter = pd.DataFrame(
#     {
#         "h1n1_vaccine": y_pred_iter[0][:, 1],
#         "seasonal_vaccine": y_pred_iter[1][:, 1],
#     },
#     index=y_test.index,
# )
roc_iter = roc_auc_score(y_test, y_pred_iter)
print(roc_iter)

fig, ax = plt.subplots(1, 2, figsize=(7, 3.5))

plot_roc(y_test["h1n1_vaccine"], y_pred_iter["h1n1_vaccine"], "h1n1_vaccine", ax=ax[0])
plot_roc(
    y_test["seasonal_vaccine"],
    y_pred_iter["seasonal_vaccine"],
    "seasonal_vaccine",
    ax=ax[1],
)
fig.tight_layout()

# %%
# from the benchmark they only used the numerical features so I will do the same to replicate the results

df_knn_num = df[list(df.dtypes[df.dtypes != "object"].index)]
columns_with_missing = list(df_knn_num.isna().sum()[df_knn_num.isna().sum() > 0].index)

# Separate the features with and without missing values

X_missing = df_knn_num[columns_with_missing]
X_complete = df_knn_num.drop(columns=columns_with_missing)

# Initialize the KNNImputer
imputer = KNNImputer(n_neighbors=9)  # You can adjust the number of neighbors

# Fit and transform the imputer on your data
X_imputed = imputer.fit_transform(X_missing)

data_knn_imputed_n = df_knn_num.copy()
# Replace the missing values in your original dataset
data_knn_imputed_n[columns_with_missing] = X_imputed

# %%
df_sim_num = df[list(df.dtypes[df.dtypes != "object"].index)]
imputer = SimpleImputer(strategy="median")
df_filled_num = imputer.fit_transform(df_sim_num)
df_filled_num = pd.DataFrame(df_filled_num, columns=df_sim_num.columns)

# %%
df_iter_num = df[list(df.dtypes[df.dtypes != "object"].index)]
imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0)

df_imputed_n = imputer.fit_transform(df_iter_num)

# Convert the imputed data back to a DataFrame
df_imputed_n = pd.DataFrame(df_imputed_n, columns=df_iter_num.columns)
# %%
df_sim_mod_n = df_filled_num.iloc[:, 1:-2]
df_knn_mod_n = data_knn_imputed_n.iloc[:, 1:-2]
df_iter_mod_n = df_imputed_n.iloc[:, 1:-2]
# %%
params = {k.replace("estimator__", ""): v for k, v in best_params_iter.items()}
classifier_sim = MultiOutputClassifier(LogisticRegression(**params))

X_train, X_test, y_train, y_test = train_test_split(
    df_sim_mod_n, df_y, random_state=0, shuffle=True, test_size=0.20, stratify=df_y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

classifier_sim.fit(X_train_scaled, y_train)
y_pred_sim_n = classifier_sim.predict_proba(X_test_scaled)
y_pred_sim_n = pd.DataFrame(
    {
        "h1n1_vaccine": y_pred_sim_n[0][:, 1],
        "seasonal_vaccine": y_pred_sim_n[1][:, 1],
    },
    index=y_test.index,
)
# %%
from sklearn.metrics import roc_curve, roc_auc_score, classification_report

roc_sim_n = roc_auc_score(y_test, y_pred_sim_n)
print(roc_sim_n)

fig, ax = plt.subplots(1, 2, figsize=(7, 3.5))

plot_roc(y_test["h1n1_vaccine"], y_pred_sim_n["h1n1_vaccine"], "h1n1_vaccine", ax=ax[0])
plot_roc(
    y_test["seasonal_vaccine"],
    y_pred_sim_n["seasonal_vaccine"],
    "seasonal_vaccine",
    ax=ax[1],
)
fig.tight_layout()
# %%
classifier_knn = MultiOutputClassifier(LogisticRegression(**params))

X_train, X_test, y_train, y_test = train_test_split(
    df_knn_mod_n, df_y, random_state=0, shuffle=True, test_size=0.20, stratify=df_y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

classifier_knn.fit(X_train_scaled, y_train)
y_pred_knn_n = classifier_sim.predict_proba(X_test_scaled)

y_pred_knn_n = pd.DataFrame(
    {
        "h1n1_vaccine": y_pred_knn_n[0][:, 1],
        "seasonal_vaccine": y_pred_knn_n[1][:, 1],
    },
    index=y_test.index,
)
# %%
roc_knn_n = roc_auc_score(y_test, y_pred_knn_n)
print(roc_knn_n)

fig, ax = plt.subplots(1, 2, figsize=(7, 3.5))

plot_roc(y_test["h1n1_vaccine"], y_pred_knn_n["h1n1_vaccine"], "h1n1_vaccine", ax=ax[0])
plot_roc(
    y_test["seasonal_vaccine"],
    y_pred_knn_n["seasonal_vaccine"],
    "seasonal_vaccine",
    ax=ax[1],
)
fig.tight_layout()
# %%
classifier_iter = MultiOutputClassifier(LogisticRegression(**params))

X_train, X_test, y_train, y_test = train_test_split(
    df_iter_mod_n, df_y, random_state=0, shuffle=True, test_size=0.20, stratify=df_y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

classifier_iter.fit(X_train_scaled, y_train)
y_pred_iter_n = classifier_iter.predict_proba(X_test_scaled)

y_pred_iter_n = pd.DataFrame(
    {
        "h1n1_vaccine": y_pred_iter_n[0][:, 1],
        "seasonal_vaccine": y_pred_iter_n[1][:, 1],
    },
    index=y_test.index,
)
# %%
roc_iter_n = roc_auc_score(y_test, y_pred_iter_n)
print(roc_iter_n)

fig, ax = plt.subplots(1, 2, figsize=(7, 3.5))

plot_roc(
    y_test["h1n1_vaccine"], y_pred_iter_n["h1n1_vaccine"], "h1n1_vaccine", ax=ax[0]
)
plot_roc(
    y_test["seasonal_vaccine"],
    y_pred_iter_n["seasonal_vaccine"],
    "seasonal_vaccine",
    ax=ax[1],
)
fig.tight_layout()
# %%
coeff = [est.coef_ for est in best_logistic_regression_iter.estimators_]
feature_names = X_train.columns  # Replace with your feature names
coefficients_and_features = []

for i, coef in enumerate(coeff):
    output_feature_names = [f"{feature}_output_{i}" for feature in feature_names]
    coefficients_and_features.extend(list(zip(output_feature_names, coef[0])))

# Print the coefficients and their corresponding feature names for output 0
for feature, coef in coefficients_and_features:
    print(f"{feature}: {coef}")

# %%
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer, KNNImputer
import category_encoders as ce

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# %%
df_iter = df.copy()
# Identify columns with missing values
df_iter = df_iter.drop(
    columns=[
        "hhs_geo_region",
        "census_msa",
    ]
)
df_iter_mod = df_iter.iloc[:, 1:-2]
df_y = label_df.iloc[:, 1:]

cat_cols = [
    col
    for col in list(df_iter_mod.dtypes[df_iter_mod.dtypes == "object"].index)
    if col not in ["education", "income_poverty", "age_group"]
]

bi_encoder = ce.BinaryEncoder(
    cols=cat_cols,
    return_df=True,
    handle_missing="return_nan",
    handle_unknown="return_nan",
)
ed_encoder = ce.OrdinalEncoder(
    cols=["education"],
    mapping=[
        {
            "col": "education",
            "mapping": {
                "< 12 Years": 1,
                "12 Years": 2,
                "Some College": 3,
                "College Graduate": 4,
            },
        }
    ],
    return_df=True,
    handle_missing="return_nan",
    handle_unknown="return_nan",
)
inc_encoder = ce.OrdinalEncoder(
    cols=["income_poverty"],
    mapping=[
        {
            "col": "income_poverty",
            "mapping": {
                "Below Poverty": 1,
                "<= $75,000, Above Poverty": 2,
                "> $75,000": 3,
            },
        }
    ],
    return_df=True,
    handle_missing="return_nan",
    handle_unknown="return_nan",
)
age_encoder = ce.OrdinalEncoder(
    cols=["age_group"],
    mapping=[
        {
            "col": "age_group",
            "mapping": {
                "18 - 34 Years": 1,
                "35 - 44 Years": 2,
                "45 - 54 Years": 3,
                "55 - 64 Years": 4,
                "65+ Years": 5,
            },
        }
    ],
    return_df=True,
    handle_missing="return_nan",
    handle_unknown="return_nan",
)
# %%
np.random.seed(42)

impute_scale = Pipeline(
    [
        (
            "imputer",
            IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0),
        ),
        ("scaler", StandardScaler()),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", bi_encoder, cat_cols),
        ("ed_encoder", ed_encoder, ["education"]),
        ("inc_encoder", inc_encoder, ["income_poverty"]),
        ("age_encoder", age_encoder, ["age_group"]),
        ("impute_scale", impute_scale, list(df_iter_mod.columns)),
    ],
    remainder="passthrough",
)


# Create the final pipeline including preprocessing and the classifier
pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", MultiOutputClassifier(LogisticRegression())),
    ]
)

# Define the hyperparameter grid for GridSearchCV
param_grid = {
    "classifier__estimator__C": [0.001, 0.01, 0.1, 1, 10],
    "classifier__estimator__penalty": ["l1", "l2", "none"],
    "classifier__estimator__solver": ["lbfgs", "liblinear", "saga"],
    "classifier__estimator__class_weight": [None, "balanced"],
    "classifier__estimator__multi_class": ["ovr", "multinomial"],
    "classifier__estimator__tol": [1e-4, 1e-3, 1e-2],
    "classifier__estimator__max_iter": [100, 200, 300],
}

# Create GridSearchCV
grid_search = GridSearchCV(
    estimator=pipeline, param_grid=param_grid, scoring="accuracy", cv=5, n_jobs=-1
)

# Split the data and fit the pipeline
X_train, X_test, y_train, y_test = train_test_split(
    df_iter_mod, df_y, random_state=0, shuffle=True, test_size=0.20, stratify=df_y
)
# %%
# Fit the pipeline with the training data
grid_search.fit(X_train, y_train)

# Get the best estimator and best parameters
best_pipeline = grid_search.best_estimator_
best_params = grid_search.best_params_

# %%
# Make predictions using the best pipeline
y_pred = best_pipeline.predict_proba(X_test)
y_pred = pd.DataFrame(
    {
        "h1n1_vaccine": y_pred[0][:, 1],
        "seasonal_vaccine": y_pred[1][:, 1],
    },
    index=y_test.index,
)
roc_iter = roc_auc_score(y_test, y_pred)
print(roc_iter)

fig, ax = plt.subplots(1, 2, figsize=(7, 3.5))

plot_roc(y_test["h1n1_vaccine"], y_pred["h1n1_vaccine"], "h1n1_vaccine", ax=ax[0])
plot_roc(
    y_test["seasonal_vaccine"],
    y_pred["seasonal_vaccine"],
    "seasonal_vaccine",
    ax=ax[1],
)
fig.tight_layout()

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(df_iter_mod)
best_logistic_regression_iter.fit(X_train_scaled, df_y)
test_df_copy = test_df.iloc[:, 1:].drop(
    columns=[
        "hhs_geo_region",
        "census_msa",
    ]
)

cat_cols = [
    col
    for col in list(test_df_copy.dtypes[test_df_copy.dtypes == "object"].index)
    if col not in ["education", "income_poverty", "age_group"]
]

test_df_copy_en = bi_encoder.fit_transform(test_df_copy)
test_df_copy_en = ed_encoder.fit_transform(test_df_copy_en)
test_df_copy_en = inc_encoder.fit_transform(test_df_copy_en)
test_df_copy_en = age_encoder.fit_transform(test_df_copy_en)

test_imputed = imputer.fit_transform(test_df_copy_en)

# Convert the imputed data back to a DataFrame
test_imputed = pd.DataFrame(test_imputed, columns=test_df_copy_en.columns)

test_proba = best_logistic_regression_iter.predict_proba(test_imputed)
test_proba = pd.DataFrame(
    {
        "h1n1_vaccine": test_proba[0][:, 1],
        "seasonal_vaccine": test_proba[1][:, 1],
    },
    index=test_imputed.index,
)
# %%
df_submit = pd.read_csv("submission_format.csv")

df_submit[["h1n1_vaccine", "seasonal_vaccine"]] = test_proba
df_submit["respondent_id"] = test_df["respondent_id"]
df_submit.set_index("respondent_id").to_csv("submission_format.csv")


# %%
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import category_encoders as ce
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

train_df = pd.read_csv("training_set_features.csv")
label_df = pd.read_csv("training_set_labels.csv")
test_df = pd.read_csv("test_set_features.csv")
label_df[["h1n1_vaccine", "seasonal_vaccine"]] = label_df[
    ["h1n1_vaccine", "seasonal_vaccine"]
].astype("category")
print(train_df.shape, test_df.shape, label_df.shape)


def data_preprocessing(feature_df, label_df=None, training=True):
    df_copy = feature_df.iloc[:, 1:].copy()
    if label_df is not None:
        df_y = label_df.iloc[:, 1:].astype("int")
    df_copy = df_copy.drop(
        columns=[
            "hhs_geo_region",
            "census_msa",
        ]
    )

    cat_cols = [
        col
        for col in list(df_copy.dtypes[df_copy.dtypes == "object"].index)
        if col not in ["education", "income_poverty", "age_group"]
    ]
    bi_encoder = ce.BinaryEncoder(
        cols=cat_cols,
        return_df=True,
        handle_missing="return_nan",
        handle_unknown="return_nan",
    )
    ed_encoder = ce.OrdinalEncoder(
        cols=["education"],
        mapping=[
            {
                "col": "education",
                "mapping": {
                    "< 12 Years": 1,
                    "12 Years": 2,
                    "Some College": 3,
                    "College Graduate": 4,
                },
            }
        ],
        return_df=True,
        handle_missing="return_nan",
        handle_unknown="return_nan",
    )
    inc_encoder = ce.OrdinalEncoder(
        cols=["income_poverty"],
        mapping=[
            {
                "col": "income_poverty",
                "mapping": {
                    "Below Poverty": 1,
                    "<= $75,000, Above Poverty": 2,
                    "> $75,000": 3,
                },
            }
        ],
        return_df=True,
        handle_missing="return_nan",
        handle_unknown="return_nan",
    )
    age_encoder = ce.OrdinalEncoder(
        cols=["age_group"],
        mapping=[
            {
                "col": "age_group",
                "mapping": {
                    "18 - 34 Years": 1,
                    "35 - 44 Years": 2,
                    "45 - 54 Years": 3,
                    "55 - 64 Years": 4,
                    "65+ Years": 5,
                },
            }
        ],
        return_df=True,
        handle_missing="return_nan",
        handle_unknown="return_nan",
    )

    df_enc = bi_encoder.fit_transform(df_copy)
    df_enc = ed_encoder.fit_transform(df_enc)
    df_enc = inc_encoder.fit_transform(df_enc)
    df_enc = age_encoder.fit_transform(df_enc)

    # Create an instance of IterativeImputer with BayesianRidge estimator
    imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0)

    df_imputed = imputer.fit_transform(df_enc)

    # Convert the imputed data back to a DataFrame
    df_imputed = pd.DataFrame(df_imputed, columns=df_enc.columns)
    scaler = StandardScaler()
    if training:
        X_train, X_test, y_train, y_test = train_test_split(
            df_imputed,
            df_y,
            random_state=0,
            shuffle=True,
            test_size=0.20,
            stratify=df_y,
        )

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, y_train, X_test_scaled, y_test
    else:
        df_sc_en_im = scaler.fit_transform(df_imputed)
        return df_sc_en_im


def plot_roc(y_true, y_score, label_name, ax):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--")
    ax.set_ylabel("TPR")
    ax.set_xlabel("FPR")
    ax.set_title(f"{label_name}: AUC = {roc_auc_score(y_true, y_score):.4f}")


# %%
classifier = MultiOutputClassifier(LogisticRegression())


param_grid = {
    "estimator__C": [0.001, 0.01, 0.1, 1, 10],
    "estimator__penalty": ["l1", "l2", "none"],
    "estimator__solver": ["lbfgs", "liblinear", "saga"],
    "estimator__class_weight": [None, "balanced"],
    "estimator__multi_class": ["ovr", "multinomial"],
    "estimator__tol": [1e-4, 1e-3, 1e-2],
    "estimator__max_iter": [100, 200, 300],
}

np.random.seed(42)

X_train_scaled, y_train, X_test_scaled, y_test = data_preprocessing(train_df, label_df)

grid_search_iter = GridSearchCV(
    estimator=classifier, param_grid=param_grid, scoring="accuracy", cv=5, n_jobs=-1
)

grid_search_iter.fit(X_train_scaled, y_train)
best_logistic_regression_iter = grid_search_iter.best_estimator_
best_params_iter = grid_search_iter.best_params_

y_pred_iter = best_logistic_regression_iter.predict_proba(X_test_scaled)

# %%
# y_pred_iter = pd.DataFrame(
#     {
#         "h1n1_vaccine": y_pred_iter[0][:, 1],
#         "seasonal_vaccine": y_pred_iter[1][:, 1],
#     },
#     index=y_test.index,
# )
roc_iter = roc_auc_score(y_test, y_pred_iter)
print(roc_iter)

fig, ax = plt.subplots(1, 2, figsize=(7, 3.5))

plot_roc(y_test["h1n1_vaccine"], y_pred_iter["h1n1_vaccine"], "h1n1_vaccine", ax=ax[0])
plot_roc(
    y_test["seasonal_vaccine"],
    y_pred_iter["seasonal_vaccine"],
    "seasonal_vaccine",
    ax=ax[1],
)
fig.tight_layout()
# %%

test_df_pro = data_preprocessing(test_df, training=False)
test_proba = best_logistic_regression_iter.predict_proba(test_df_pro)
test_proba = pd.DataFrame(
    {
        "h1n1_vaccine": test_proba[0][:, 1],
        "seasonal_vaccine": test_proba[1][:, 1],
    },
    index=test_imputed.index,
)
# %%
df_submit = pd.read_csv("submission_format.csv")

df_submit[["h1n1_vaccine", "seasonal_vaccine"]] = test_proba
df_submit["respondent_id"] = test_df["respondent_id"]
df_submit.set_index("respondent_id").to_csv("submission_format.csv")
