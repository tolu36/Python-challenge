# %% [markdown]
# # Economic Factors
#
# 1. Employment Variation Rate (emp.var.rate)
#     - measures the change in the employment rate over a quarterly period.
#     - tells us how the job market is doing, positive value implies improving employment situation while the opposite is true if negative.
#     - a positive emp.var.rate may correlate with higher spending and a greater likelihood to subscribe to term deposit.
#     - the opposite is true for a negative emp.var.rate.
#
# 2. Consumer Price Index (cons.price.idx)
#     - the average change in prices paid by consumers for goods and services monthly (inflation or deflation tracker).
#     - an increasing cpi means inflation meaning less buying power for the consumer and probably less likely to subscribe to the term deposit (this is true in most cases).
#     - the opposite is true (in most cases).
#
# 3. Consumer Confidence Index (cons.conf.idx)
#     - tells us how the consumer is feeling about the economy, including financial health and spending/investing.
#     - high cons.conf.idx will likely lead to higher term deposit subscription as they are more willing to invest.
#     - the opposite is true.
#
# 4. Euribor 3-Month Rate (euribor3m)
#     - The interest rate European banks lend to each other for a period of 3 months.
#     - high euribor3m rate could make term deposit more attractive as banks may offer higher interest rates on the deposits. This is not always the case be higher borrowing cost means less money for certain clients.
#     - the opposite is true.
#
# A positive economic outlook (e.g., rising employment, stable prices, high consumer confidence) can drive higher subscription rates.
# A negative outlook might require targeting clients more likely to subscribe despite economic uncertainty (e.g., risk-averse individuals who value guaranteed returns).

# %%
# packages
import pandas as pd
import numpy as np
from dataprep.eda import create_report
import warnings

warnings.filterwarnings("ignore")

import plotly.express as px
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import chi2_contingency

import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import skew

import plotly.figure_factory as ff

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    ParameterGrid,
)
from sklearn.metrics import (
    recall_score,
    precision_score,
    classification_report,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import IsolationForest
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA

np.random.seed(42)


# %%
df = pd.read_csv("term deposit data.csv", sep=";")
create_report(df).show()

# %% [markdown]
# # Inital EDA:
# - there's 41,188 rows of data
# - 20 features, one target
# - no missing values
# - 88.73% of people said no to the subscription and 11.27% said yes (imbalanced classes)
# - majority of people called are married (~60%)
# - average is roughly 40 yrs old
# - admin, blue-collar and technician make up roughly 64% of the clients
# - most client have some kind of higher education (~65%)
# - most client have no personal loans (~82%)
# - most client are contacted by cell (~64%)
# - most clients were contacted during months of may-aug (~78%)
# - days of the week isn't an impact on the subscription likely a feature to drop
# - most calls are under 5 mins (~72%)
# - During this campaign most clients were contacted between 1 to 2 times (~69%)
# - This campaign is mostly targeting clients never been contacted rather than previously contacted clients
# - most client don't have a credit in default, this feature isn't really tell us anything it might be best to drop it -- 79.12% say no, 20.87% are unknown and 0.01% are yes.
# - similar story can be said about pdays
# - much of the economic features are strongly correlated with each other, meaning feature selection will need to happen to select the best one.

# %% [markdown]
# # Potential Business Objective
# The business objective is to expand the customer base for term deposits by targeting a large pool of first-time contacts, maximizing the identification of potential subscribers through the marketing campaign. The focus is on improving subscription rates among new clients while maintaining efficiency in resource allocation, ensuring a balance between growth and campaign effectiveness.
#
#
# # metric focus
# - Primary: Recall (ensure you don’t miss too many potential "yes" clients).
# - Secondary: Precision (focus on correct "yes" predictions).
#
# Given the trend in the data, the campaign aims to grow the subscription base and maximize coverage among new clients, recall should be the primary metric, with precision as a secondary consideration.

# %%
print(df.groupby(["default"])["y"].value_counts())


# Counts of successes (subscriptions)


subscriptions = [443, 4197]


# Total counts for each group


totals = [8154 + 443, 4197 + 28391]


# Perform two-proportion Z-test


z_stat, p_value = proportions_ztest(count=subscriptions, nobs=totals)


print(f"Z-Statistic: {z_stat:.2f}, P-Value: {p_value:.4f}")


# Interpret results


if p_value < 0.05:

    print("Statistically significant difference between the two groups.")
else:

    print("No statistically significant difference between the two groups.")

# Contingency table (rows: "unknown" and "no"; columns: "yes" and "no")
contingency_table = [[28391, 4197], [8154, 443], [3, 0]]  # "unknown"  # "no"

# Perform chi-square test
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

print(f"Chi-Square Statistic: {chi2:.2f}, P-Value: {p_value:.4f}")

# Interpret results
if p_value < 0.05:
    print("Statistically significant association between default and subscription.")
else:
    print("No statistically significant association between default and subscription.")

# %%
df_melted = (
    df.groupby(["y", "default"])["default"]
    .agg({"count"})
    .sort_values("count", ascending=False)
    .reset_index()
)
df_melted["percentage"] = df_melted.groupby(["default"])["count"].apply(
    lambda x: x / x.sum() * 100
)
fig = px.bar(
    df_melted,
    x="default",
    y="count",
    color="y",
    barmode="group",
    title="default Distribution by Subscription",
    text=df_melted["percentage"].apply(lambda x: f"{x:.2f}%"),
)
fig.show()

# %% [markdown]
# The majority of clients who said "no" to defaulting on a loan did not subscribe, but there is a small fraction who did.
# Clients with "unknown" default status have a lower subscription rate compared to "no," while those with "yes" are negligible.
# This suggests default status might not be a strong predictor but "unknown" could indicate uncertainty. There is only 3 clients with default as 'yes' and they chose not subscribe. There are no clients with default as 'yes' and subscription is 'yes'
#

# %% [markdown]
# Binning pdays makes it easier to understand and visualize how recency affects conversion rates by grouping raw values into intuitive categories. The bins clearly show that clients previously contacted more recently are significantly more likely to subscribe when compared to those who were contacted today only.

# %%
bins = [-1, 0, 7, 14, 30, 1000]  # Extend 1000 to capture 999 correctly
labels = ["Today", "Last Week", "Past Two Weeks", "Past Month", "Not Contacted"]
df["pdays_binned"] = pd.cut(df["pdays"], bins=bins, labels=labels)
print(df.groupby(["pdays_binned"])["y"].value_counts())

contingency_table = [
    [10, 5],  # Today
    [764, 398],  # Last Week
    [157, 119],  # Past Two Weeks
    [36, 26],  # Past Month
    [3673, 36000],  # Not Contacted
]

# Perform chi-square test
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

print(f"Chi-Square Statistic: {chi2:.2f}, P-Value: {p_value:.4f}")

# Interpret results
if p_value < 0.05:
    print("Statistically significant association between pdays and subscription.")
else:
    print("No statistically significant association between pdays and subscription.")

# %%
df_melted

# %%
df_melted = (
    df.groupby(["y", "pdays"])["pdays"]
    .agg({"count"})
    .sort_values("count", ascending=False)
    .reset_index()
)
df_melted["pdays"].loc[df_melted.pdays == 999] = -1
df_melted["percentage"] = df_melted.groupby(["pdays"])["count"].apply(
    lambda x: x / x.sum() * 100
)
fig = px.bar(
    df_melted,
    x="pdays",
    y="count",
    color="y",
    barmode="group",
    title="pdays Distribution by Subscription",
    text=df_melted["percentage"].apply(lambda x: f"{x:.2f}%"),
)
fig.show()

# %%
df_melted = (
    df.groupby(["y", "pdays_binned"])["pdays_binned"]
    .agg({"count"})
    .sort_values("count", ascending=False)
    .reset_index()
)
df_melted["percentage"] = df_melted.groupby(["pdays_binned"])["count"].apply(
    lambda x: x / x.sum() * 100
)
fig = px.bar(
    df_melted,
    x="pdays_binned",
    y="count",
    color="y",
    barmode="group",
    title="pdays_binned Distribution by Subscription",
    text=df_melted["percentage"].apply(lambda x: f"{x:.2f}%"),
)
fig.show()

# %% [markdown]
# From the above there's statistically significant evidence that the more recently the last call from the financial institution was the more likely the person is to subscribe.

# %%
print(df.groupby(["poutcome"])["y"].value_counts())

contingency_table = [
    [605, 3647],  # failure
    [3141, 32422],  # nonexistent
    [894, 479],  # success
]


# Perform chi-square test


chi2, p_value, dof, expected = chi2_contingency(contingency_table)


print(f"Chi-Square Statistic: {chi2:.2f}, P-Value: {p_value:.4f}")


# Interpret results


if p_value < 0.05:

    print("Statistically significant association between poutcome and subscription.")


else:

    print("No statistically significant association between poutcome and subscription.")

# %% [markdown]
# The data suggests that if you previously contacted a person and they subscribed, they are highly likely to subscribe again. While clients who initially said "no" are slightly more likely to subscribe compared to first-time contacts, the chances remain relatively low. This indicates that developing a rapport with the client, particularly through successful prior interactions, can significantly aid in increasing conversion rates.

# %%
print(df.groupby(["poutcome", "previous"])["y"].value_counts())

df_melted = df.groupby(["poutcome", "previous", "y"])["y"].agg({"count"}).reset_index()


df_melted["percentage"] = df_melted.groupby(["poutcome", "previous"])["count"].apply(
    lambda x: x / x.sum() * 100
)


# Create a bar plot with plotly
fig = px.bar(
    df_melted,
    x="previous",
    y="count",
    color="y",
    facet_col="poutcome",
    barmode="group",
    title="Count of Subscription Outcome (y) by Previous Contacts and Poutcome",
    labels={
        "previous": "Number of Previous Contacts",
        "count": "Count",
        "y": "Subscription Outcome",
    },
    text=df_melted["percentage"].apply(lambda x: f"{x:.2f}%"),
)


fig.show()

# %% [markdown]
# It appears that as the number of prior contacts increases, clients become more likely to say "yes" or convert to a subscription. However, beyond a certain point, excessive outreach (5+ contacts) leads to diminishing returns and potential client fatigue. The optimal number of prior contacts seems to be 3, where the likelihood of conversion is highest, suggesting that prior engagement is beneficial up to this threshold.
#
# It makes sense to set previous as a categorical variable. There a few unique values, it has a non-linear relationship with y and it is easier to interpret the impact of each group within the category on y.

# %%
df["previous"] = pd.Categorical(
    df["previous"], categories=[0, 1, 2, 3, 4, 5, 6, 7], ordered=True
)

for feat in df.select_dtypes(exclude=[np.number]).columns.drop(
    ["y", "poutcome", "previous", "default", "pdays_binned"]
):
    df_melted = (
        df.groupby(["y", feat])[feat]
        .agg({"count"})
        .sort_values("count", ascending=False)
        .reset_index()
    )
    df_melted["percentage"] = df_melted.groupby([feat])["count"].apply(
        lambda x: x / x.sum() * 100
    )
    fig = px.bar(
        df_melted,
        x=feat,
        y="count",
        color="y",
        barmode="group",
        title=f"{feat} Distribution by Subscription",
        text=df_melted["percentage"].apply(lambda x: f"{x:.2f}%"),
    )

    fig.show()

# %% [markdown]
# 1. Job Distribution by Subscription
# Subscription rates appear relatively higher among students, retired individuals, and unemployed clients compared to other job categories.
#
# 2. Marital Distribution by Subscription
# Married clients account for the majority of "no" outcomes, but single clients have a slightly higher proportion of "yes" outcomes compared to divorced or married individuals. Clients in the "unknown" group rarely subscribed, this may not be a meaningful group.
#
# 3. Education Distribution by Subscription
# Clients with higher education tend to have higher subscription rates over those with basic education.
#
# 4. Housing Distribution by Subscription
# Clients with a housing loan are less likely to subscribe compared to those without one.
#
# 5. Loan Distribution by Subscription
# Clients' loan status is negatively correlated with their subscription likelihood.
#
# 6. Contact Distribution by Subscription
# Clients contacted via cellular have a significantly higher proportion of subscriptions compared to telephone contacts.
#
# 7. Month Distribution by Subscription
# Subscription rates seem relatively consistent across months, much like days of the week this may not be a good feature to put in the model as it's not very discriminatory.
#
# 8. Day of Week Distribution by Subscription
# Subscription outcomes are relatively consistent across all days of the week, with a similar proportion of "yes" and "no."
# The day of the week may not significantly impact subscription likelihood. As mentioned before this feature can and will be dropped.

# %%
print(
    f'average age of subscribers {df.loc[df["y"] == "yes"]["age"].mean():.2f}\naverage age of non-subscribers {df.loc[df["y"] =="no"]["age"].mean():.2f}\naverage age of clients {df["age"].mean():.2f}'
)


print(skew(np.log(df["age"])))


sns.histplot(
    np.log(df[df["y"] == "yes"]["age"]),
    kde=True,
    label="yes",
    color="blue",
    alpha=0.6,
)


sns.histplot(
    np.log(df[df["y"] == "no"]["age"]),
    kde=True,
    label="no",
    color="red",
    alpha=0.6,
)
plt.legend()


plt.title("Age Distribution for Subscription Outcomes")


plt.show()


# QQ-plot for one group


stats.probplot(np.log(df[df["y"] == "yes"]["age"]), dist="norm", plot=plt)


plt.title("QQ-Plot for Age (yes)")


plt.show()


stats.probplot(np.log(df[df["y"] == "no"]["age"]), dist="norm", plot=plt)


plt.title(f"QQ-Plot for Age (no)")


plt.show()

# Define bins and labels
age_bins = [16, 30, 45, 60, 99]  # Adjusted based on density plot
age_labels = [
    "Young (17-30)",
    "Early Adults (31-45)",
    "Middle-Aged (46-60)",
    "Seniors (60+)",
]

# Apply binning to the 'age' column
df["age_binned"] = pd.cut(df["age"], bins=age_bins, labels=age_labels, right=True)

# %%
for feat in df.select_dtypes(include=[np.number]).columns.drop(["pdays", "age"]):
    x_yes = df[df["y"] == "yes"][feat]
    x_no = df[df["y"] == "no"][feat]

    fig = ff.create_distplot(
        [x_yes, x_no],
        group_labels=["Yes", "No"],  # Legends for the classes
        show_hist=False,  # Hide the histogram
        show_rug=False,
    )
    fig.update_layout(
        title=f"Density Plot of {feat} by Subscription Outcome (Y)",
        xaxis_title=f"{feat}",
        yaxis_title="Density",
    )
    fig.write_image(f"Density Plot of {feat} by Subscription Outcome.png")
    fig.show()

# %% [markdown]
# 1. Age
# As mentioned above both 'yes' and 'no' have an average of about 40 yrs. However, we do see that those over the age of 60+ are more likely to subscribe than say no. when compared to those between 30-59 yrs of age. Age could be discriminative feature for those 60+.
#
# 2. Duration
# "Yes" responses tend to have longer call durations, while shorter calls are majority associated with "no" responses. However, because duration is only known after the call is made we can not include it in our model because it will lead to data leakage. If model uses duration to predict y, you would need the duration of the call beforehand.
#
# 3. Campaign
# Most responses occur after 1-2 contacts, and both "yes" and "no" responses drops significantly as the number of contacts increases. This implies that the optimal number of contacts is around 1-3. I noticed the right skew of the campaign feature and will adjust this with a boxcox transformation.
#
# 4. Employment Variation Rate (emp.var.rate)
# Negative employment variation rates are more associated with "yes" responses, while positive rates correlate with "no." Clients may want a safer option for their money during economic uncertainty. The clear separation of densities indicates this is a useful feature for prediction.
#
# 5. Consumer Price Index (cons.price.idx)
# We have peaks in "no" responses at certain price levels meaning specific price levels are negatively associated with subscriptions. We see that "Yes" responses are less concentrated at these peaks and more spread out. This suggests that clients are less likely to subscribe when the consumer price index is at certain high levels, possibly reflecting economic conditions or reduced purchasing power.
#
# 6. Consumer Confidence Index (cons.conf.idx)
# Very similar to cpi, clients with a more stable or higher consumer confidence index are more likely to subscribe. The sharp peaks in "no" responses suggest that low confidence is correlated with non-subscription.
#
# 7. Euribor 3-Month Rate (euribor3m)
# Higher Euribor rates is strongly correlated with "no" responses, suggesting that clients are less likely to subscribe during periods of high borrowing costs. "Yes" responses are concentrated at lower Euribor rates, indicating that favorable financial conditions encourage subscription. This aligns with the economic relationship between lower interest rates and higher investment activity.
#
# 8. Number of Employees (nr.employed)
# "No" responses are concentrated at higher employee levels while "yes" responses are slightly more concentrated at lower levels.
# This suggests that larger employment levels (possibly indicating economic stability) reduce the need for clients to invest in term deposits.
#

# %%
df[
    ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
].corr()

# %%
# Updated Hyperparameter Grids
param_grids = {
    "logistic_regression": {
        "penalty": ["l2"],  # Regularization to avoid overfitting
        "C": [0.01, 0.1, 1, 10, 100],  # Wider range for regularization strength
        "solver": ["liblinear", "lbfgs"],
        "class_weight": ["balanced"],  # Adjust weights for imbalance
    },
    "random_forest": {
        "n_estimators": [
            100,
            200,
            300,
            400,
            500,
        ],  # Increase number of trees for stability
        "max_depth": [10, 20, 30, None],  # Add more depth levels
        "min_samples_split": [2, 5, 10],  # Higher values for pruning
        "min_samples_leaf": [1, 2, 5, 10],  # Wider range to handle imbalance
        "max_features": ["auto", "sqrt", "log2"],  # Test common feature subsets
        "criterion": ["gini", "entropy"],  # Evaluate both splitting criteria
        "class_weight": ["balanced", {0: 1, 1: 8.09}],  # Include custom weight
    },
    "xgboost": {
        "n_estimators": [100, 200, 300, 500],  # More estimators for complex patterns
        "max_depth": [3, 5, 7, 10, 20],  # Deeper trees for imbalance
        "learning_rate": [0.01, 0.1, 0.2],  # Focus on smaller learning rates
        "scale_pos_weight": [8, 10, 15],  # Balances the positive class weight
        "subsample": [0.8, 1.0],  # Randomly sample training data
        "colsample_bytree": [0.8, 1.0],  # Randomly sample features per tree
    },
    "naive_bayes": {},  # Naive Bayes has no hyperparameters to tune
    "isolation_forest": {
        "n_estimators": [100, 200, 300],  # More estimators for stability
        "max_samples": [0.5, 0.8, 1.0],  # Sampling more data for robust training
        "contamination": [0.05, 0.1, 0.2],  # Tune based on outlier expectations
        "random_state": [42],
    },
}

# %%
# features to drop
drop_features = [
    # "age",
    # "pdays",
    "duration",
    "y",
    # "nr.employed",
    # "emp.var.rate",
    # "default",
    # "age_binned",
    # "pdays_binned",
]
potential_drop = ["month", "day_of_week"]
X = df.drop(columns=drop_features)
X["previous"] = X["previous"].astype(int)
y = df["y"].map({"yes": 1, "no": 0})


categorical_cols = X.select_dtypes(include=["object", "category"]).columns
numerical_cols = X.select_dtypes(include=["number"]).columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
# Train-test split with stratified sampling


# Preprocessing function
def preprocess_data(X, y=None, model_type=None, resample=False, feature_list=[]):
    train = X[0]
    test = X[1]
    train_idx = train.index
    test_idx = test.index
    full_data = pd.concat([train, test])

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "scaler",
                StandardScaler(),
                (
                    [cat for cat in numerical_cols if cat in feature_list]
                    if feature_list
                    else numerical_cols
                ),
            ),
        ],
        remainder="passthrough",
    )
    full_data = pd.DataFrame(preprocessor.fit_transform(full_data))
    return full_data.loc[train_idx], full_data.loc[test_idx]


# Feature selection function
def select_features(model, X_train, y_train, feature_cols, top_n=40):
    """
    Select the top N features based on feature importance from the given model.
    """
    model.fit(X_train, y_train)
    feature_importances = model.feature_importances_
    important_features = pd.DataFrame(
        {"Feature": feature_cols, "Importance": feature_importances}
    ).sort_values(by="Importance", ascending=False)

    print(
        f"Top {len(important_features)} Features:\n",
        important_features.head(len(important_features)),
    )
    fig = px.bar(important_features, x="Feature", y="Importance")
    fig.show()
    selected_features = important_features.head(top_n)["Feature"].tolist()
    y_pred_train = model.predict(X_train)

    recall = recall_score(y_train, y_pred_train, pos_label=1)
    precision = precision_score(y_train, y_pred_train, pos_label=1)
    print(f"Training Recall: {recall:.4f}")
    print(f"Training Precision: {precision:.4f}")
    print(classification_report(y_train, y_pred_train))

    important_features["Cumulative_Importance"] = important_features[
        "Importance"
    ].cumsum()

    # Select features contributing to 95% of total importance
    selected_features_reduced = important_features[
        important_features["Cumulative_Importance"] <= 0.9
    ]["Feature"].tolist()
    print(
        f"{len(selected_features_reduced)} Selected Features: {selected_features_reduced}"
    )
    return selected_features, selected_features_reduced, important_features, model


# Train-test split with stratified sampling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Models with imbalance handling
models = {
    "logistic_regression": LogisticRegression(random_state=42, class_weight="balanced"),
    "random_forest": RandomForestClassifier(
        random_state=42, class_weight="balanced", n_jobs=-1
    ),
    "xgboost": XGBClassifier(
        objective="binary:logistic",
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
        scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
        n_jobs=-1,
    ),
    "naive_bayes": GaussianNB(),
    "isolation_forest": IsolationForest(
        random_state=42,
        contamination=len(y_train[y_train == 1]) / len(y_train),
        n_jobs=-1,
    ),
}

mod = "random_forest"

X_train_bf, X_test_bf = preprocess_data(
    X=[X_train.copy(), X_test.copy()], model_type=mod
)

ros = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train_bf, y_train)


feature_importance_dict = {}

for train_x, train_y, df_name in [
    (X_train_bf, y_train, "regular"),
    (X_train_resampled, y_train_resampled, "oversample"),
]:
    # Feature selection
    print(f"\n--- Feature Selection with {mod} ---")
    top_features, top_features_reduced, important_features, model = select_features(
        models[mod], train_x, train_y, list(X_train.columns)
    )
    feature_importance_dict[df_name] = [
        important_features,
        top_features,
        top_features_reduced,
    ]

# %% [markdown]
# nr.employed and emp.var.rate are highly correlated with euribor3m. I'll use pca to create a new feature combining all 3.
#

# %%
economic_features = ["emp.var.rate", "euribor3m", "nr.employed"]

scaler = StandardScaler()
X_econ_scaled = pd.DataFrame(
    scaler.fit_transform(X[economic_features]), columns=economic_features
)


pca = PCA()
X_economic_pca = pca.fit_transform(X_econ_scaled)

# Check explained variance ratio
explained_variance = pca.explained_variance_ratio_
cumulative_variance = explained_variance.cumsum()
print("Explained Variance Ratio:", explained_variance)
print("Cumulative Variance:", cumulative_variance)

# Choose the number of components to retain (e.g., 95% variance)
n_components = (cumulative_variance <= 0.95).sum() + 1
pca = PCA(n_components=n_components)
X_economic_pca = pca.fit_transform(X_economic_pca)

# Convert PCA-transformed data to DataFrame
pca_columns = [f"economic_PC{i+1}" for i in range(n_components)]
X_economic_pca_df = pd.DataFrame(X_economic_pca, columns=pca_columns, index=X.index)
X["econ_feat"] = X_economic_pca_df
X.drop(columns=economic_features, inplace=True)

# %%
# Train-test split with stratified sampling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Models with imbalance handling
models = {
    "logistic_regression": LogisticRegression(random_state=42, class_weight="balanced"),
    "random_forest": RandomForestClassifier(
        random_state=42, class_weight="balanced", n_jobs=-1
    ),
    "xgboost": XGBClassifier(
        objective="binary:logistic",
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
        scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
        n_jobs=-1,
    ),
    "naive_bayes": GaussianNB(),
    "isolation_forest": IsolationForest(
        random_state=42,
        contamination=len(y_train[y_train == 1]) / len(y_train),
        n_jobs=-1,
    ),
}

mod = "random_forest"

X_train_bf, X_test_bf = preprocess_data(
    X=[X_train.copy(), X_test.copy()], model_type=mod, feature_list=list(X.columns)
)

ros = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train_bf, y_train)


feature_importance_dict = {}

for train_x, train_y, df_name in [
    # (X_train_bf, y_train, "regular"),
    (X_train_resampled, y_train_resampled, "oversample"),
]:
    # Feature selection
    print(f"\n--- Feature Selection with {mod} ---")
    top_features, top_features_reduced, important_features, model = select_features(
        models[mod], train_x, train_y, list(X_train.columns)
    )
    feature_importance_dict[df_name] = [
        important_features,
        top_features,
        top_features_reduced,
    ]

# %% [markdown]
# I was expecting pdays to and previous to be up more the list of important features.

# %%
df_copy = pd.concat([X.copy(deep=True), y], axis=1)
df_copy["previous"] = df_copy["previous"].astype(int)


fig, ax = plt.subplots(3, 3, figsize=(16, 12))
ax = ax.ravel()

for i, col in enumerate(
    [
        "pdays",
        "econ_feat",
        "cons.price.idx",
        "campaign",
        "cons.conf.idx",
        "age",
        "previous",
    ]
):
    sns.boxplot(df_copy[col], ax=ax[i])
    ax[i].set_xlabel(col)

df_copy["campaign_cut_7"] = np.where(df_copy.campaign <= 7, 1, 0)

# %%
df_cut_7 = df_copy.loc[df_copy["campaign_cut_7"] == 1].copy().reset_index(drop=True)

# Step 3: Select correlated features for imputation
imputation_features = [
    "pdays_numeric",
    "econ_feat",
    "cons.price.idx",
    "campaign",
    "cons.conf.idx",
    "age",
    "previous",
]  # Add other relevant numeric features

for df_i in [df_copy, df_cut_7]:
    df_i["pdays_missing"] = np.where(df_i["pdays"] == 999, 1, 0)
    df_i["pdays_numeric"] = np.where(df_i["pdays"] == 999, np.nan, df_i["pdays"])
    df_i["pdays_med_impute"] = np.where(
        df_i["pdays"] == 999, df_i["pdays_numeric"].median().round(), df_i["pdays"]
    )

    # Step 4: Standardize the features
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_i[imputation_features]), columns=imputation_features
    )

    # Step 5: Apply KNN Imputation
    imputer = KNNImputer(n_neighbors=5, weights="uniform")
    df_imputed_scaled = imputer.fit_transform(df_scaled)

    # Step 6: Restore imputed values to the original scale
    df_imputed_knn = pd.DataFrame(
        scaler.inverse_transform(df_imputed_scaled), columns=imputation_features
    )
    df_i["pdays_knn_impute"] = df_imputed_knn["pdays_numeric"].round()

    print(df_i.shape)

# %%
for df_i in [df_copy, df_cut_7]:
    for col in ["pdays_med_impute", "pdays_knn_impute"]:
        fig = px.histogram(df_i[col])
        fig.show()

# %% [markdown]
# My assumptions when we use the full dataset is KNN imputation is best choice for pdays because it adhere's to the natural right-skewed distribution of pdays which aligns with our initial EDA findings. Median imputation oversimplifies the data causing a lack of variability and the loss of information.
#

# %% [markdown]
# The distribution of campaign suggests that most clients are called less than 10 times, while distribution of previous suggests that reasonable cutoff for repeated contact is 7. I will confirm via the model performance if clients who received than 7 calls should be included in the modelling data or not.

# %%
df_copy.y.sum()

# %%
# features to drop
drop_features = [
    # "age",
    "pdays",
    "pdays_numeric",
    "pdays_med_impute",
    # "duration",
    "y",
    # "nr.employed",
    # "emp.var.rate",
    # "default",
    # "age_binned",
    # "pdays_binned",
]
potential_drop = ["month", "day_of_week"]
X = df_copy.copy()
X = df_copy.drop(columns=drop_features)
X["previous"] = X["previous"].astype(int)
y = df_copy.y


# Train-test split with stratified sampling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Models with imbalance handling
models = {
    "logistic_regression": LogisticRegression(random_state=42, class_weight="balanced"),
    "random_forest": RandomForestClassifier(
        random_state=42, class_weight="balanced", n_jobs=-1
    ),
    "xgboost": XGBClassifier(
        objective="binary:logistic",
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
        scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
        n_jobs=-1,
    ),
    "naive_bayes": GaussianNB(),
    "isolation_forest": IsolationForest(
        random_state=42,
        contamination=len(y_train[y_train == 1]) / len(y_train),
        n_jobs=-1,
    ),
}

mod = "random_forest"

X_train_bf, X_test_bf = X_train.copy(), X_test.copy()

ros = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train_bf, y_train)


feature_importance_dict = {}

for train_x, train_y, df_name in [
    (X_train_bf, y_train, "regular"),
    (X_train_resampled, y_train_resampled, "oversample"),
]:
    # Feature selection
    print(f"\n--- Feature Selection with {mod} ---")
    top_features, top_features_reduced, important_features, model = select_features(
        models[mod], train_x, train_y, list(X_train.columns)
    )
    feature_importance_dict[df_name] = [
        important_features,
        top_features,
        top_features_reduced,
    ]

# %% [markdown]
# # model with all the feature engineer removed

# %%
# Cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# # Grid search and evaluation
best_models1 = {}
results = []

for train_x, train_y, df_name in [
    # (X_train_bf, y_train, "regular"),
    (X_train_resampled, y_train_resampled, "oversample"),
]:
    for feature_list in [
        feature_importance_dict[df_name][1],
        feature_importance_dict[df_name][2],
    ]:
        for name, model in models.items():
            print(f"\nTraining {name}...")
            best_recall = 0

            param_grid = param_grids.get(name, {})
            # Perform grid search for models
            if name == "isolation_forest":

                for params in ParameterGrid(param_grid):
                    print(f"Training Isolation Forest with params: {params}")
                    model.fit(train_x)

                    # Predictions for train and test
                    y_pred_train = model.predict(train_x)
                    y_pred_test = model.predict(X_test_bf)

                    # Convert predictions (-1: anomaly, 1: inlier) to binary labels (1: yes, 0: no)
                    y_pred_train = (y_pred_train == -1).astype(int)
                    y_pred_test = (y_pred_test == -1).astype(int)

                    # Evaluate on training data
                    recall_train = recall_score(train_y, y_pred_train, pos_label=1)
                    precision_train = precision_score(
                        train_y, y_pred_train, pos_label=1
                    )

                    # Evaluate on test data
                    recall_test = recall_score(y_test, y_pred_test, pos_label=1)
                    precision_test = precision_score(y_test, y_pred_test, pos_label=1)

                    # Update best model based on recall
                    if recall_test > best_recall:
                        best_models1[name] = best_model
                        best_params = params

            else:
                grid = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    scoring="recall",  # Focus on recall
                    cv=cv,
                    n_jobs=-1,
                    verbose=1,
                )
                grid.fit(train_x, train_y)
                print(f"Best parameters for {name}: {grid.best_params_}")
                best_model = grid.best_estimator_
                best_models1[name] = best_model

                y_pred_train = best_model.predict(train_x)
                y_pred_test = best_model.predict(X_test_bf)

            # Evaluate on training data
            recall_train = recall_score(train_y, y_pred_train, pos_label=1)
            precision_train = precision_score(train_y, y_pred_train, pos_label=1)
            f1_train = f1_score(train_y, y_pred_train, pos_label=1)
            accuracy_train = accuracy_score(train_y, y_pred_train)
            print(classification_report(train_y, y_pred_train))

            # Evaluate on test data
            recall_test = recall_score(y_test, y_pred_test, pos_label=1)
            precision_test = precision_score(y_test, y_pred_test, pos_label=1)
            f1_test = f1_score(y_test, y_pred_test, pos_label=1)
            accuracy_test = accuracy_score(y_test, y_pred_test)
            print(classification_report(y_test, y_pred_test))

            # Save results
            results.append(
                {
                    "Model": name,
                    "Best Params": (
                        grid.best_params_ if name != "isolation_forest" else best_params
                    ),
                    "Train Recall": recall_train,
                    "Train Precision": precision_train,
                    "Train F1": f1_train,
                    "Train Accuracy": accuracy_train,
                    "Test Recall": recall_test,
                    "Test Precision": precision_test,
                    "Test F1": f1_test,
                    "Test Accuracy": accuracy_test,
                    "Data Type": df_name,
                    "Feature Length": len(feature_list),
                }
            )


# Convert results to a DataFrame
results_df2 = pd.DataFrame(results)


# %%
# Evaluate Model Performance
def evaluate_model_performance(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_prob = (
        model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    )

    # Metrics
    recall = recall_score(y_test, y_pred, pos_label=1)
    precision = precision_score(y_test, y_pred, pos_label=1)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob) if y_pred_prob is not None else None

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    if roc_auc:
        print(f"ROC-AUC: {roc_auc:.4f}")

    # Create Charts and Graphs
    plot_roc_auc_curve(y_test, y_pred_prob) if roc_auc else None
    plot_confusion_matrix(y_test, y_pred)

    return recall, precision, accuracy, roc_auc


# Plot ROC-AUC Curve
def plot_roc_auc_curve(y_test, y_pred_prob):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr,
        tpr,
        color="blue",
        label=f"ROC Curve (AUC = {roc_auc_score(y_test, y_pred_prob):.4f})",
    )
    plt.plot([0, 1], [0, 1], color="red", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-AUC Curve")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()


# Plot Confusion Matrix
def plot_confusion_matrix(y_test, y_pred):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


# Example Usage
recall_test, precision_test, accuracy_test, roc_auc_test = evaluate_model_performance(
    model=best_models4["random_forest"], X_test=X_test_bf, y_test=y_test
)

# %% [markdown]
# In the model building process, I addressed the imbalance in the target class by selecting models known to perform well on imbalanced data and leveraging oversampling techniques. However, oversampling led to overfitting in boosted models and a decline in Random Forest's recall and precision. Feature engineering had minimal impact, as models with and without engineered features performed similarly. The best model achieved a recall of 63.36% and a precision of 43.75% for the minority class, with an overall accuracy of 86.70% and an ROC-AUC of 0.8159. While the model shows promise in identifying the minority class, the relatively low precision indicates room for improvement in reducing false positives. Further steps may include advanced sampling techniques, cost-sensitive learning, or exploring additional features to boost performance.

# %% [markdown]
# 2. Once your model is built, think about how you’d deploy it in production, which technologies you’d use?
#
#     I would build production-level code for my model using sklearn.compose to create an ML pipeline that includes preprocessing, feature engineering, and model training steps. The pipeline would be containerized with Docker and deployed to Azure Kubernetes Service (AKS) for scalability and resilience, with auto-scaling enabled to handle growing workloads. The deployment process would be automated using GitLab CI/CD, ensuring smooth updates and rollback capabilities.
#
#     For monitoring, I would track prediction drift, data drift, and target drift, set performance thresholds, and implement alerts for anomalies. I’d also monitor system performance (latency, resource usage) and set up retraining pipelines triggered by drift or performance degradation. Using tools like Azure Monitoring, I would ensure ongoing model performance meets business requirements, and alerts notify stakeholders if issues arise. The model would be deployed as an API, allowing seamless integration into a website, dashboards, or email.
#
# 3. How the marketing team can use/consume your model?
#
#     I believe the marketing can leverage this model through an integrated architecture that monitors the client database and dynamically identifies high-priority clients for contact. During a campaign, the model would generate a daily subset of clients with a high likelihood of subscription conversion. This list could be delivered via automated emails or integrated into a dedicated app or dashboard, where agents can view and prioritize their calls.
#
#
# 4. How do you decide when your model needs to be refreshed/retrained?
#
#     There are 3 key metrics that I would be monitoring in order to determine if the model needs a refresh or retraining. Those are prediction drift, data drift, and target drift.  If thresholds for these metrics are exceeded, alerts are triggered to evaluate or retrain the model. For time-dependent features like economic variables, regular retraining is scheduled (e.g., monthly or quarterly) to account for changing trends. Performance metrics like precision, recall, or AUC are also monitored in production, and significant degradation beyond acceptable level signals the need for a model refresh.
#
#
# 5. What other data sources you think can help improve your model?
#
#     •  Demographic Information: Client income, family size, and geographic location could help identify financial priorities and constraints.
#     •  Service Familiarity: Information about the client’s familiarity with the services being offered, including prior usage or subscriptions at other institutions, could provide context for their likelihood of subscribing.
#     •  Client Sentiment: Data on the client’s perception of the financial institution, such as survey responses, social media sentiment, or customer feedback, could help uncover clients’ trust and satisfaction levels.
#     •  Competitor Analysis: Insights into whether clients are subscribed to similar services at other institutions and how those services compare.
#     •  Transaction Data: Data on spending patterns to identify clients who are more likely to be subscribers.
#
