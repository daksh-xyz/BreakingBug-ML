# Imported all the correct files
import this
import antigravity
import pyjokes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from yellowbrick.cluster import KElbowVisualizer
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestRegressor, AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore') 
import seaborn as sns

# Changed the path to dataset file
df = pd.read_csv("dataset.csv")

df.head()

df.info()

df.shape

df['id'].min(), df['id'].max()

df['age'].min(), df['age'].max()

df['age'].describe()


custom_colors = ["#FF5733", "#3366FF", "#33FF57"]

# logic correction
bin_edges = np.histogram_bin_edges(df['age'], bins=3)
for i in range(len(bin_edges) - 1):
    sns.histplot(df['age'], kde=True, color=custom_colors[i], fill=True)

plt.show()

sns.histplot(df['age'], kde=True)
plt.axvline(df['age'].mean(), color='Red')
plt.axvline(df['age'].mode()[0], color='Blue')
plt.axvline(df['age'].median(), color= 'Green')

print('Mean', df['age'].mean())
print('Median', df['age'].median())
print('Mode', df['age'].mode()[0]) # Added list index to display only the mode

fig = px.histogram(data_frame=df, x='age', color= 'sex')
fig.show()

df['sex'].value_counts()

male_count = df['sex'].value_counts()["Male"]
female_count = df['sex'].value_counts()["Female"]

total_count = male_count + female_count

male_percentage = (male_count/total_count)*100
female_percentages = (female_count/total_count)*100

print(f'Male percentage in the data: {male_percentage:.2f}%')
print(f'Female percentage in the data : {female_percentages:.2f}%')

difference_percentage = ((male_count - female_count)/female_count) * 100
print(f'Males are {difference_percentage:.2f}% more than female in the data.')

726/194

df.groupby('sex')['age'].value_counts()

df['dataset'].unique() # Changed counts() to unique()
df['dataset'].value_counts() # added value_counts()

df.head()

agg_df = df.groupby(['dataset', 'sex']).size().reset_index(name='count') # Added group by function to display count of males and females

fig = px.bar(agg_df, x='dataset', y='count', color='sex')
fig.show()

print (df.groupby('sex')['dataset'].value_counts())

"""make a plot of age column using plotly and coloring by dataset"""

fig = px.histogram(data_frame=df, x='age', color= 'dataset')
fig.show()

print("___________________________________________________________")
print("Mean of the dataset: ",df['age'].mean())
print("___________________________________________________________")
print("Median of the dataset: ",df['age'].median())
print("___________________________________________________________")
print("Mode of the dataset: ",df['age'].mode()[0]) # Added index to mode to display only mode
print("___________________________________________________________")

df['cp'].value_counts()

sns.countplot(df, x='cp', hue= 'sex')

sns.countplot(df,x='cp',hue='dataset')

"""Draw the plot of age column group by cp column"""

fig = px.histogram(data_frame=df, x='age', color='cp')
fig.show()

df['trestbps'].describe()

print(f"Percentage of missing values in trestbps column: {df['trestbps'].isnull().sum() /len(df) *100:.2f}%")

imputer1 = IterativeImputer(max_iter=10, random_state=42)

imputer1.fit(df[['trestbps']])

df['trestbps'] = imputer1.transform(df[['trestbps']])

print(f"Missing values in trestbps column: {df['trestbps'].isnull().sum()}")

df.info()

(df.isnull().sum()/ len(df)* 100).sort_values(ascending=False)

imputer2 = IterativeImputer(max_iter=10, random_state=42)

imputer2.fit(df[['ca', 'oldpeak', 'chol', 'thalch']])

df[['ca', 'oldpeak', 'chol', 'thalch']] = imputer2.transform(df[['ca', 'oldpeak', 'chol', 'thalch']])

(df.isnull().sum()/ len(df)* 100).sort_values(ascending=False)

print(f"The missing values in thal column are: {df['thal'].isnull().sum()}")

df['thal'].value_counts()

df.tail()

df.isnull().sum()[df.isnull().sum()>0].sort_values(ascending=True)



missing_data_cols = df.isnull().sum()[df.isnull().sum()>0].index.tolist()

missing_data_cols

# find categorical Columns
cat_cols = df.select_dtypes(include='object').columns.tolist()
cat_cols

# find Numerical Columns
Num_cols = df.select_dtypes(exclude='object').columns.tolist()
Num_cols

print(f'categorical Columns: {cat_cols}')
print(f'numerical Columns: {Num_cols}')



# Find columns
categorical_cols = ['sex', 'dataset', 'cp', 'restecg', 'slope', 'thal']
bool_cols = ['fbs', 'exang']
numerical_cols = ['id', 'age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca', 'num']

def impute_categorical_missing_data(passed_col):

    # if passed_col not in categorical_cols:
    #     raise ValueError(f"{passed_col} is not in the list of categorical columns")

    df_null = df[df[passed_col].isnull()]
    df_not_null = df[df[passed_col].notnull()]

    X = df_not_null.drop(categorical_cols, axis=1)
    y = df_not_null[passed_col]


    label_encoder = LabelEncoder()
    onehotencoder = OneHotEncoder(sparse=False, drop='first')


    for col in X.select_dtypes(include=['object']).columns:
        X[col] = label_encoder.fit_transform(X[col].astype(str))
    if y.dtype == 'object':
        y = label_encoder.fit_transform(y.astype(str))


    iterative_imputer = IterativeImputer(max_iter=10, random_state=42)
    X = pd.DataFrame(iterative_imputer.fit_transform(X), columns=X.columns)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    rf_classifier = RandomForestClassifier(random_state=42)
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)
    acc_score = accuracy_score(y_test, y_pred)

    print(f"The feature '{passed_col}' has been imputed with {round(acc_score * 100, 2)}% accuracy\n")
    X = df_null.drop(categorical_cols, axis=1)

    if len(df_null) > 0:
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = label_encoder.fit_transform(X[col].astype(str))
        X = pd.DataFrame(iterative_imputer.transform(X), columns=X.columns)

        df_null[passed_col] = rf_classifier.predict(X)

    df_combined = pd.concat([df_not_null, df_null])

    return df_combined[passed_col]

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def impute_continuous_missing_data(df, passed_col, missing_data_cols):
    # Separate data with and without missing values
    df_null = df[df[passed_col].isnull()]
    df_not_null = df[df[passed_col].notnull()]

    X = df_not_null.drop(passed_col, axis=1)
    y = df_not_null[passed_col]

    # Identify other columns with missing values
    other_missing_cols = [col for col in missing_data_cols if col != passed_col]

    # Handle categorical features in X
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Initialize the imputer
    imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=16), add_indicator=True)

    # Impute the missing values in X
    X_imputed = imputer.fit_transform(X)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

    # Fit the RandomForestRegressor model
    rf_regressor = RandomForestRegressor(random_state=42)
    rf_regressor.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = rf_regressor.predict(X_test)
    print("MAE =", mean_absolute_error(y_test, y_pred))
    print("RMSE =", mean_squared_error(y_test, y_pred, squared=False))
    print("R2 =", r2_score(y_test, y_pred))

    # Prepare the imputer for predicting missing values
    X_null = df_null.drop(passed_col, axis=1)
    categorical_cols_null = X_null.select_dtypes(include=['object']).columns
    for col in categorical_cols_null:
        X_null[col] = LabelEncoder().fit_transform(X_null[col].astype(str))

    # Impute the missing values
    X_null_imputed = imputer.transform(X_null)
    df_null[passed_col] = rf_regressor.predict(X_null_imputed)

    # Combine the data
    df_combined = pd.concat([df_not_null, df_null], ignore_index=True)

    return df_combined

df.isnull().sum().sort_values(ascending=False)

# remove warning
import warnings
warnings.filterwarnings('ignore')

# impute missing values using our functions
for col in missing_data_cols:
    print("Missing Values", col, ":", str(round((df[col].isnull().sum() / len(df)) * 100, 2))+"%")
    if col in categorical_cols:
        df[col] = impute_categorical_missing_data(col)
    elif col in bool_cols:
        df[col] = impute_categorical_missing_data(col)
    elif col in numerical_cols:
        df[col] = impute_continuous_missing_data(col)
    else:
        pass

df.isnull().sum().sort_values(ascending=False)

print("_________________________________________________________________________________________________________________________________________________")

sns.set(rc={"axes.facecolor":"#87CEEB","figure.facecolor":"#EEE8AA"})  # Change figure background color

palette = ["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"]
cmap = ListedColormap(["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"])

num_cols = len(df.columns)
num_rows = (num_cols + 1) // 2

plt.figure(figsize=(12, num_rows * 4))

for i, col in enumerate(df.columns):
    plt.subplot(num_rows, 2, i + 1)
    if pd.api.types.is_numeric_dtype(df[col]):
        sns.boxenplot(y=df[col], color=palette[i % len(palette)])
        plt.title(col)
    else:
        sns.countplot(x=df[col], color=palette[i % len(palette)])
        plt.title(col)

plt.tight_layout()
plt.show()

plt.figure(figsize=(10,8));

df.info()

# Create subplots
num_cols = len(df.columns)
num_rows = (num_cols + 1) // 2  # Compute number of rows needed

plt.figure(figsize=(20, num_rows * 4))  # Adjust height based on number of rows

for i, col in enumerate(df.columns):
    plt.subplot(num_rows, 2, i + 1)  # Create subplot
    if pd.api.types.is_numeric_dtype(df[col]):
        sns.boxenplot(y=df[col], color=palette[i % len(palette)])  # Plot numerical data
        plt.title(col)  # Set subplot title
    else:
        sns.countplot(x=df[col], color=palette[i % len(palette)])

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

##E6E6FA

# print the row from df where trestbps value is 0
df[df['trestbps']==0]

# Remove the column because it is an outlier because trestbps cannot be zero.
df= df[df['trestbps']!=0]

sns.set(rc={"axes.facecolor": "#B76E79", "figure.facecolor": "#C0C0C0"})
modified_palette = ["#C44D53", "#B76E79", "#DDA4A5", "#B3BCC4", "#A2867E", "#F3AB60"]
cmap = ListedColormap(["#C44D53", "#B76E79", "#DDA4A5", "#B3BCC4", "#A2867E", "#F3AB60"])

plt.figure(figsize=(20,10))



# Create subplots
num_cols = len(df.columns)
num_rows = (num_cols + 1) // 2  # Compute number of rows needed

plt.figure(figsize=(20, num_rows * 4))  # Adjust height based on number of rows

for i, col in enumerate(df.columns):
    plt.subplot(num_rows, 2, i + 1)  # Create subplot
    if pd.api.types.is_numeric_dtype(df[col]):
        sns.boxenplot(y=df[col], color=modified_palette[i % len(modified_palette)])  # Plot numerical data
        plt.title(col)  # Set subplot title
    else:
        sns.countplot(x=df[col], color=modified_palette[i % len(modified_palette)])

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

plt.show()

df.trestbps.describe()

df.describe()

print("___________________________________________________________________________________________________________________________________________________________________")

# Set facecolors
sns.set(rc={"axes.facecolor": "#FFF9ED", "figure.facecolor": "#FFF9ED"})

# Define the "night vision" color palette
night_vision_palette = ["#00FF00", "#FF00FF", "#00FFFF", "#FFFF00", "#FF0000", "#0000FF"]
cmap= ListedColormap(["#00FF00", "#FF00FF", "#00FFFF", "#FFFF00", "#FF0000", "#0000FF"])

# Create subplots
num_cols = len(df.columns)
num_rows = (num_cols + 1) // 2  # Compute number of rows needed

plt.figure(figsize=(20, num_rows * 4))  # Adjust height based on number of rows

for i, col in enumerate(df.columns):
    plt.subplot(num_rows, 2, i + 1)  # Create subplot
    if pd.api.types.is_numeric_dtype(df[col]):
        sns.boxenplot(y=df[col], color=night_vision_palette[i % len(night_vision_palette)])  # Plot numerical data
        plt.title(col)  # Set subplot title
    else:
        sns.countplot(x=df[col], color=night_vision_palette[i % len(night_vision_palette)])

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

plt.show()

df.age.describe()

palette = ["#999999", "#666666", "#333333"]



sns.histplot(data=df,
             x='trestbps',
             kde=True,
             color=palette[0])
plt.title('Resting Blood Pressure')
plt.xlabel('Pressure (mmHg)')
plt.ylabel('Count')

plt.style.use('default')
plt.rcParams['figure.facecolor'] = palette[1]
plt.rcParams['axes.facecolor'] = palette[2]

# create a histplot trestbops column to analyse with sex column
sns.histplot(df, x='trestbps', kde=True, palette = "Spectral", hue ='sex')

df.info()

df.columns

df.head()

# split the data into X and y
X= df.drop('num', axis=1)
y = df['num']

# Initialize dictionary to hold the LabelEncoders for each categorical column
label_encoders = {col: LabelEncoder() for col in categorical_cols}

# Apply LabelEncoder to each categorical column
for col in categorical_cols:
    df[col] = df[col].astype(str)  # This ensures all values are strings
    df[col] = label_encoders[col].fit_transform(df[col])

# Apply OneHotEncoder to the entire dataset
onehotencoder = OneHotEncoder()
X_encoded = onehotencoder.fit_transform(df[categorical_cols]).toarray()

# Create a DataFrame with the encoded features
encoded_df = pd.DataFrame(X_encoded, columns=onehotencoder.get_feature_names_out(categorical_cols))

# Example usage for train/test split
X = encoded_df
y = df['num']  # Assume 'num' is the target column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to inverse transform the encoded data back to original
def inverse_transform(encoded_data, label_encoders, onehotencoder, categorical_cols):
    # Inverse transform OneHotEncoder
    inverse_transformed = onehotencoder.inverse_transform(encoded_data)

    # Create a DataFrame with inverse transformed data
    original_df = pd.DataFrame(inverse_transformed, columns=categorical_cols)

    # Apply inverse transformation using LabelEncoder
    for col in categorical_cols:
        original_df[col] = label_encoders[col].inverse_transform(original_df[col].astype(int))

    return original_df

# Example inverse transformation
inverse_transformed_data = inverse_transform(X_test, label_encoders, onehotencoder, categorical_cols)
print(inverse_transformed_data.head())

# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)



from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestRegressor, AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB



#importing pipeline
from sklearn.pipeline import Pipeline

# import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')

"""create a list of models to evaluate"""

models = [
    ('Logistic Regression', LogisticRegression(random_state=42)),
    ('Gradient Boosting', GradientBoostingClassifier(random_state=42)),
    ('KNeighbors Classifier', KNeighborsClassifier()),
    ('Decision Tree Classifier', DecisionTreeClassifier(random_state=42)),
    ('AdaBoost Classifier', AdaBoostClassifier(random_state=42)),
    ('Random Forest', RandomForestClassifier(random_state=42)),
    ('XGboost Classifier', XGBClassifier(random_state=42)),

    ('Support Vector Machine', SVC(random_state=42)),

    ('Naye base Classifier', GaussianNB())
]



best_model = None
best_accuracy = 0.0

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# Initialize variables to keep track of the best model and accuracy
best_accuracy = 0
best_model = None

# Iterate over the models
for name, model in models:
    # Ensure all categorical columns are uniformly converted to strings
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()

    categorical_cols = X_train_encoded.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        X_train_encoded[col] = X_train_encoded[col].astype(str)
        X_test_encoded[col] = X_test_encoded[col].astype(str)

    # Create a pipeline for each model
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False)),
        ('model', model)
    ])

    # Perform cross-validation
    scores = cross_val_score(pipeline, X_train_encoded, y_train, cv=5)

    # Calculate mean accuracy from cross-validation scores
    mean_accuracy = scores.mean()

    # Fit the pipeline on the training data
    pipeline.fit(X_train_encoded, y_train)

    # Make predictions on the test data
    y_pred = pipeline.predict(X_test_encoded)

    # Calculate test accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Print the performance metrics
    print("Model:", name)
    print("Cross-validation accuracy: ", round(mean_accuracy * 100, 2), "%")
    print("Test Accuracy: ", round(accuracy * 100, 2), "%")
    print()

    # Check if the current model has the best accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = pipeline

print("Best Model:", best_model)

print("Best Accuracy: ", round(best_accuracy * 100, 2), "%")

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.base import is_classifier

best_accuracy = 0
best_model = None

# Iterate over the models
for name, model in models:
    # Create a pipeline for each model
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False)),
        ('model', model)
    ])

    try:
        scores = cross_val_score(pipeline, X_train, y_train, cv=5, error_score='raise')

        mean_accuracy = scores.mean()

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        print("Model:", name)
        print("Cross-Validation Accuracy: ", round(mean_accuracy * 100, 2), "%")
        print("Test Accuracy: ", round(accuracy * 100, 2), "%")
        print()

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = pipeline

    except Exception as e:
        print(f"Error with model {name}: {e}")

print("Best Model:", best_model)
print("Best Accuracy: ", round(best_accuracy * 100, 2), "%")

# Retrieve the best model
print("Best Model: ", best_model)



categorical_cols = ['sex', 'dataset', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

def evaluate_classification_models(X, y, categorical_columns):
    # Encode categorical columns
    X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(),
        "NB": GaussianNB(),
        "SVM": SVC(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "GradientBoosting": GradientBoostingClassifier(),
        "AdaBoost": AdaBoostClassifier()
    }

    # Train and evaluate models
    results = {}
    best_model = None
    best_accuracy = 0.0
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = name

    return results, best_model



# Example usage:
results, best_model = evaluate_classification_models(X, y, categorical_cols)
print("Model accuracies:", results)
print("Best model:", best_model)

df.num.head()

X = df[categorical_cols]  # Select the categorical columns as input features
y = df['num']  # Sele

def hyperparameter_tuning(X, y, categorical_columns, models):
    # Define dictionary to store results
    results = {}

    # Encode categorical columns
    X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # Define parameter grids for hyperparameter tuning
    param_grids = {
        'Logistic Regression': {'C': [0.1, 1, 10, 100]},
        'KNN': {'n_neighbors': [3, 5, 7, 9]},
        'NB': {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]},
        'SVM': {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 1, 10, 100]},
        'Decision Tree': {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]},
        'Random Forest': {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]},
        'XGBoost': {'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7]},
        'GradientBoosting': {'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7]},
        'AdaBoost': {'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [50, 100, 200]}
    }

    # Perform hyperparameter tuning for each model
    for model_name, model in models.items():
        param_grid = param_grids.get(model_name, {})
        if param_grid:
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
            grid_search.fit(X_train, y_train)

            # Get best hyperparameters and evaluate on test set
            best_params = grid_search.best_params_
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Store results in dictionary
            results[model_name] = {'best_params': best_params, 'accuracy': accuracy}

    return results

# Define models dictionary
models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "NB": GaussianNB(),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier()
}
# Example usage:
results = hyperparameter_tuning(X, y, categorical_cols, models)
for model_name, result in results.items():
    print("Model:", model_name)
    print("Best hyperparameters:", result['best_params'])
    print("Accuracy:", result['accuracy'])
    print()
