import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_excel("movie_hit_flop.xlsx")
print(df.head())
print(df.shape)
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.describe())
print(df["genre"].unique())
print(df["language"].unique())
print(df["result"].value_counts())
# 1. Remove duplicates
df = df.drop_duplicates()

# 2. Drop the ID and the cheating column
df = df.drop(columns=["movie_id", "box_office_crore"])

# 3. Fix messy text (remove spaces + fix capitals)
df["genre"] = df["genre"].str.strip().str.title()
df["language"] = df["language"].str.strip().str.title()

# 4. Fill blanks
df["budget_crore"] = df["budget_crore"].fillna(df["budget_crore"].median())
df["trailer_views_million"] = df["trailer_views_million"].fillna(df["trailer_views_million"].median())
df["language"] = df["language"].fillna(df["language"].mode()[0])
df["star_power"] = df["star_power"].fillna(df["star_power"].mode()[0])

# 5. Remove impossible values
df = df[(df["budget_crore"] > 0) & (df["runtime_min"] <= 240)]

# Check
print(df["genre"].unique())
print(df["language"].unique())
print(df.isnull().sum())
print(df.shape)

df["festival_release"] = df["festival_release"].map({"Yes": 1, "No": 0})
df["sequel"] = df["sequel"].map({"Yes": 1, "No": 0})
df["result"] = df["result"].map({"Hit": 1, "Flop": 0})
df["star_power"] = df["star_power"].map({"Low": 0, "Medium": 1, "High": 2})

df = pd.get_dummies(df, columns=["genre", "language"], dtype=int)

print(df.dtypes)
print(df.isnull().sum().sum())
print(df.shape)

# Clues and answer
X = df.drop(columns=["result"])
y = df["result"]

# Split 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Test
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# Create a new movie (copy the structure of one row)
new_movie = X.iloc[[0]].copy()

# Set its details
new_movie["budget_crore"] = 250
new_movie["star_power"] = 2              # High
new_movie["festival_release"] = 1        # Yes
new_movie["sequel"] = 1                  # Yes
new_movie["runtime_min"] = 150
new_movie["trailer_views_million"] = 40
new_movie["director_previous_hits"] = 4

# Set genre = Action (all genre columns 0, then Action 1)
for col in X.columns:
    if col.startswith("genre_"):
        new_movie[col] = 0
new_movie["genre_Action"] = 1

# Set language = Tamil
for col in X.columns:
    if col.startswith("language_"):
        new_movie[col] = 0
new_movie["language_Tamil"] = 1

# Predict
print("Prediction:", "HIT 🎉" if model.predict(new_movie)[0] == 1 else "FLOP 😢")
print("Hit chance:", round(model.predict_proba(new_movie)[0][1] * 100, 1), "%")



new_movie = X.iloc[[0]].copy()

new_movie["budget_crore"] = 20
new_movie["star_power"] = 0
new_movie["festival_release"] = 0
new_movie["sequel"] = 0
new_movie["runtime_min"] = 180
new_movie["trailer_views_million"] = 2
new_movie["director_previous_hits"] = 0

for col in X.columns:
    if col.startswith("genre_"):
        new_movie[col] = 0
new_movie["genre_Drama"] = 1

for col in X.columns:
    if col.startswith("language_"):
        new_movie[col] = 0
new_movie["language_Tamil"] = 1

print("Prediction:", "HIT 🎉" if model.predict(new_movie)[0] == 1 else "FLOP 😢")
print("Hit chance:", round(model.predict_proba(new_movie)[0][1] * 100, 1), "%")