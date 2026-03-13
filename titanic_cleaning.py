import pandas as pd

# ── STEP 1: Load Data ──────────────────────────────
df = pd.read_csv("Week 2/titanic.csv")

# ── STEP 2: Explore ───────────────────────────────
# Print shape, info, isnull().sum(), describe()

print(df.shape)
print(df.info())
print(df.isnull().sum)
print(df.describe())

# ── STEP 3: Clean ─────────────────────────────────
# a) Fill missing Age with mean
# b) Fill missing Embarked with "S"
# c) Drop Cabin column
# d) Drop Name, Ticket, PassengerId columns
# e) Verify — isnull().sum() should be all zeros

df["Age"].fillna(df["Age"].mean(), inplace=True)
df["Embarked"].fillna("S", inplace=True)
df.drop(columns="Cabin", inplace=True)
df.drop(columns=["Name","Ticket","PassengerId"], inplace=True )
print(df.isnull().sum())


# ── STEP 4: Feature Engineering ───────────────────
# a) Create "age_group" column (Child/Adult/Senior)
#    Remember — handle NaN before applying!
# b) Create "family_size" column (SibSp + Parch + 1)
# c) Create "fare_category" column (Low/High)

def check_age(Age):
    if pd.isna(Age):
        return "Unknown"
    elif Age <=17 :
        return "Child"
    elif Age >= 18 and Age<60:
        return "Adult"
    else:
        return "Senior"
df["age_group"] = df["Age"].apply(check_age)

df["family_size"] =  df["SibSp"]+df["Parch"]+1
df["fare_category"] =df["Fare"].apply( lambda x:"Low" if x<50 else "High")


# 1. What was the survival rate by gender?
# 2. What was the average fare by passenger class?
# 3. How many children (age < 18) survived?
# 4. What was the average age by passenger class?
# 5. Which embarkation point had the highest survival rate?

print("1. Survival rate by gender:")
print("->",df.groupby("Sex")["Survived"].mean())
print("2. Average fare by passenger class:")
print("->",df.groupby("Pclass")["Fare"].mean())
print("3. children (age < 18) survived:")
children_survived = df[(df["age_group"] == "Child") & (df["Survived"] == 1)]
print("->", len(children_survived))
print("4. average age by passenger class:")
print("->", df.groupby("Pclass")["Age"].mean())
print("5. highest survival rate by highest embarkation point:")
print("->", df.groupby("Embarked")["Survived"].mean())


# ── STEP 6: Save Clean Dataset ────────────────────
df.to_csv("titanic_clean.csv", index=False)
print("Clean dataset saved!")
print("Final shape:", df.shape)

## 📌 Expected Final Output

# Final shape: (891, 11)
# Columns: Survived, Pclass, Sex, Age, SibSp, 
#          Parch, Fare, Embarked, age_group, 
#          family_size, fare_category

# 1. Survival rate by gender:
#    female    0.74
#    male      0.19

# 2. Average fare by class:
#    1st → high
#    3rd → low
# ...
# Clean dataset saved! ✅
