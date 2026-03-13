# 🚢 Titanic Data Cleaning Pipeline

Complete data cleaning and analysis pipeline on the Titanic dataset.
Built as part of my AI/ML learning journey.

## 🛠️ Tech Stack
- Python
- Pandas
- Numpy

## 📊 What This Project Does
- Loads raw Titanic dataset (891 rows, 12 columns)
- Explores data — shape, info, missing values
- Cleans data — fills nulls, drops irrelevant columns
- Engineers 3 new features:
  - age_group (Child / Adult / Senior)
  - family_size (SibSp + Parch + 1)
  - fare_category (Low / High)
- Answers 5 business questions using groupby
- Saves clean dataset as CSV

## 📈 Key Findings
- Female survival rate: 74% vs Male: 19%
- 1st class passengers paid highest average fare
- Passengers from Cherbourg had highest survival rate

## 🚀 How to Run
pip install pandas
python titanic_cleaning.py

## 👤 Author
Shanti | Aspiring AI/ML Developer | Open to Work
