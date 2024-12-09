import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load Data
s = pd.read_csv('./social_media_usage.csv')

# Data Cleaning
def clean_sm(x):
    return np.where(x == 1, 1, 0)

s['web1h'] = clean_sm(s['web1h'])
ss = s[
    ['web1h', 'income', 'educ2', 'par', 'marital', 'gender', 'age']
].dropna()

ss = ss[(ss['income'] <= 9) & (ss['educ2'] <= 8) & (ss['age'] <= 98)]

ss['parent'] = clean_sm(ss['par'])
ss['married'] = np.where(ss['marital'] == 1, 1, 0)  # Encode marital status as binary
ss['female'] = clean_sm(ss['gender'] == 2)

ss.rename(columns={'web1h': 'sm_li'}, inplace=True)
ss = ss[['sm_li', 'income', 'educ2', 'parent', 'married', 'female', 'age']]

# Prepare Data for Modeling
y = ss["sm_li"]
X = ss[["income", "educ2", "female", "married",]]

X_train, X_test, y_train, y_test = train_test_split(X.values, y, stratify=y, test_size=0.2, random_state=555)

lr = LogisticRegression(class_weight='balanced')
lr.fit(X_train, y_train)

# Streamlit App
st.title("LinkedIn User Prediction")

st.markdown("### Enter Your Information Below")

# Dictionaries for dropdown labels
income_labels = {
    1: "Less than $10,000",
    2: "$10,000 to under $20,000",
    3: "$20,000 to under $30,000",
    4: "$30,000 to under $40,000",
    5: "$40,000 to under $50,000",
    6: "$50,000 to under $75,000",
    7: "$75,000 to under $100,000",
    8: "$100,000 to under $150,000",
    9: "$150,000 or more",
}

educ2_labels = {
    1: "Less than high school (Grades 1-8 or no formal schooling)",
    2: "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)",
    3: "High school graduate (Grade 12 with diploma or GED certificate)",
    4: "Some college, no degree (includes some community college)",
    5: "Two-year associate degree from a college or university",
    6: "Four-year college or university degree/Bachelor’s degree",
    7: "Some postgraduate or professional schooling, no postgraduate degree",
    8: "Postgraduate or professional degree (e.g., MA, MS, PhD, MD, JD)",
}



# User Input
income = st.selectbox("Income Level (household)", options=list(income_labels.keys()), format_func=lambda x: income_labels[x])
education = st.selectbox("Highest level of school/degree completed", options=list(educ2_labels.keys()), format_func=lambda x: educ2_labels[x])
gender = st.radio("Gender", ("Male", "Female"))
married_status = st.radio("Are you married?", ("Yes", "No"))



# Convert Input to Features
female = 1 if gender == "Female" else 0
married = 1 if married_status == "Yes" else 0

user_data = pd.DataFrame({
    "income": [income],
    "educ2": [education],
    "female": [female],
    "married": [married],
})

# Predict LinkedIn User
if st.button("Predict"):
    prediction = lr.predict(user_data)[0]
    probability = lr.predict_proba(user_data)[0][1]  # Probability of being a LinkedIn user

    st.markdown(f"### Prediction Probability: {probability:.2f}")
    if prediction == 1:
        st.success("You are likely a LinkedIn user!")
    else:
        st.warning("You are not likely a LinkedIn user.")
