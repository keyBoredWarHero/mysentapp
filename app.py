import pandas as pd
import streamlit as st


st.markdown("# welcome to my app")



s = pd.read_csv('downloads/social_media_usage.csv')


print(s.head)


s.isnull().sum()

import numpy as np

def clean_sm(x):
    return np.where(x == 1, 1, 0)
    toy = { 
    'd': [4,1,0],
    'e': [1,1,3]
}
toys = pd.DataFrame(toy)

toys

toys.applymap(clean_sm)

ss = s[[
    'web1h',  
    'income', 
    'educ2', 
    'par',  
    'marital',  
    'gender',  
    'age' ]]

ss.rename(columns={'web1h': 'sm_li'}, inplace=True) 

ss['sm_li'] = clean_sm(ss['sm_li'])


ss = ss[
    (ss['income'] <= 9) & 
    (ss['educ2'] <= 8) & 
    (ss['age'] <= 98)
].dropna()

ss['parent'] = clean_sm(ss['par'])
ss['married'] = clean_sm(ss['marital'])
ss['female'] = clean_sm(ss['gender'] == 2)

ss = ss[['sm_li', 'income', 'educ2', 'parent', 'married', 'female', 'age']]


ss


ss['sm_li'].sum()

import altair as alt

alt.Chart(ss.groupby(["educ2", "female"], as_index=False)["sm_li"].mean()).\
mark_circle().\
encode(x="educ2",
      y="sm_li",
      color="female:N")


y = ss["sm_li"]
X = ss[["age", "income", "female", "educ2", "parent", "married"]]

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


X_train, X_test, y_train, y_test = train_test_split(X.values,
                                                    y,
                                                    stratify=y,       
                                                    test_size=0.2,    
                                                    random_state=555)


lr = LogisticRegression()

lr.fit(X_train, y_train)


y_pred = lr.predict(X_test)


pd.DataFrame(confusion_matrix(y_test, y_pred),
            columns=["Predicted negative", "Predicted positive"],
            index=["Actual negative","Actual positive"]).style.background_gradient(cmap="PiYG")

34/84

34/61


print(classification_report(y_test, y_pred))



newdata = pd.DataFrame({
    "age": [42, 82],
    "educ2": [7, 7],
    "income": [8, 8],
    "par": [0, 0],
    "married": [1,1],
    "female": [0,0]
})

newdata


newdata["prediction_sm_li"] = lr.predict(newdata)


newdata