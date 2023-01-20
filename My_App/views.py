
from django.shortcuts import render,HttpResponse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns',500)
import warnings
warnings.filterwarnings("ignore")

def home(request):
    return render(request,"home.html")

def Predict(request):
    return render(request,"Predict.html")

def result(request):
    df=pd.read_csv("C:\Datasets\Bondora_Preprocessed.csv")
    X=df[["PrincipalBalance","Amount","PreviousRepaymentsBeforeLoan","MonthlyPayment","AppliedAmount"]]
    y=df["LoanStatus"]
    from sklearn.preprocessing import LabelEncoder
    le=LabelEncoder()
    y=le.fit_transform(y)
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
    
    from sklearn.ensemble import RandomForestRegressor
    model=RandomForestRegressor()
    model.fit(X_train,y_train)
    val1=float(request.GET['n1'])
    val2=float(request.GET['n2'])
    val3=float(request.GET['n3'])
    val4=float(request.GET['n4'])
    val5=float(request.GET['n5'])
    model.fit(X_train, y_train)

    from sklearn.preprocessing import StandardScaler
    sc=StandardScaler()
    X_train=sc.fit_transform(X_train)
    X_test=sc.transform(X_test)
    pred=model.predict(sc.transform(([[val1,val2,val3,val4,val5]])))

    result2 =""
    if pred>0.5:
        result1="Person is not Defaulted"

    else:
        result1="Person is Defaulted"

    return render(request,"predict.html",{"result2":result1})

# Create your views here.
