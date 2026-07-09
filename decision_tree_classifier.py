
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

DATA_PATH="data/bank.csv"
OUT_DIR="outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def load_data(path):
    return pd.read_csv(path, sep=';')

def preprocess(df):
    df=df.copy()
    encoders={}
    for c in df.select_dtypes(include="object").columns:
        le=LabelEncoder()
        df[c]=le.fit_transform(df[c].astype(str))
        encoders[c]=le
    return df

def main():
    df=load_data(DATA_PATH)
    print(df.head())
    df=preprocess(df)

    X=df.drop("y", axis=1)
    y=df["y"]

    X_train,X_test,y_train,y_test=train_test_split(
        X,y,test_size=0.2,random_state=42,stratify=y
    )

    model=DecisionTreeClassifier(max_depth=5,criterion="entropy",random_state=42)
    model.fit(X_train,y_train)

    pred=model.predict(X_test)

    acc=accuracy_score(y_test,pred)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test,pred))

    fig=plt.figure(figsize=(14,8))
    plot_tree(model,feature_names=X.columns,class_names=["No","Yes"],filled=True,max_depth=3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,"decision_tree.png"))
    plt.close(fig)

    cm=confusion_matrix(y_test,pred)
    disp=ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.savefig(os.path.join(OUT_DIR,"confusion_matrix.png"))
    plt.close()

    importance=pd.Series(model.feature_importances_,index=X.columns).sort_values()
    importance.plot(kind="barh",figsize=(8,6))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,"feature_importance.png"))
    plt.close()

    with open(os.path.join(OUT_DIR,"model_metrics.txt"),"w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(classification_report(y_test,pred))

if __name__=="__main__":
    main()
