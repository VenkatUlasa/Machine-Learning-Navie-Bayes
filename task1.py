import numpy as np
import pandas as pd
import sklearn
import sys
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

class NBT:
    def __init__(self,path):
        try:
            self.df = pd.read_csv(path)
            self.df = self.df.drop("id", axis=1)
            self.df["diagnosis"] = self.df["diagnosis"].map({"M": 0, "B": 1})
            self.X = self.df.iloc[:,1:]
            self.y = self.df.iloc[:,0]

            self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
            self.nb_algo = GaussianNB()
            self.nb_algo.fit(self.X_train,self.y_train)

            self.y_train_pred = self.nb_algo.predict(self.X_train)

        except Exception as e :
            er_type, er_msg,line_no = sys.exc_info()
            print(f'<{er_type}> ..... <{er_msg}> ..... <{line_no.tb_lineno}>')

    def training(self):
        try:
            c_m = confusion_matrix(self.y_train,self.y_train_pred)
            acc = accuracy_score(self.y_train,self.y_train_pred)
            print(acc)
            print(classification_report(self.y_train,self.y_train_pred))
        except Exception as e :
            er_type, er_msg,line_no = sys.exc_info()
            print(f'<{er_type}> ..... <{er_msg}> ..... <{line_no.tb_lineno}>')

    def testing(self):
        try:
            y_test_pred = self.nb_algo.predict(self.X_test)
            tn,fp,fn,tp = confusion_matrix(self.y_test,y_test_pred).ravel()

            acc = (tp+tn)/(tp+tn+fp+fn)
            print(f'Test Accuracy : {acc}')

            pre = (tp/(tp+fp))
            re = (tp/(tp+fn))
            f1_score = 2 *((pre*re)/(pre+re))

            print(f'Test Precision : {pre}')
            print(f'Test Recall : {re}')
            print(f'Test F1-Score : {f1_score}')

        except Exception as e :
            er_type, er_msg,line_no = sys.exc_info()
            print(f'<{er_type}> ..... <{er_msg}> ..... <{line_no.tb_lineno}>')

if __name__ == "__main__"  :
    c1 = NBT("breast-cancer.csv")
    # c1.training()
    c1.testing()
