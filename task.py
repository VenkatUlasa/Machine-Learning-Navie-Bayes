import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import warnings
warnings.filterwarnings("ignore")
import pickle

import sys

class NBT:
    def __init__(self,path1,path2):
        try:
            self.X = pd.read_csv(path1)
            self.y = pd.read_csv(path2)
            self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)

        except Exception as e :
            er_type, er_msg,line_no = sys.exc_info()
            print(f'<{er_type}> ..... <{er_msg}> ..... <{line_no.tb_lineno}>')

    def training(self):
        try:
            print(f"Train Accuracy : {accuracy_score(self.y_train,self.y_train_pred)}")
        except Exception as e :
            er_type, er_msg, line_no = sys.exc_info()
            print(f'<{er_type}> ..... <{er_msg}> ..... <{line_no.tb_lineno}>')
    def testing(self):
        try:
            print(f"Test Accuracy : {accuracy_score(self.y_test,self.y_test_pred)}")
        except Exception as e :
            er_type, er_msg, line_no = sys.exc_info()
            print(f'<{er_type}> ..... <{er_msg}> ..... <{line_no.tb_lineno}>')

    def navie_bayes(self):
        try:
            self.algo = GaussianNB()
            self.algo.fit(self.X_train, self.y_train)
            self.y_train_pred = self.algo.predict(self.X_train)
            self.y_test_pred = self.algo.predict(self.X_test)
            return self.algo
        except Exception as e:
            er_type, er_msg, line_no = sys.exc_info()
            print(f'<{er_type}> ..... <{er_msg}> ..... <{line_no.tb_lineno}>')

if __name__ == "__main__" :
    try:
        nb = NBT("Aids_classification_dataset.csv","aids_target_data.csv")
        with open('aids_clasi.pkl', 'wb') as f:
            pickle.dump(nb.navie_bayes(), f)
    except Exception as e :
        er_type, er_msg, line_no = sys.exc_info()
        print(f'<{er_type}> ..... <{er_msg}> ..... <{line_no.tb_lineno}>')

