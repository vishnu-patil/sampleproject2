import pandas as pd
import numpy as np
import pickle
import json

class Titanic():
    def __init__(self,Pclass,Gender,Age,SibSp,Parch,Fare,Embarked):
        self.Pclass = Pclass
        self.Gender = Gender
        self.Age = Age
        self.SibSp = SibSp
        self.Parch = Parch
        self.Fare = Fare
        self.Embarked = Embarked

        # self.mongo_client = pymongo.MongoClient("mongodb://localhost:27017")
        # self.db = self.mongo_client['titanic']

        # self.collection_titanic = self.db['titanic']
        # self.collection_titanic.insert_one({"Pclass":self.Pclass,
        #                                 "Gender":self.Gender,
        #                                 "Age":self.Age,
        #                                 "SibSp":self.SibSp,
        #                                 "Parch":self.Parch,
        #                                 "Fare":self.Fare,
        #                                 "Embarked":self.Embarked,
        #                                 "Prediction": np.nan})

    def load_saved_files(self):
        with open(r"model.pkl",'rb') as f:
           self.model =  pickle.load(f)

        with open(r"project_data",'r') as f:
            self.project_data = json.load(f)

    def predict_model(self):
        self.load_saved_files()

        Embarked = 'Embarked_' + self.Embarked
        index = self.project_data['Columns'].index(Embarked)

        columns = len(self.project_data['Columns'])
        test_array = np.zeros(columns)

        test_array[0] = self.Pclass
        test_array[1] = self.project_data['Gender'][self.Gender]
        test_array[2] = self.Age
        test_array[3] = self.SibSp
        test_array[4] = self.Parch
        test_array[5] = np.sqrt(self.Fare)
        test_array[index] = 1
        print(test_array)

        predict_value = self.model.predict([test_array])[0]
        print("The Prediction is:",predict_value)

        # old_query = {"Prediction":np.nan}
        # new_query = {'$set':{"Prediction":int(predict_value)}}

        # self.collection_titanic.update_one(old_query,new_query)
        return predict_value

if __name__ == "__main__":
    obj = Titanic
    obj
        


