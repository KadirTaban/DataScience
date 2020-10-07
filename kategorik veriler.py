
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

#kodlar
#veriyükleme
veriler = pd.read_csv('eksikveriler.csv')
print(veriler)
#veriönisleme
boy = veriler[['boy']]

print(boy)

boykilo = veriler[['boy','kilo']]

print(boykilo)

class insan :
    boy = 180
    def kosmak(self,b):
        return b+10

ali = insan()
print(ali.boy)
print(ali.kosmak(90))

#eksik veriler
#sci_kit learn

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan , strategy="mean")

Yas = veriler.iloc[:,1:4].values
print(Yas)

imputer=imputer.fit(Yas[:,1:4])

Yas[:,1:4] = imputer.transform(Yas[:,1:4])

print(Yas)
ulke = veriler.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing
le= preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(veriler.iloc[:,0])

print(ulke)

ohe = preprocessing.OneHotEncoder()

ulke = ohe.fit_transform(ulke).toarray()

print(ulke)