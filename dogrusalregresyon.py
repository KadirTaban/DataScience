# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:50:13 2020

@author: kadirtaban
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('satislar.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)
#veri on isleme

aylar = veriler[['Aylar']]
print(aylar)

satislar = veriler[['Satislar']]
print(satislar)


#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33, random_state=0)



from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x_train,y_train)

tahmin = lr.predict(x_test)
x_train = x_train.sort_index()
y_train = y_train.sort_index()
plt.plot(x_train,y_train)
