from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

main = tkinter.Tk()
main.title("Detection of Stroke Disease using Machine Learning Algorithms")
main.geometry("1000x650")

global filename, le1,le2,le3,le4,le5, dataset, rf
global X, Y
global X_train, X_test, y_train, y_test
accuracy = []
precision = []
recall = []
fscore = []

def loadDataset():
    global filename, dataset
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.insert(END,str(filename)+" loaded\n\n")
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset.head()))

def preprocessDataset():
    text.delete('1.0', END)
    global X, Y
    global X_train, X_test, y_train, y_test
    global dataset, le1,le2,le3,le4,le5
    le1 = LabelEncoder()
    le2 = LabelEncoder()
    le3 = LabelEncoder()
    le4 = LabelEncoder()
    le5 = LabelEncoder()
    dataset.fillna(0, inplace = True)    
    dataset['gender'] = pd.Series(le1.fit_transform(dataset['gender'].astype(str)))
    dataset['ever_married'] = pd.Series(le2.fit_transform(dataset['ever_married'].astype(str)))
    dataset['work_type'] = pd.Series(le3.fit_transform(dataset['work_type'].astype(str)))
    dataset['Residence_type'] = pd.Series(le4.fit_transform(dataset['Residence_type'].astype(str)))
    dataset['smoking_status'] = pd.Series(le5.fit_transform(dataset['smoking_status'].astype(str)))
    text.insert(END,str(dataset.head())+"\n\n")
    text.update_idletasks()
    label = dataset.groupby('stroke').size()
    dataset = dataset.values
    text.insert(END,"\nTotal attributes before applying features selection: "+str(dataset.shape[1])+"\n\n")
    X = dataset[:,1:dataset.shape[1]-1]
    Y = dataset[:,dataset.shape[1]-1]
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    text.insert(END,"\nTotal attributes after applying features selection: "+str(X.shape[1])+"\n\n")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Total records found in dataset : "+str(X.shape[0])+"\n\n")
    text.insert(END,"Dataset split for train and test. 80% for training and 20% for testing\n\n")
    text.insert(END,"Total records used to train Machine Learning Algorithms : "+str(X_train.shape[0])+"\n")
    text.insert(END,"Total records used to test Machine Learning Algorithms : "+str(X_test.shape[0])+"\n")    
    label.plot(kind="bar")
    plt.title("Number of Normal & Stroke Disease Instances in dataset")
    plt.show()

def calculateMetrics(predict, testY, algorithm):
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100    
    text.insert(END,algorithm+' Accuracy  : '+str(a)+"\n")
    text.insert(END,algorithm+' Precision : '+str(p)+"\n")
    text.insert(END,algorithm+' Recall    : '+str(r)+"\n")
    text.insert(END,algorithm+' FScore    : '+str(f)+"\n\n")
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.update_idletasks()
    LABELS = ['Normal','Stroke'] 
    conf_matrix = confusion_matrix(testY, predict) 
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,2])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()    

def trainNaiveBayes():
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    cls = GaussianNB()
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test)
    calculateMetrics(predict, y_test, "Naive Bayes")
    
def trainDT():
    global X_train, X_test, y_train, y_test
    cls = DecisionTreeClassifier()
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test)
    calculateMetrics(predict, y_test, "J48 Algorithm")

def trainKNN():
    global X_train, X_test, y_train, y_test
    cls = KNeighborsClassifier(n_neighbors = 2) 
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test)
    calculateMetrics(predict, y_test, "KNN")    

def trainRanfomForest():
    global X_train, X_test, y_train, y_test, rf
    cls = RandomForestClassifier() 
    cls.fit(X_train, y_train)
    rf = cls
    predict = cls.predict(X_test)
    calculateMetrics(predict, y_test, "Random Forest")

def graph():
    df = pd.DataFrame([['Naive Bayes','Precision',precision[0]],['Naive Bayes','Recall',recall[0]],['Naive Bayes','F1 Score',fscore[0]],['Naive Bayes','Accuracy',accuracy[0]],
                       ['J48','Precision',precision[1]],['J48','Recall',recall[1]],['J48','F1 Score',fscore[1]],['J48','Accuracy',accuracy[1]],
                       ['KNN','Precision',precision[2]],['KNN','Recall',recall[2]],['KNN','F1 Score',fscore[2]],['KNN','Accuracy',accuracy[2]],
                       ['Random Forest','Precision',precision[3]],['Random Forest','Recall',recall[3]],['Random Forest','F1 Score',fscore[3]],['Random Forest','Accuracy',accuracy[3]],
                       ['ANN','Precision',precision[4]],['ANN','Recall',recall[4]],['ANN','F1 Score',fscore[4]],['ANN','Accuracy',accuracy[4]],
                       
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot(index="Parameters", columns="Algorithms", values="Value").plot(kind='bar')
    plt.show()


def trainANN():
    global X, Y
    Y1 = to_categorical(Y)
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X, Y1, test_size=0.2)
    ann_model = Sequential()
    ann_model.add(Dense(512, input_shape=(X_train1.shape[1],)))
    ann_model.add(Activation('relu'))
    ann_model.add(Dropout(0.3))
    ann_model.add(Dense(512))
    ann_model.add(Activation('relu'))
    ann_model.add(Dropout(0.3))
    ann_model.add(Dense(2))
    ann_model.add(Activation('softmax'))
    ann_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(ann_model.summary())
    acc_history = ann_model.fit(X, Y1, epochs=200, validation_data=(X_test1, y_test1))
    print(ann_model.summary())
    predict = ann_model.predict(X_test1)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(y_test1, axis=1)
    calculateMetrics(predict, testY, "ANN") 


def predict():
    text.delete('1.0', END)
    global rf, le1,le2,le3,le4,le5
    testfile = filedialog.askopenfilename(initialdir="Dataset")
    dataset = pd.read_csv(testfile)
    dataset.fillna(0, inplace = True)    
    dataset['gender'] = pd.Series(le1.transform(dataset['gender'].astype(str)))
    dataset['ever_married'] = pd.Series(le2.transform(dataset['ever_married'].astype(str)))
    dataset['work_type'] = pd.Series(le3.transform(dataset['work_type'].astype(str)))
    dataset['Residence_type'] = pd.Series(le4.transform(dataset['Residence_type'].astype(str)))
    dataset['smoking_status'] = pd.Series(le5.transform(dataset['smoking_status'].astype(str)))
    dataset = dataset.values

    dataset = dataset[:,1:dataset.shape[1]]
    predict = rf.predict(dataset)
    print(predict)
    for i in range(len(predict)):
        if predict[i] == 0:
            text.insert(END,"Test Data = "+str(dataset[i])+" PREDICTED AS ====> NO STROKE\n\n")
        if predict[i] == 1:
            text.insert(END,"Test Data = "+str(dataset[i])+" PREDICTED AS ====> STROKE\n\n")    
    

font = ('times', 15, 'bold')
title = Label(main, text='Detection of Stroke Disease using Machine Learning Algorithms', justify=LEFT)
title.config(bg='lavender blush', fg='DarkOrchid1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 12, 'bold')
loadButton = Button(main, text="Upload Stroke Dataset", command=loadDataset)
loadButton.place(x=10,y=100)
loadButton.config(font=font1)  

preprocessButton = Button(main, text="Dataset Preprocessing & Features Selection", command=preprocessDataset)
preprocessButton.place(x=300,y=100)
preprocessButton.config(font=font1)

nbButton = Button(main, text="Train Naive Bayes Algorithm", command=trainNaiveBayes)
nbButton.place(x=730,y=100)
nbButton.config(font=font1)

dtButton = Button(main, text="Train J48 Algorithm", command=trainDT)
dtButton.place(x=10,y=150)
dtButton.config(font=font1)

knnButton = Button(main, text="Train KNN Algorithm", command=trainKNN)
knnButton.place(x=300,y=150)
knnButton.config(font=font1)

rfButton = Button(main, text="Train Random Forest Algorithm", command=trainRanfomForest)
rfButton.place(x=730,y=150)
rfButton.config(font=font1)

annButton = Button(main, text="Train ANN Algorithm", command=trainANN)
annButton.place(x=10,y=200)
annButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=300,y=200)
graphButton.config(font=font1)

predictButton = Button(main, text="Predict Disease on Test Data", command=predict)
predictButton.place(x=730,y=200)
predictButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=160)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1) 

main.config(bg='light coral')
main.mainloop()
