import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from yellowbrick.classifier import ConfusionMatrix

# Load the dataset
base = pd.read_csv('insurance.csv')
base = base.drop(columns=['Unnamed: 0'])

base.shape

y = base.iloc[:,7].values
x = base.iloc[:,[0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]].values

labelencoder = LabelEncoder()

for i in range(x.shape[1]):
  if x[:,i].dtype == 'object':
    x[:,i] = labelencoder.fit_transform(x[:,i])
    
# x independente
# y a variavel dependente
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x,y,test_size=0.3, random_state=1)

modelo = GaussianNB()
modelo.fit(x_treinamento, y_treinamento)

previsoes = modelo.predict(x_teste)

acurrancy = accuracy_score(y_teste, previsoes)
precision = precision_score(y_teste, previsoes, average=None)
recall = recall_score(y_teste, previsoes, average='weighted')
f1 = f1_score(y_teste, previsoes, average='weighted')
print(f'Acuracia: {acurrancy}, Precisao: {precision}, Recall: {recall}, f1: {f1}')

report = classification_report(y_teste, previsoes)
print(report)

confusao = ConfusionMatrix(modelo, classes=['None','Severe','Mild','Moderate'])
confusao.fit(x_treinamento, y_treinamento)
confusao.score(x_teste, y_teste)
confusao.poof
