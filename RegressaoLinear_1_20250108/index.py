from RegressaoLinear import LinearRegression
import numpy as np

x = np.array([1,2,3,4,5])
y = np.array([2,4,6,8,10])

lr = LinearRegression(x,y)
previsao = lr.previsao(6)
print(previsao)