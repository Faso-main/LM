import numpy as np

def my_func(x):
    return x**2

# Пример использования векторизации
vectorized_func = np.vectorize(my_func)
data = np.array([1, 2, 3, 4, 5])
result = vectorized_func(data)
print(result)