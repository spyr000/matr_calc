import numpy as np
import pandas as pd
import random as rnd


# df = pd.DataFrame(columns=['left', 'right'])
# df['left'] = np.zeros((10, 10))
# df['right'] = np.ones((10, 1))
# print(df)


# n = 9
# a = np.arange(n)
# print(a)
# print(a[4:0:-1])

def to_matrix(a, b, c, p, q, f, k):
    matrix = (np.diag(a, -1) \
             + np.diag(b) \
             + np.diag(c, 1))
    matrix = np.hstack((matrix, f.reshape(-1, 1)))
    matrix[k, :-1] = p
    matrix[:, k] = q
    return matrix


def step_one(a, b, c, p, q, f, k):
    for i in range(k - 1):
        c[i] /= b[i]
        q[i] /= b[i]
        f[i] /= b[i]
        b[i] = 1
        # print('Делим ', str(i) + '-ю строку на b_' + str(i) + ':\n', pd.DataFrame(to_matrix(a, b, c, p, q, f, k)))

        b[i + 1] -= c[i] * a[i]
        q[i + 1] -= q[i] * a[i]
        f[i + 1] -= f[i] * a[i]
        a[i] = 0

        p[i + 1] -= c[i] * p[i]
        f[k] -= f[i] * p[i]
        q[k] -= q[i] * p[i]
        b[k] -= q[i] * p[i]
        p[k] -= q[i] * p[i]
        p[i] = 0

        # print('Вычитаем из ', str(i + 1) + '-й строки', str(i) + '-ю умноженную на a_' + str(i + 1) + ';\n',
        #       'Вычитаем из ', str(k) + '-й строки', str(i) + '-ю умноженную на p_' + str(i) + ':\n',
        #       pd.DataFrame(to_matrix(a, b, c, p, q, f, k)))


    # q_(k-1) ≡ b_(k-1)
    q[k - 1] /= b[k - 1]
    c[k - 1] = q[k - 1]
    f[k - 1] /= b[k - 1]
    b[k - 1] = 1
    # q_k ≡ p_k ≡ b_k
    q[k] -= q[k - 1] * p[k - 1]
    p[k] = q[k]
    b[k] = q[k]
    f[k] -= f[k - 1] * p[k - 1]
    # p_(k-1) ≡ a_(k-1)
    p[k - 1] = 0
    a[k - 1] = 0
    return a, b, c, p, q, f


def step_two(a, b, c, p, q, f, k):
    n = len(b)
    for i in range(n - 1, k + 1, -1):
        a[i - 1] /= b[i]
        q[i] /= b[i]
        f[i] /= b[i]
        b[i] = 1
        # print('Делим ', str(i) + '-ю строку на b_' + str(i) + ':\n', pd.DataFrame(to_matrix(a, b, c, p, q, f, k)))

        b[i - 1] -= a[i - 1] * c[i - 1]
        q[i - 1] -= q[i] * c[i - 1]
        f[i - 1] -= f[i] * c[i - 1]
        c[i - 1] = 0

        p[i - 1] -= a[i - 1] * p[i]
        f[k] -= f[i] * p[i]
        q[k] -= q[i] * p[i]
        b[k] -= q[i] * p[i]
        p[k] -= q[i] * p[i]
        p[i] = 0

        # print('Вычитаем из ', str(i + 1) + '-й строки', str(i) + '-ю умноженную на c_' + str(i - 1) + ';\n',
        #       'Вычитаем из ', str(k) + '-й строки', str(i) + '-ю умноженную на p_' + str(i) + ':\n',
        #       pd.DataFrame(to_matrix(a, b, c, p, q, f, k)))

    # q_(k-1) ≡ b_(k-1)
    q[k + 1] /= b[k + 1]
    a[k] = q[k + 1]
    f[k + 1] /= b[k + 1]
    b[k + 1] = 1

    q[k] -= q[k + 1] * p[k + 1]
    f[k] -= f[k + 1] * p[k + 1]
    f[k] /= q[k]
    # q_k ≡ p_k ≡ b_k
    p[k] = 1
    q[k] = 1
    b[k] = 1

    # p_(k-1) ≡ a_(k-1)
    p[k + 1] = 0
    c[k + 1] = 0

    return a, b, c, p, q, f


def step_three(a, b, c, p, q, f, k):
    f[:k] -= q[:k] * f[k]
    q[:k] = 0
    c[k - 1] = 0

    f[k + 1:] -= q[k + 1:] * f[k]
    q[k + 1:] = 0
    a[k] = 0

    return a, b, c, p, q, f


def step_four(a, b, c, p, q, f, k):
    for i in range(k - 1, -1, -1):
        f[i] -= f[i + 1] * c[i]
        c[i] = 0

    n = len(b)
    for i in range(k + 2, n):
        f[i] -= f[i - 1] * a[i - 1]
        a[i - 1] = 0
    return a, b, c, p, q, f


def generate_random_vals(n, k, low=-25,high=25):
    lambdas = np.array([0])
    a, b, c, p, q, f, x = None, None, None, None, None, None, None
    while np.any(lambdas == 0):
        a = np.random.uniform(low, high, size=(n - 1,))
        b = np.random.uniform(low, high, size=(n,))
        c = np.random.uniform(low, high, size=(n - 1,))
        p = np.random.uniform(low, high, size=(n,))
        q = np.random.uniform(low, high, size=(n,))
        matrix = np.diag(a, -1) + np.diag(b) + np.diag(c, 1)
        matrix[k, :] = p
        matrix[:, k] = q
        x = np.random.uniform(low, high, size=(n,))
        lambdas, v = np.linalg.eig(matrix.T)
        f = matrix @ x

    return a, b, c, p, q, f, x

def get_error():
    a = np.arange(2, 17, 2, dtype=float)
    b = np.arange(1, 18, 2, dtype=float)
    c = a.copy()
    f = np.array([9, 16, 23, 30, 27, 90, 39, 58, 47], dtype=float)
    p = np.arange(6, 15, dtype=float)
    q = np.arange(6, 15, dtype=float)
    a, b, c, p, q, f = step_one(a, b, c, p, q, f, k)
    a, b, c, p, q, f = step_two(a, b, c, p, q, f, k)
    a, b, c, p, q, f = step_three(a, b, c, p, q, f, k)
    a, b, c, p, q, f = step_four(a, b, c, p, q, f, k)




if __name__ == '__main__':
    # a = np.arange(2, 17, 2, dtype=float)
    # b = np.arange(1, 18, 2, dtype=float)
    # c = a.copy()
    # f = np.array([9, 16, 23, 30, 27, 90, 39, 58, 47], dtype=float)
    # p = np.arange(6, 15, dtype=float)
    # q = np.arange(6, 15, dtype=float)
    n, k = 10, 0
    a, b, c, p, q, f, x = generate_random_vals(n, k)

    print('\n\t ИСХОДНАЯ МАТРИЦА: \n')
    matrix = to_matrix(a, b, c, p, q, f, k)
    print(pd.DataFrame(matrix))
    print('\n\t ШАГ 1: \n')
    a, b, c, p, q, f = step_one(a, b, c, p, q, f, k)
    print(pd.DataFrame(to_matrix(a, b, c, p, q, f, k)))
    print('\n\t ШАГ 2: \n')
    a, b, c, p, q, f = step_two(a, b, c, p, q, f, k)
    print(pd.DataFrame(to_matrix(a, b, c, p, q, f, k)))
    print('\n\t ШАГ 3: \n')
    a, b, c, p, q, f = step_three(a, b, c, p, q, f, k)
    print(pd.DataFrame(to_matrix(a, b, c, p, q, f, k)))
    a, b, c, p, q, f = step_four(a, b, c, p, q, f, k)
    print('\n\t МАТРИЦА ПОСЛЕ ПРЕОБРАЗОВАНИЙ: \n')
    print(pd.DataFrame(to_matrix(a, b, c, p, q, f, k)),'\n')
    print('Исходное решение:', x)
    print('Решение с помощью моего метода:', f)
    # df.to_excel('file.xlsx')
