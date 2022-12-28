import numpy as np
import pandas as pd
from tqdm import tqdm


def to_matrix(a, b, c, p, q, f, k):
    matrix = (np.diag(a, -1) + np.diag(b) + np.diag(c, 1))
    matrix = np.hstack((matrix, f.reshape(-1, 1)))
    matrix[k, :-1] = p
    matrix[:, k] = q
    return matrix


def step_one(a, b, c, p, q, f, k):
    for i in range(k - 1):
        if b[i] == 0:
            return a, b, c, p, q, f
        c[i] /= b[i]
        q[i] /= b[i]
        f[i] /= b[i]
        b[i] = 1
        print('Делим ', str(i) + '-ю строку на b_' + str(i) + ':\n', pd.DataFrame(to_matrix(a, b, c, p, q, f, k)))

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

        print('Вычитаем из ', str(i + 1) + '-й строки', str(i) + '-ю умноженную на a_' + str(i + 1) + ';\n',
              'Вычитаем из ', str(k) + '-й строки', str(i) + '-ю умноженную на p_' + str(i) + ':\n',
              pd.DataFrame(to_matrix(a, b, c, p, q, f, k)))

    # q_(k-1) ≡ b_(k-1)
    if b[k - 1] == 0:
        return a, b, c, p, q, f
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
        if b[i] == 0:
            return a, b, c, p, q, f
        a[i - 1] /= b[i]
        q[i] /= b[i]
        f[i] /= b[i]
        b[i] = 1
        print('Делим ', str(i) + '-ю строку на b_' + str(i) + ':\n', pd.DataFrame(to_matrix(a, b, c, p, q, f, k)))

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

        print('Вычитаем из ', str(i + 1) + '-й строки', str(i) + '-ю умноженную на c_' + str(i - 1) + ';\n',
              'Вычитаем из ', str(k) + '-й строки', str(i) + '-ю умноженную на p_' + str(i) + ':\n',
              pd.DataFrame(to_matrix(a, b, c, p, q, f, k)))

    # q_(k+1) ≡ a_k
    if b[k + 1] == 0:
        return a, b, c, p, q, f
    q[k + 1] /= b[k + 1]
    a[k] = q[k + 1]
    f[k + 1] /= b[k + 1]
    b[k + 1] = 1

    q[k] -= q[k + 1] * p[k + 1]
    f[k] -= f[k + 1] * p[k + 1]

    if q[k] == 0:
        return a, b, c, p, q, f
    f[k] /= q[k]
    # q_k ≡ p_k ≡ b_k
    p[k] = 1
    q[k] = 1
    b[k] = 1

    # p_(k+1) ≡ c_k
    p[k + 1] = 0
    c[k] = 0

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


def generate_random_vals(n, k, low=-25, high=25):
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
        a[k - 1] = p[k - 1]
        p[k] = q[k]
        b[k] = p[k]
        c[k - 1] = q[k - 1]
        a[k] = q[k + 1]
        c[k] = p[k + 1]
        x = np.random.uniform(low, high, size=(n,))
        lambdas, v = np.linalg.eig(matrix.T)
        f = matrix @ x

    return a, b, c, p, q, f, x


def generate_random_vals_for_unit_x(n, k, low=-25, high=25):
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
        a[k - 1] = p[k - 1]
        p[k] = q[k]
        b[k] = p[k]
        c[k - 1] = q[k - 1]
        a[k] = q[k + 1]
        c[k] = p[k + 1]
        x = np.ones(n)
        lambdas, v = np.linalg.eig(matrix.T)
        f = matrix @ x

    return a, b, c, p, q, f, x


def solve(a, b, c, p, q, f, k):
    steps = [step_one, step_two, step_three, step_four]
    for i in range(len(steps)):
        a, b, c, p, q, f = steps[i](a, b, c, p, q, f, k)
    return f


def get_error():
    err_dict = {
        'Размерность системы': [],
        'Диапазон значений элементов матрицы': [],
        'Средняя относительная погрешность системы': [],
        'Среднее значение оценки точности': []
    }

    for n_ in tqdm(np.array([10, 100, 1000]), position=0, leave=True):
        n_power = np.round(np.log10(n_)).astype(int)
        s = '10^' + str(n_power)
        for rng in 10, 100, 1000:
            err_dict['Размерность системы'].append('10^' + str(n_power))
            rng_power = np.round(np.log10(rng)).astype(int)
            s1 = '-10^' + str(rng_power) + '÷' + '10^' + str(rng_power)
            # print('Диапазон значений:', s1)
            err_dict['Диапазон значений элементов матрицы'].append(s1)
            rel_err = []
            accuracy = []
            for _ in range(10):
                # n = np.random.randint(n_, n_ * 10)
                n = n_ + 1
                k = np.random.randint(2, n - 1)
                a, b, c, p, q, f, x = generate_random_vals(n, k, low=-rng, high=rng)
                x_ = solve(a, b, c, p, q, f, k)
                eps = []
                for i in range(len(x)):
                    if np.abs(x_[i]) > n_ / 2:
                        eps.append(np.abs(x_[i] - x[i]) / np.abs(x[i]))
                    else:
                        eps.append(np.abs(x_[i] - x[i]))
                rel_err.append(np.max(eps))
                a, b, c, p, q, f, x = generate_random_vals_for_unit_x(n, k, low=-rng, high=rng)
                x_ = solve(a, b, c, p, q, f, k)
                eps = np.abs(x_ - x)
                accuracy.append(np.max(eps))

            err_dict['Средняя относительная погрешность системы'].append(np.round(np.mean(rel_err), 3))
            err_dict['Среднее значение оценки точности'].append(np.round(np.mean(accuracy), 3))

    df = pd.DataFrame(err_dict)
    try_flag = True
    while try_flag:
        try:
            df.to_excel('Вычисления.xlsx')
        except PermissionError:
            print('PermissionError! Файл Вычисления.xlsx уже используется другим приложением')
            continue
        else:
            print('Запись в файл завершена')
            try_flag = False


if __name__ == '__main__':
    # get_error()
    # n = 100
    # k = np.random.randint(1, n)
    # a, b, c, p, q, f, x = generate_random_vals(n, k)

    a = np.arange(2, 17, 2, dtype=float)
    b = np.arange(1, 18, 2, dtype=float)
    c = a.copy()
    f = np.array([9, 16, 23, 30, 27, 90, 39, 58, 47], dtype=float)
    p = np.arange(6, 15, dtype=float)
    q = np.arange(6, 15, dtype=float)
    k = 5
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
    print(pd.DataFrame(to_matrix(a, b, c, p, q, f, k)), '\n')
    # print('Исходное решение:\n', x)
    print('Решение с помощью моего метода:\n', f)
    # print('Абсолютная погрешность:',np.max(np.abs(x - f)))
