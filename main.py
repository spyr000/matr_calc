import numpy as np


class Matrix():
    def __init__(self, a, b, c, p, q, f, k=8):
        self.a = a
        self.b = b
        self.c = c
        self.p = p
        self.q = q
        self.f = f
        self.k_row = k
        self.k_col = k
        self.matrix = np.diag(a) + np.diag(b, -1) + np.diag(c, 1)
        self.matrix[k, :] = p
        self.matrix[:, k] = q

    def __setitem__(self, key, value: np.ndarray):
        if key == 'a':
            np.fill_diagonal(self.matrix, value)
        elif key == 'b':
            np.fill_diagonal(self.matrix[1:, :-1], value)
        elif key == 'c':
            np.fill_diagonal(self.matrix[:-1, 1:], value)
        elif key == 'p':
            self.matrix[self.k_row] = value
        elif key == 'q':
            self.matrix[:,self.k_col] = value
        elif key == 'f':
            self.f = value
        elif key[0] == 'a':
            np.fill_diagonal(self.matrix, value)
        elif key[0] == 'b':
            np.fill_diagonal(self.matrix[1:, :-1], value)
        elif key[0] == 'c':
            np.fill_diagonal(self.matrix[:-1, 1:], value)
        elif key[0] == 'p':
            self.matrix[self.k_row] = value
        elif key[0] == 'q':
            self.matrix[:,self.k_col] = value
        elif key[0] == 'f':
            self.f = value
        else:
            raise Exception()

    def __getitem__(self, item):
        if item == 'a':
            return  self.matrix.diagonal()
        elif item == 'b':
            return  self.matrix.diagonal(-1)
        elif item == 'c':
            return  self.matrix.diagonal(1)
        elif item == 'p':
            return self.matrix[self.k_row]
        elif item == 'q':
            return self.matrix[:,self.k_col]
        elif item == 'f':
            return self.f
        else:
            raise Exception()


def make_first_step(matr):
    k  = matr.k_row
    for i in range(k):
        R = 1 / matr['b'][i]
        matr['b'][i] = 1
        matr['c'][i] = R * matr['c'][i]
        matr['f'] = R * matr['f']
        R = matr['a'][i+1]
        matr['a'][i+1] = 0
        matr['b'][i+1] = matr['b'][i+1] - R * matr['c'][i]
        matr['f'][i+1] = matr['f'][i+1] - R * matr['f'][i]
        R = matr['p'][i]
        matr['p'][i] = 0
        matr['p'][i+1] = matr['p'][i+1] - R * matr['c'][i]
        matr['f'][k] = matr['f'][k] - R * matr['f'][i]
        R = matr['q'][i]
        matr['q'][i] = 0
        matr['q'][i + 1] = matr['q'][i + 1] - R * matr['c'][i]

if __name__ == '__main__':
    a = np.full(16, 2)
    b = np.full(15, 1)
    c = np.full(15, 3)
    p = np.full(16, 4)
    q = np.full(16, 5)
    f = np.arange(16)
    matr = Matrix(a, b, c, p, q, f)
    print(matr.matrix)
    # matr['a'] = np.arange(16)
    make_first_step(matr)
    print(matr.matrix)
