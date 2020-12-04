class MatrixCalculator:

    @staticmethod
    def is_matrix(mat):
        try:
            if not isinstance(mat, list):
                return False
            rows = len(mat)
            cols = len(mat[0])
            for i in range(rows):
                for j in range(cols):
                    if not isinstance(mat[i][j], (int, float)):
                        return False
                    if len(mat[i]) != cols:
                        return False
            return True
        except IndexError:
            return False

    @staticmethod
    def get_matrix_size(mat):
        return [len(mat), len(mat[0])]

    @staticmethod
    def is_squared_matrix(mat):
        size_mat = MatrixCalculator.get_matrix_size(mat)
        return size_mat[0] == size_mat[1]

    @staticmethod
    def mat_sum(mat_a, mat_b):
        message = None
        try:
            if not MatrixCalculator.is_matrix(mat_a) and not MatrixCalculator.is_matrix(mat_b):
                message = "Values to be summed are not matrices"
                raise Exception
            size_a = MatrixCalculator.get_matrix_size(mat_a)
            size_b = MatrixCalculator.get_matrix_size(mat_b)
            if size_a != size_b:
                message = "The matrices must have the same size"
                raise Exception()
            sum_matrix = []
            for i in range(len(mat_a)):
                sun_row = []
                for j in range(len(mat_b[i])):
                    sun_row.append(mat_a[i][j] + mat_b[i][j])
                sum_matrix.append(sun_row)
            return sum_matrix
        except Exception as err:
            if not message:
                message = "Error: {}".format(err)
            print(message)

    @staticmethod
    def mat_scalar_mult(scalar, mat_a):
        message = None
        try:
            if not isinstance(scalar, (int, float)):
                raise ValueError
            if not MatrixCalculator.is_matrix(mat_a):
                message = "Values to be multiplied should be a scalar and a matrix"
                raise Exception
            scalar_mat = []
            for i in range(len(mat_a)):
                scalar_row = []
                for j in range(len(mat_a[i])):
                    scalar_row.append(scalar * mat_a[i][j])
                scalar_mat.append(scalar_row)
            return scalar_mat
        except ValueError:
            message = "The scalar should be numeric"
            print(message)
        except Exception as err:
            if not message:
                message = "Error: {}".format(err)
            print(message)

    @staticmethod
    def mat_mult(mat_a, mat_b):
        message = None
        try:
            if not MatrixCalculator.is_matrix(mat_a) and not MatrixCalculator.is_matrix(mat_b):
                message = "Values to be multiplied must be matrices"
                raise Exception
            size_a = MatrixCalculator.get_matrix_size(mat_a)
            size_b = MatrixCalculator.get_matrix_size(mat_b)
            if size_a[1] != size_b[0]:
                message = "The size of the columns of the first matrix should be the same of the size of rows of the " \
                          "second matrix "
                raise Exception
            mult_mat = []
            for i in range(len(mat_a)):
                mult_row = []
                for j in range(len(mat_b[0])):
                    temp_c = 0
                    for k in range(len(mat_b)):
                        temp_c = temp_c + mat_a[i][k] * mat_b[k][j]
                    mult_row.append(temp_c)
                mult_mat.append(mult_row)
            return mult_mat
        except Exception as err:
            if not message:
                message = "Error: {}".format(err)
            print(message)

    @staticmethod
    def mat_transpose(mat_a):
        message = None
        try:
            if not MatrixCalculator.is_matrix(mat_a):
                message = "The value to get the transpose must be a matrix"
            transpose_mat = []
            for i in range(len(mat_a[0])):
                trans_row = []
                for j in range(len(mat_a)):
                    trans_row.append(mat_a[j][i])
                transpose_mat.append(trans_row)
            return transpose_mat
        except Exception as err:
            if not message:
                message = "Error: {}".format(err)
            print(message)

    @staticmethod
    def mat_determinant(mat_a):
        message = None
        try:
            if not MatrixCalculator.is_matrix(mat_a):
                message = "The value to get the determinant must be a matrix"
                raise Exception
            if not MatrixCalculator.is_squared_matrix(mat_a):
                message = "The matrix must be squared to get the determinant"
                raise Exception
            if len(mat_a) == 1:
                return MatrixCalculator.get_determinant_dim_1(mat_a)
            else:
                return MatrixCalculator.get_determinant_dim_n(mat_a)
        except Exception as err:
            if not message:
                message = "Error: {}".format(err)
            print(message)

    @staticmethod
    def mat_adjugate(mat_a):
        message = None
        index_list = 0
        mat_adj = []
        try:
            if not MatrixCalculator.is_matrix(mat_a):
                message = "To get the adjugate matrix you need to enter a matrix."
                raise Exception
            if not MatrixCalculator.is_squared_matrix(mat_a):
                message = "To get the adjugate matrix, the matrix should be squared."
                raise Exception
            mat_transp = MatrixCalculator.mat_transpose(mat_a)
            cofactors = MatrixCalculator.calculate_cof_mat(mat_transp)
            size = len(mat_transp)
            for i in range(size):
                temp_row = []
                for j in range(size):
                    temp_row.append(cofactors[index_list])
                    index_list += 1
                mat_adj.append(temp_row)
            return mat_adj
        except Exception as err:
            if not message:
                message = "Error: {}".format(err)
            print(message)

    @staticmethod
    def mat_inverse(mat_a):
        message = None
        try:
            if not MatrixCalculator.is_matrix(mat_a):
                raise Exception
            if not MatrixCalculator.is_squared_matrix(mat_a):
                raise Exception
            det = MatrixCalculator.mat_determinant(mat_a)
            scalar = 1/det
            adj = MatrixCalculator.mat_adjugate(mat_a)
            mat_inverse = MatrixCalculator.mat_scalar_mult(scalar, adj)
            return mat_inverse
        except ZeroDivisionError:
            message = "To determinant of the matrix is Zero"
            print(message)
        except Exception as err:
            if not message:
                message = "Error: {}".format(err)
            print(message)

    @staticmethod
    def get_rows(mat_a, indexes):
        rows_selected = []
        exists = False
        try:
            for i in indexes:
                for j in range(len(rows_selected)):
                    if rows_selected[j] == mat_a[i]:
                        exists = True
                if not exists:
                    rows_selected.append(mat_a[i])
                exists = False
            return rows_selected
        except IndexError:
            print("Index does not exists in the matrix.")

    @staticmethod
    def get_cols(mat_a, indexes):
        mat_transp = MatrixCalculator.mat_transpose(mat_a)
        selected_cols = MatrixCalculator.get_rows(mat_transp, indexes)
        return selected_cols

    @staticmethod
    def cut_matrix(mat_a, from_col_index, numb_cols):
        cols = []
        mat_cut = []
        index_list = 0
        try:
            mat_transp = MatrixCalculator.mat_transpose(mat_a)
            for i in range(from_col_index, from_col_index + numb_cols):
                cols.append(mat_transp[i])
            sub_mat = MatrixCalculator.mat_transpose(cols)
            return sub_mat
        except IndexError:
            print("Index does not exists in the matrix. ")

    @staticmethod
    def get_matrix_reduced(adj_i, adj_j, mat_to_reduce):
        mat_red = []
        for i in range(len(mat_to_reduce)):
            temp_row = []
            for j in range(len(mat_to_reduce)):
                if i != adj_i and j != adj_j:
                    temp_row.append(mat_to_reduce[i][j])
            if temp_row:
                mat_red.append(temp_row)
        return mat_red

    @staticmethod
    def calculate_cof_mat(mat_a):
        cofactors = []
        for k in range(len(mat_a)):
            for l in range(len(mat_a)):
                aux_mat = MatrixCalculator.get_matrix_reduced(k, l, mat_a)
                cofactors.append(MatrixCalculator.calculate_cof_det(k, l, aux_mat))
        return cofactors

    @staticmethod
    def get_determinant_dim_1(mat_a):
        return mat_a[0][0]

    @staticmethod
    def get_determinant_dim_n(mat_a):
        cofactors = []
        elements = []
        for k in range(len(mat_a)):
            aux_mat = MatrixCalculator.get_matrix_reduced(k, 0, mat_a)
            cofactors.append(MatrixCalculator.calculate_cof_det(k, 0, aux_mat))
            elements.append(mat_a[k][0])
        det_a = sum(list(map(lambda x, y: x * y, cofactors, elements)))
        return det_a

    @staticmethod
    def calculate_cof_det(index_i, index_j, mat_a):
        if (len(mat_a)) == 1:
            minor_comp = MatrixCalculator.get_determinant_dim_1(mat_a)
        else:
            minor_comp = MatrixCalculator.get_determinant_dim_n(mat_a)
        return pow(-1, index_i + index_j) * minor_comp

    @staticmethod
    def print_matrix(mat_a):
        mat_print = ""
        if MatrixCalculator.is_matrix(mat_a):
            for i in range(len(mat_a)):
                for j in range(len(mat_a[i])):
                    mat_print = mat_print + "\t{:5}".format(mat_a[i][j])
                mat_print = mat_print + "\n"
            m_size = [len(mat_a), len(mat_a[0])]
            mat_print = mat_print + "Matrix Size: {}".format(m_size)
            return mat_print
        else:
            return "The value is not a matrix."