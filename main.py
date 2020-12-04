import time
import msvcrt
from os import system


class SizeError(Exception):
    pass


class DimensionError(Exception):
    pass


class DimensionNoneZero(Exception):
    pass


class Matrix:

    def __init__(self, name, dimensionx, dimensiony):
        self.name = name
        self.dimensionx = dimensionx
        self.dimensiony = dimensiony
        self.matrix = self.create_matrix(dimensionx, dimensiony)

    def create_matrix(self, r, c):
        matrix = []
        for i in range(r):
            row = []
            for j in range(c):
                row.append(None)
            matrix.append(row)
        return matrix

    def fill_matrix_from_mat_list(self, matrix_list):
        try:
            rows = len(matrix_list)
            columns = len(matrix_list[0])
            if self.get_matrix_size() != [rows, columns]:
                raise DimensionError()
            for i in range(rows):
                temp_row = []
                for j in range(columns):
                    temp_row.append(matrix_list[i][j])
                self.matrix[i] = temp_row
        except DimensionError:
            print("The matrix list differ to the matrix dimensions")

    def fill_matrix_from_single_list(self, single_list, dim_i, dim_j):
        try:
            index_list = 0
            temp_mat = []
            if self.dimensionx * self.dimensiony != len(single_list):
                raise DimensionError()
            for i in range(dim_i):
                temp_row = []
                for j in range(dim_j):
                    temp_row.append(single_list[index_list])
                    index_list += 1
                self.matrix[i] = temp_row
        except DimensionError:
            print("The list size differ to the matrix dimensions")

    def fill_matrix(self):
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix[i])):
                while True:
                    try:
                        print("Please insert a numeric value for the row {} column {} for the matrix {}".format(i, j, self.name))
                        value = float(input())
                        self.matrix[i][j] = value
                        break
                    except ValueError:
                        print("You must insert a float value.")
                    except:
                        print("Unexpected error")
                        raise

    def is_square_matrix(self):
        size_m = self.get_matrix_size()
        if size_m[0] != size_m[1]:
            return False
        else:
            return True

    def get_matrix_order(self):
        size_m = self.get_matrix_size()
        if self.is_square_matrix():
            return size_m[0]
        else:
            raise SizeError()

    def get_matrix(self):
        return self.matrix

    def __str__(self):
        mat_print = "The matrix {} is:\n\n".format(self.name)
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix[i])):
                mat_print = mat_print + "\t{:5}".format(self.matrix[i][j])
            mat_print = mat_print + "\n"
        m_size = self.get_matrix_size()
        mat_print = mat_print + "Matrix Size: {}".format(m_size)
        return mat_print

    def get_matrix_size(self):
        #size = [len(self.matrix), len(self.matrix[0])]
        size = [self.dimensionx, self.dimensiony]
        return size


class Cofactor:

    def __init__(self, matrix_det):
        self.matrix_det = matrix_det
        self.mat_calc = MatrixCalculator()
        self.cofactors = []
        self.elements = []

    def calculate_cofactors_dim_n(self):
        temp_a = self.matrix_det.get_matrix()
        for k in range(len(temp_a)):
            aux_mat = self.get_matrix_reduced(k, 0, temp_a)
            self.cofactors.append(self.calculate_det(k, 0, aux_mat))
            self.elements.append(temp_a[k][0])

    def calculate_cofactors_mat(self):
        temp_a = self.matrix_det.get_matrix()
        for k in range(len(temp_a)):
            for l in range(len(temp_a)):
                aux_mat = self.get_matrix_reduced(k, l, temp_a)
                self.cofactors.append(self.calculate_det(k, l, aux_mat))

    def calculate_det(self, index_i, index_j, auxiliar_matrix):
        aux_mat = auxiliar_matrix.get_matrix()
        if (len(aux_mat)) == 1:
            minor_comp = self.mat_calc.matrix_det_dim_1(auxiliar_matrix)
            cof = pow(-1, index_i + index_j) * minor_comp
            return cof
        else:
            minor_comp = self.mat_calc.matrix_det_dim_n(auxiliar_matrix)
            cof = pow(-1, index_i + index_j) * minor_comp
            return cof

    def get_matrix_reduced(self, adj_i, adj_j, mat_list_to_reduce):
        mat_red = []
        for i in range(len(mat_list_to_reduce)):
            temp_row = []
            for j in range(len(mat_list_to_reduce)):
                if i != adj_i and j != adj_j:
                    temp_row.append(mat_list_to_reduce[i][j])
            if temp_row:
                mat_red.append(temp_row)
        #mat_adj = self.mat_calc.convert_mat_from_mat_list("C_{}{}".format(adj_i, adj_j), mat_red)
        mat_adj = Matrix("C_{}{}".format(adj_i, adj_j), len(mat_red), len(mat_red[0]))
        mat_adj.fill_matrix_from_mat_list(mat_red)
        return mat_adj

    def get_elements(self):
        return self.elements

    def get_cofactors(self):
        return self.cofactors


class MatrixCalculator:

    def matrix_sum(self, matrix_a, matrix_b, result_name):
        try:
            size_a = matrix_a.get_matrix_size()
            size_b = matrix_b.get_matrix_size()
            if size_a != size_b:
                raise SizeError()
            temp_a = matrix_a.get_matrix()
            temp_b = matrix_b.get_matrix()
            sum_matrix = []
            for i in range(len(temp_a)):
                sun_row = []
                for j in range(len(temp_b[i])):
                    sun_row.append(temp_a[i][j] + temp_b[i][j])
                sum_matrix.append(sun_row)
            #sum_result = self.convert_mat_from_mat_list(result_name, sum_matrix)
            #return sum_result
            result = Matrix(result_name, len(sum_matrix), len(sum_matrix[i]))
            result.fill_matrix_from_mat_list(sum_matrix)
            return result
        except SizeError as e:
            print("The matrices to be summed should have the same size.")
            print("\nPress any key to continue\n")
            msvcrt.getch()
            main()
        except:
            print("Unexpected error")
            raise

    def scalar_mult(self, scalar, matrix_a, result_name):
        temp_a = matrix_a.get_matrix()
        mult_matrix = []
        for i in range(len(temp_a)):
            mult_row = []
            for j in range(len(temp_a[i])):
                mult_row.append((temp_a[i][j] * scalar))
            mult_matrix.append(mult_row)
        #mult_result = self.convert_mat_from_mat_list(result_name, mult_matrix)
        print(mult_row)
        result = Matrix(result_name, len(mult_matrix), len(mult_matrix[0]))
        result.fill_matrix_from_mat_list(mult_matrix)
        #return mult_result
        return result

    def matrix_mult(self, matrix_a, matrix_b, result_name):
        try:
            size_a = matrix_a.get_matrix_size()
            size_b = matrix_b.get_matrix_size()
            if size_a[1] != size_b[0]:
                raise SizeError
            temp_a = matrix_a.get_matrix()
            temp_b = matrix_b.get_matrix()
            mult_mat = []
            for i in range(len(temp_a)):
                mult_row = []
                for j in range(len(temp_b[0])):
                    temp_c = 0
                    for k in range(len(temp_b)):
                        temp_c = temp_c + temp_a[i][k] * temp_b[k][j]
                    mult_row.append(temp_c)
                mult_mat.append(mult_row)
            #mat_mult_result = self.convert_mat_from_mat_list(result_name, mult_mat)
            #return mat_mult_result
            result = Matrix(result_name, len(mult_mat), len(mult_mat[i]))
            result.fill_matrix_from_mat_list(mult_mat)
            return result
        except SizeError:
            print("The number of rows of the matrix {} should be the same of number of rows of the matrix {}".format(
                matrix_a.name, matrix_b.name))
            print("\nPress any key to continue\n")
            msvcrt.getch()
            main()
        except:
            print("Unexpected error")
            raise

    def matrix_transpose(self, matrix_a, result_name):
        temp_a = matrix_a.get_matrix()
        transpose_mat = []
        for i in range(len(temp_a[0])):
            trans_row = []
            for j in range(len(temp_a)):
                trans_row.append(temp_a[j][i])
            transpose_mat.append(trans_row)
        #mat_trans_result = self.convert_mat_from_mat_list(result_name, transpose_mat)
        #return mat_trans_result
        result = Matrix(result_name, len(transpose_mat), len(transpose_mat[i]))
        result.fill_matrix_from_mat_list(transpose_mat)
        return result


    def matrix_determinant(self, ma):
        try:
            order = ma.get_matrix_order()
            if order == 1:
                return self.matrix_det_dim_1(ma)
            else:
                return self.matrix_det_dim_n(ma)
        except SizeError:
            print("To get the determinant of a matrix, this one should have a square dimension")
            print("\nPress any key to continue\n")
            msvcrt.getch()
            main()
        except:
            print("Unexpected error")
            raise

    def matrix_adjugate(self, ma, result_name):
        try:
            if not ma.is_square_matrix():
                raise SizeError()
            ta = self.matrix_transpose(ma, "AtTemp")
            cof = Cofactor(ta)
            cof.calculate_cofactors_mat()
            list_cof = cof.get_cofactors()
            #mat_adj = self.convert_mat_from_single_list(result_name, list_cof, len(ta.get_matrix()), len(ta.get_matrix()))
            #return mat_adj
            result = Matrix(result_name, ta.dimensionx, ta.dimensiony)
            result.fill_matrix_from_single_list(list_cof, ta.dimensionx, ta.dimensiony)
            return result
        except SizeError:
            print("To get the adjugate matrix, this one should have a square dimension")
            print("\nPress any key to continue\n")
            msvcrt.getch()
            main()
        except:
            print("Unexpected error")
            raise

    def matrix_inverse(self, ma, result_name):
        try:
            if not ma.is_square_matrix():
                raise SizeError()
            det = self.matrix_determinant(ma)
            scalar = 1/det
            adj = self.matrix_adjugate(ma, "AdjTemp")
            inverse = self.scalar_mult(scalar, adj, result_name)
            return inverse
        except SizeError:
            print("To get the inverse of matrix, this one should have a square dimension")
            print("\nPress any key to continue\n")
            msvcrt.getch()
            main()
        except ZeroDivisionError:
            print("To determinant of the matrix is Zero")
            print("\nPress any key to continue\n")
            msvcrt.getch()
            main()
        except:
            print("Unexpected error")
            raise

    def get_rows(self, indexes, mat):
        rows_selected = []
        exists = False
        try:
            for i in indexes:
                for j in range(len(rows_selected)):
                    if rows_selected[j] == mat.get_matrix()[i]:
                        exists = True
                if not exists:
                    rows_selected.append(mat.get_matrix()[i])
                exists = False
            #new_matrix = self.convert_mat_from_mat_list("ROWS", rows_selected)
            #return new_matrix
            result = Matrix("ROWS", len(rows_selected), len(rows_selected[0]))
            result.fill_matrix_from_mat_list(rows_selected)
            return result
        except IndexError:
            print("Index does not exists in the matrix.")

    def get_columns(self, indexes, mat):
        mat_transp = self.matrix_transpose(mat, "TEMP_At")
        temp_result = self.get_rows(indexes, mat_transp)
        new_matrix = self.matrix_transpose(temp_result, "COLUMNS")
        return new_matrix

    def cut_matrix(self, from_col_index, numb_cols, mat):
        cols = []
        try:
            mat_transp = self.matrix_transpose(mat, "TEMP_At")
            for i in range(from_col_index, from_col_index + numb_cols):
                cols.append(mat_transp.get_matrix()[i])
            temp_result = Matrix("TEMP_RESULT", len(cols), len(cols[0]))
            temp_result.fill_matrix_from_mat_list(cols)
            result = self.matrix_transpose(temp_result, "MATRIX_CUT")
            return result
        except IndexError:
            print("Index does not exists in the matrix.")

    def matrix_det_dim_1(self, ma):
        temp_a = ma.get_matrix()
        det_a = temp_a[0][0]
        return det_a

    def matrix_det_dim_n(self, ma):
        cofactors = []
        elements = []
        cf = Cofactor(ma)
        cf.calculate_cofactors_dim_n()
        elements = cf.get_elements()
        cofactors = cf.get_cofactors()
        det_a = sum(list(map(lambda x, y: x * y, cofactors, elements)))
        return det_a


def present_options(calculator):
    while True:
        print('''Please select one of the following matrices operations.
              1 - Matrices Addition
              2 - Scalar Multiplication
              3 - Matrices Multiplication
              4 - Matrix Transpose
              5 - Matrix Determinant
              6 - Adjugate Matrix 
              7 - Inverted Matrix
              8 - Extract rows
              9 - Extract columns
             10 - Cut Matrix 
             11 - Exit ''')
        option = input()
        clear_screen(0.5)
        if option == "1":
            ma = define_matrix("A")
            mb = define_matrix("B")
            result = calculator.matrix_sum(ma, mb, "A+B")
            clear_screen(0.5)
            print("Result of matrices addition: ")
            print(ma.__str__())
            print(mb.__str__())
            print(result.__str__())
            print("\nPress any key to continue\n")
            msvcrt.getch()
            clear_screen(0)
        elif option == "2":
            try:
                while True:
                    print("Please insert the scalar to multiply.")
                    scalar = int(input())
                    break
            except ValueError:
                print("You must insert a number. Please try it again.")
            except:
                print("Something went wrong.")
                raise
            ma = define_matrix("A")
            result = calculator.scalar_mult(scalar, ma, "s*A")
            print("Result of scalar multiplication: ")
            print("Scalar: {}".format(scalar))
            print(ma.__str__())
            print(result.__str__())
            print("\nPress any key to continue\n")
            msvcrt.getch()
            clear_screen(0)
        elif option == "3":
            ma = define_matrix("A")
            mb = define_matrix("B")
            result = calculator.matrix_mult(ma, mb, "A*B")
            clear_screen(0.5)
            print("Result of matrices multiplication: ")
            print(ma.__str__())
            print(mb.__str__())
            print(result.__str__())
            print("\nPress any key to continue\n")
            msvcrt.getch()
            clear_screen(0)
        elif option == "4":
            ma = define_matrix("A")
            result = calculator.matrix_transpose(ma, "At")
            print("Result of matrix transpose: ")
            print(ma.__str__())
            print(result.__str__())
            print("\nPress any key to continue\n")
            msvcrt.getch()
            clear_screen(0)
        elif option == "5":
            ma = define_matrix("A")
            result = calculator.matrix_determinant(ma)
            print("Result of the matrix determinant: ")
            print(ma.__str__())
            print("D: {}".format(result))
            print("\nPress any key to continue\n")
            msvcrt.getch()
            clear_screen(0)
        elif option == "6":
            ma = define_matrix("A")
            result = calculator.matrix_adjugate(ma, "Adj(A)")
            print("Result of the adjugate matrix: ")
            print(ma.__str__())
            print(str(result))
            print("\nPress any key to continue\n")
            msvcrt.getch()
            clear_screen(0)
        elif option == "7":
            ma = define_matrix("A")
            result = calculator.matrix_inverse(ma, "A^-1")
            print("Result of the inverse matrix: ")
            print(ma.__str__())
            print(result.__str__())
            print("\nPress any key to continue\n")
            msvcrt.getch()
            clear_screen(0)
        elif option == "8":
            ma = define_matrix("A")
            l_indexes = ask_indexes("rows")
            result = calculator.get_rows(l_indexes, ma)
            print("Result of the rows selected: ")
            print(ma.__str__())
            print(result.__str__())
            print("\nPress any key to continue\n")
            msvcrt.getch()
            clear_screen(0)
        elif option == "9":
            ma = define_matrix("A")
            l_indexes = ask_indexes("columns")
            result = calculator.get_columns(l_indexes, ma)
            print("Result of the columns selected: ")
            print(ma.__str__())
            print(result.__str__())
            print("\nPress any key to continue\n")
            msvcrt.getch()
            clear_screen(0)
        elif option == "10":
            ma = define_matrix("A")
            from_to = ask_from_to()
            result = calculator.cut_matrix(from_to[0], from_to[1], ma)
            print("Result of the cut matrix is: ")
            print(ma.__str__())
            print(result.__str__())
            print("\nPress any key to continue\n")
            msvcrt.getch()
            clear_screen(0)
        elif option == "11":
            break
        else:
            print("Invalid Option. Please try it again.")


def ask_dimension(typo, name_matrix):
    while True:
        try:
            print("Please insert the number of {} for the matrix {}".format(typo, name_matrix))
            x = int(input())
            if x == 0:
                raise Exception()
            return x
        except ValueError:
            print("You must insert a numeric value.\n")
        except DimensionNoneZero:
            print("The number should be greater than 0 (zero)\n")
        except:
            print("Unexpected error")
            raise


def ask_indexes(typo):
    try:
        rows = []
        print("How many {} do you want to retrieve?".format(typo))
        num_rows = int(input())
        for i in range(num_rows):
            if i == 0:
                print("Please insert the first index of the {} you want to get.".format(typo))
            else:
                print("Please insert the next index of the {} you want to get.".format(typo))
            rows.append(int(input()))
        return rows
    except ValueError:
        print("You should insert only numeric numbers.")
    except:
        print("Unexpected error")
        raise


def ask_from_to():
    from_to = []
    while True:
        try:
            print("Please insert the index from where you want to cut the matrix")
            f = int(input())
            if f < 0:
                print("You must insert a number greater than 0")
                raise Exception()
            from_to.append(f)
            print("Please insert how many columns you want to get")
            t = int(input())
            if t <= 0:
                print("You mast insert at least one column")
            from_to.append(t)
            return(from_to)
        except ValueError:
            print("You must insert a numeric value.\n")
        except:
            print("Unexpected error")
            raise


def define_matrix(m_name):
    r = ask_dimension("rows", m_name)
    c = ask_dimension("columns", m_name)
    clear_screen(0.5)
    mtx = Matrix(m_name, r, c)
    mtx.fill_matrix()
    clear_screen(0.5)
    return mtx

def clear_screen(delay):
    time.sleep(delay)
    system('cls')

def main():
    print("Matrix Calculator")
    print("-----------------\n")
    calculator = MatrixCalculator()
    present_options(calculator)

def main2():
    print("TEST")
    print("-----------------\n")
    print("-----------------\n")
    c = MatrixCalculator()
    #a = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    #a = [[8, 52, -2, 3], [0, 15, 3, -4], [3, 1, 2, 3], [4, 5, 6, 3]]
    a = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]]
    #a = [[11, 12, 13, 14, 15], [12, 12, -9, 1, 1], [1, 5, 3, 8, 88], [1, 2, 7, 89, 6], [2, 14, 5, 30, 21]]
    #a = [[0.0, -1, 2, 0], [-1, 2, 2, 3], [-2, 0, 1, 0], [0, 1, 2, 0]]
    #a= [[2,3], [3,4]]
    #a = [[0.1, -1, 2], [-1, 2, 3], [-2, 0, 1]]
    #a = [[11, 12, 13, 14, 15], [12, 12, -9, 1, 1], [1, 5, 3, 8, 88], [0, 0, 0, 0, 0], [2, 14, 5, 30, 21]]
    #a = [[8, 52, 0, 3], [0, 15, 0, -4], [3, 1, 0, 3], [4, 5, 0, 3]]

    '''ma = Matrix("TEMP", len(a), len(a))
    for i in range(len(a)):
        temp_row = []
        for j in range(len(a)):
            temp_row.append(a[i][j])
        ma.get_matrix()[i] = temp_row

    print(ma.__str__())
    det = c.matrix_det_dim_n(ma)
    print("Determinant: {}".format(det))'''

    '''a = [[0, -1, 2], [-1, 2, 3], [-2, 0, 1]]
    ma = Matrix("TEMP", len(a), len(a))
    for i in range(len(a)):
        temp_row = []
        for j in range(len(a)):
            temp_row.append(a[i][j])
        ma.get_matrix()[i] = temp_row

    print(ma.__str__())
    cf = Cofactor(ma)
    m = cf.get_matrix_reduced(0, 1, ma.get_matrix())
    print(m.__str__())'''

    '''ma = Matrix("TEMP", len(a), len(a))
    for i in range(len(a)):
        temp_row = []
        for j in range(len(a)):
            temp_row.append(a[i][j])
        ma.get_matrix()[i] = temp_row
    print(ma.__str__())
    ta = c.transpose_matrix(ma)
    print(ta.__str__())

    cofactors = []
    cof = Cofactor(ta)
    cof.calculate_cofactors_mat()
    list_cof = cof.get_cofactors()
    print(type(list_cof))
    print(list_cof)
    result = c.convert_mat_from_single_list("ADJ", list_cof, len(ta.get_matrix()), len(ta.get_matrix()))
    print(result.__str__())'''

    ma = Matrix("TEMP", len(a), len(a))
    for i in range(len(a)):
        temp_row = []
        for j in range(len(a[i])):
            temp_row.append(a[i][j])
        ma.get_matrix()[i] = temp_row
    print(ma.__str__())

    rows = []
    mat = c.cut_matrix(1, 2, ma)
    print(mat.__str__())

if __name__ == "__main__":
    main()
