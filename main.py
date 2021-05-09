# @author   :   tony
# @date     :   2021/4/1
# @title    :   pre2007 paper practice
# @dec      :   pre2007 FIG.2. bipartite representation

import string
from fractions import Fraction  # In terms of fractions

# get ith row's node degree
def getLNodeDegree(index, matrix,colLen):
    count = 0
    for j in range(0, colLen):
        if matrix[index][j] == 1:
            count = count + 1
    return count

# function that get matrix_T
def matrixT(matrix):
    rt = [[] for i in matrix[0]]
    for ele in matrix:
        for i in range(len(ele)):
            rt[i].append(ele[i])
    return rt

# function that compute the weight matrix
def crtWeightMatrix(matrix, colLen, indexLen):
    # start to create a new weight matrix
    weightMatrix = []
    for i in range(0, indexLen):
        temp = []
        for j in range(0, colLen):
            if matrix[i][j] == 1:
                # compute the node total degree
                degree = 0
                for k in range(0,indexLen):
                    degree = degree + matrix[k][j]
                fractional = Fraction(1, degree)
                temp.append(fractional)
            else:
                temp.append(0)
        weightMatrix.append(temp)

    return weightMatrix

# function that compute the very adjacent matrix
def crtVeryAdjMatrix (isAdjMatrix, weightMatrix, colLen, indexLen):

    veryAdjMatrix_T = []
    for idx in range(0,colLen):
        temp = []
        for j in range(0,colLen):
            count = 0
            for i in range(0,indexLen):
                if isAdjMatrix[i][j] == 1:
                    degree = getLNodeDegree(i, isAdjMatrix, colLen)
                    fractional = Fraction(1, degree)
                    count = count + fractional * weightMatrix[i][idx]

            if i == indexLen-1:
                temp.append(count)

        veryAdjMatrix_T.append(temp)

    # Transposed matrix
    veryAdjMatrix = matrixT(veryAdjMatrix_T)

    return veryAdjMatrix




if __name__ == '__main__':
    # input the upper node num,lower node num
    upperLen = (int)(input('upperLen='))    # numbers of upper node
    lowerLen = (int)(input('lowerLen='))    # numbers of lower node

    # input the upper node's weight
    weight_list = []
    for i in range(0, upperLen):
        weight_list.append((int)(input('node%d val='% i)))

    # --- matrix input method 1
    # input an adjacent matrix
    # A = []
    # for i in range(0, lowerLen):
    #     temp = []
    #     for j in range(0, upperLen):
    #         # input adjacent situation      0:not adjacent / 1:adjacent
    #         print('upper%d lower%d:'%(i,j))
    #         isAdj = (int)(input("isAdj="))
    #         temp.append(isAdj)
    #     A.append(temp)

    # --- matrix input method 2
    A = eval(input("please input your matrix:\n"))     # m*n m:lowerLen    n:upperLen  ex: 4*3 [[1, 0, 0], [1, 1, 1], [0, 1, 1], [1, 0, 1]]

    # Or you can initialize it directly
    # A = [[1, 0, 0], [1, 1, 1], [0, 1, 1], [1, 0, 1]]

    print('\n')
    print('---- your adjacent matrix is : -----')
    print('\n')
    for i in range(0, lowerLen):
        for j in range(0, upperLen):
            print(str(A[i][j]) + "   ", end='')
        print('\n')



    # compute step 1
    B = crtWeightMatrix(A, upperLen, lowerLen)

    # compute step 2
    veryAdjMatrix = crtVeryAdjMatrix(A, B, upperLen, lowerLen)

    # output
    print('\n')
    print('---- veryAdjMatrix is : -----')
    print('\n')

    for i in range(0, upperLen):
        for j in range(0, upperLen):
            print(str(veryAdjMatrix[i][j]) + "   ", end='')
        print('\n')