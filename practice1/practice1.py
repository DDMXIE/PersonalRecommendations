# @Author : tony
# @Date   : 2021/4/2
# @Title  : pre2007 paper practice
# @Dec    : NBI GRM CF Algorithm implementation

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from fractions import Fraction
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from decimal import Decimal
from sklearn.metrics.pairwise import cosine_similarity

# compute weight matrix through the degree of nodes
def crtWeightMatrix(matrix,rowlen):
    new_matrix = []
    for i in range(0, rowlen):
        matrix_np = np.array(matrix)
        include1Cluster = sum(matrix_np[i] == 1)
        print(include1Cluster)

        matrix_np_i = np.array(matrix[i])
        np.set_printoptions(formatter={'all': lambda x: str(Fraction(x).limit_denominator())})
        if include1Cluster != 0:
            matrix_np_i = matrix_np_i / include1Cluster
        new_matrix.append(matrix_np_i)
        print(matrix_np_i)

    return new_matrix

# compute the final very adjacent matrix
def getVeryAdjacentMatrix(df,df2):
    # # compute the degree and get degree_weight_matrix
    degree_weight_matrix = crtWeightMatrix(df.values, 1682)
    print(degree_weight_matrix)
    print(degree_weight_matrix[1681])

    # compute the degree and get degree_weight_matrix_2
    degree_weight_matrix_2 = crtWeightMatrix(df2.values, 943)
    print(degree_weight_matrix_2)
    print(degree_weight_matrix_2[942])

    # # test matrix use this
    # # # compute the degree and get degree_weight_matrix
    # degree_weight_matrix = crtWeightMatrix(df.values, 3)
    # print(degree_weight_matrix)
    # print(degree_weight_matrix[2])
    #
    # # compute the degree and get degree_weight_matrix_2
    # degree_weight_matrix_2 = crtWeightMatrix(df2.values, 4)
    # print(degree_weight_matrix_2)
    # print(degree_weight_matrix_2[3])

    # use dataframe transform the degree_weight_matrix
    df_weight = pd.DataFrame(degree_weight_matrix)
    print('---- your df_weight is -----')
    print(df_weight)

    # use dataframe transform the degree_weight_matrix_2
    df_weight_2 = pd.DataFrame(degree_weight_matrix_2)
    print('---- your df_weight2 is -----')
    print(df_weight_2)

    df_weight_T = pd.DataFrame(df_weight.values.T, index=df_weight.columns, columns=df_weight.index)  # 转置
    df_weight_2_T = pd.DataFrame(df_weight_2.values.T, index=df_weight_2.columns, columns=df_weight_2.index)  # 转置
    print('---- your df_weight_T is -----')
    print(df_weight_T)
    print('---- your df_weight_2_T is -----')
    print(df_weight_2_T)

    # do matrix multiplication
    res = np.dot(df_weight_2_T, df_weight_T)
    # get the very adjacent matrix
    df_res = pd.DataFrame(res)
    print('---- your very adjacent matrix is -----')
    print(df_res)
    return df_res

def getSimilarityMatrix(df,df2):
    newMatrix = []

    # test
    # rowlen = 4
    rowlen = 943

    print('------ start -------')
    print(df)
    print('\n')
    print(df2)
    dotMatrix = np.dot(df2, df)
    print(dotMatrix)

    for i in range(0, rowlen):
        arr = np.array(df.sum(axis=0))
        arr1 = arr
        arr1[arr1 > arr1[i]] = arr1[i]
        c = np.array(dotMatrix[i]) / np.array(arr1)
        # print(arr1)
        newMatrix.append(c)
    # cos_matrix = cosine_similarity(df2)
    # print(cos_matrix)
    print(newMatrix)
    print('------ end -------')
    return newMatrix

# get the rank array and draw the figure
def getRankArrayNBI(matrix, adjMatrix, testMatrix):
    rankList = []
    # B = [[1, 2, 3, 881250949], [3, 1, 3, 891717742]]
    df_test = pd.DataFrame(testMatrix)
    # df_test = pd.DataFrame(B)
    for row in df_test.values:
        arr = np.array(matrix[row[0] - 1])
        include0Cluster = sum(arr == 0)
        if matrix[row[0]-1][row[1]-1] == 0:
            result = np.dot(adjMatrix, matrix[row[0]-1])
            # change a row to a col
            a = np.array(result)
            a = a.reshape(-1, 1)
            b = pd.DataFrame(a)

            # print(np.array(matrix[row[0]-1]))
            arr1 = np.array(matrix[row[0]-1])
            g = np.where(arr1 == 1)
            # print(g)
            # exclude that have already been collected
            for each in g[0]:
                b.drop([each], inplace=True)
            # rank desc
            b = b[0].rank(ascending=False)

            rank = b[row[1] - 1] / include0Cluster
            rankList.append(rank)
            print('----- computing -----')
    print(rankList)
    rankList_sort = sorted(rankList)
    print(rankList_sort)
    print(pd.DataFrame(rankList_sort))
    df_rank = pd.DataFrame(rankList_sort)
    df_rank[1] = df_rank[0].rank(method="first")
    print(df_rank)
    rankArray = np.array(rankList)
    return (rankArray,df_rank)
    #
    # average_rank = np.mean(rankArray)
    # average_rank_final = Decimal(average_rank).quantize(Decimal("0.000"))
    # label = 'NBI <r>=' + str(average_rank_final)
    # print(average_rank_final)
    # print(label)
    #
    # x = np.array(df_rank[1].values)
    # y = np.array(df_rank[0].values)
    # plt.ylim(-0.1, 1.05)
    # plt.plot(x, y, color='b', label=label, linestyle=':')
    # plt.xlabel("Rank")
    # plt.ylabel("r")
    # plt.legend()
    # plt.show()

# get the rank array and draw the figure
def getRankArrayGRM(matrix, adjMatrix, testMatrix):
    rankList = []
    # B = [[1, 2, 3, 881250949], [3, 1, 3, 891717742]]
    df_test = pd.DataFrame(testMatrix)
    # df_test = pd.DataFrame(B)
    print(matrix)

    # 创建全为1的数组(4行1列)
    t = np.ones((943, 1))
    # t = np.ones((4, 1))
    df_t = pd.DataFrame(t)
    result = np.dot(matrix, df_t)
    for row in df_test.values:
        df_degree = pd.DataFrame(result)
        # print(df_degree)
        arr = np.array(matrix[row[0] - 1])
        include0Cluster = sum(arr == 0)
        if matrix[row[0] - 1][row[1] - 1] == 0:
            arr1 = np.array(matrix[row[0] - 1])
            g = np.where(arr1 == 1)
            for each in g[0]:
                df_degree.drop([each], inplace=True)
            # print(df_degree)
            df_degree[1] = df_degree[0].rank(ascending=False)  # method可不加
            rank = df_degree[1][row[1] - 1] / include0Cluster
            rankList.append(rank)
            print('----- computing -----')
    print(rankList)
    rankList_sort = sorted(rankList)
    print(rankList_sort)
    print(pd.DataFrame(rankList_sort))
    df_rank = pd.DataFrame(rankList_sort)
    df_rank[1] = df_rank[0].rank(method="first")
    print(df_rank)
    rankArray = np.array(rankList)
    return (rankArray,df_rank)


def getRankArrayCF(matrix, simMatrix, testMatrix):
    rankList = []
    # B = [[1, 2, 3, 881250949], [3, 1, 3, 891717742]]
    df_test = pd.DataFrame(testMatrix)
    # df_test = pd.DataFrame(B)

    # test
    # simMatrix[np.eye(4, dtype=np.bool_)] = 0
    simMatrix[np.eye(943, dtype=np.bool_)] = 0

    for row in df_test.values:
        arr = np.array(matrix[row[0] - 1])
        include0Cluster = sum(arr == 0)
        if matrix[row[0] - 1][row[1] - 1] == 0:
            a = np.dot(matrix, simMatrix[row[0] - 1])
            a = a.reshape(-1, 1)
            b = pd.DataFrame(a) / sum(simMatrix[row[0] - 1])
            # print(b)
            arr1 = np.array(matrix[row[0] - 1])
            g = np.where(arr1 == 1)
            for each in g[0]:
                b.drop([each], inplace=True)
            # print(b)
            b = b[0].rank(ascending=False)

            rank = b[row[1] - 1] / include0Cluster
            rankList.append(rank)
            print('----- computing -----')
    print(rankList)
    rankList_sort = sorted(rankList)
    print(rankList_sort)
    print(pd.DataFrame(rankList_sort))
    df_rank = pd.DataFrame(rankList_sort)
    df_rank[1] = df_rank[0].rank(method="first")
    print(df_rank)
    rankArray = np.array(rankList)
    return (rankArray, df_rank)



def drawFigures(arrNBI, dfNBI, arrGRM, dfGRM, arrCF, dfCF):
    average_rank_NBI = np.mean(arrNBI)
    average_rank_GRM = np.mean(arrGRM)
    average_rank_CF = np.mean(arrCF)
    average_rank_NBI_final = Decimal(average_rank_NBI).quantize(Decimal("0.000"))
    average_rank_GRM_final = Decimal(average_rank_GRM).quantize(Decimal("0.000"))
    average_rank_CF_final = Decimal(average_rank_CF).quantize(Decimal("0.000"))

    label_GRM = 'GRM <r>=' + str(average_rank_GRM_final)
    label_NBI = 'NBI <r>=' + str(average_rank_NBI_final)
    label_CF = 'CF <r>=' + str(average_rank_CF_final)

    print(label_NBI)
    print(label_GRM)
    print(label_CF)

    x = np.array(dfNBI[1].values)
    y = np.array(dfNBI[0].values)
    y2 = np.array(dfGRM[0].values)
    y3 = np.array(dfCF[0].values)

    plt.ylim(-0.1, 1.05)
    plt.plot(x, y2, color='k', label=label_GRM)
    plt.plot(x, y3, color='r', label=label_CF, linestyle='--')
    plt.plot(x, y, color='b', label=label_NBI, linestyle=':')

    plt.xlabel("Rank")
    plt.ylabel("r")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    data = pd.read_csv('./data/u.data', sep='\t', index_col=False, header=None)
    print(data)

    df = pd.DataFrame(data)

    df_rating = df[df[2] >= 3]
    print(df_rating)

    df_sort = df_rating.sort_values(by=0)
    print(df_sort)

    # use sklearn to divide the dataset
    train, test, train_label, test_lable = train_test_split(df_rating, range(len(df_rating)), test_size=0.1)

    print('训练集 90% 数据：')
    print(train)
    print('测试集 10% 数据：')
    print(test)

    # init the adjacent matrix[i][j]
    # 1682 rows     943 cols
    matrix = [[0] * 943 for i in range(1682)]

    for row in train.values:
        i = row[1] - 1  # object
        j = row[0] - 1  # user
        matrix[i][j] = 1

    # test matrix
    # matrix = [[1, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1]]

    # use dataframe to transform the matrix
    df = pd.DataFrame(matrix)
    print(df)
    df2 = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
    print(df2)



    # NBI
    # use the function above to get the very adjacent matrix
    veryAdjacentMatrix = getVeryAdjacentMatrix(df, df2)
    # print('\nvery adjacent matrix dataframe')
    # print(veryAdjacentMatrix)
    # print('\n')
    df_veryAdjacentMatrix = pd.DataFrame(veryAdjacentMatrix)
    arr_NBI = getRankArrayNBI(df, df_veryAdjacentMatrix, test)

    # GRM
    arr_GRM = getRankArrayGRM(df, df_veryAdjacentMatrix, test)

    # CF
    similarityMatrix = getSimilarityMatrix(df, df2)
    df_similarityMatrix = pd.DataFrame(similarityMatrix)
    arr_CF = getRankArrayCF(df, df_similarityMatrix, test)

    # draw figures
    drawFigures(arr_NBI[0], arr_NBI[1], arr_GRM[0], arr_GRM[1], arr_CF[0], arr_CF[1])








