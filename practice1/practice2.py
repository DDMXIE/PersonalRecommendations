import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    data = pd.read_csv('./data/u.data', sep='\t', index_col=False, header=None)
    print(data)

    df = pd.DataFrame(data)
    print(data)

    df_0 = data[0]
    print(df_0)
    norepeat_df = df_0.drop_duplicates(keep='first')
    print(norepeat_df)

    df_1 = data[1]
    print(df_1)
    norepeat_df_1 = df_1.drop_duplicates(keep='first')
    print(norepeat_df_1)

    norepeat_df_1_sort = norepeat_df_1.sort_values()
    print(norepeat_df_1_sort)



    # df_rating = df[df[2] >= 3]
    # print(df_rating)

    # df_sort = df_rating.sort_values(by=0)
    # print(df_sort)
    # df_sort_1 = df_rating.sort_values(by=1)
    # print(df_sort_1)
    # print(len(df_rating.index))
    # # use sklearn to divide the dataset
    # train, test, train_label, test_lable = train_test_split(df_rating,range(len(df_rating)), test_size=0.1, random_state=len(df_rating.index))
    #
    #
    # print('训练集 90% 数据：')
    # print(train)
    # print('测试集 10% 数据：')
    # print(test)
    #
    # df_rating_test = test[test[2] >= 3]
    # print(df_rating_test)