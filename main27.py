# -*- coding: utf-8 -*-
"""
 author: weihuan
 date: 2020/5/20  23:07
"""
# 测试编辑距离


def str_distance (str1, str2):
    """
    计算字符串 str1 和 str2 的编辑距离
    :param str1
    :param str2
    :return:
    """
    matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]

    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if (str1[i - 1] == str2[j - 1]):
                d = 0
            else:
                d = 1

            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)

    return matrix[len(str1)][len(str2)]

print(str_distance("a","a"))


# 下面的方法采用库来完成
import  Levenshtein
print(Levenshtein.distance("今天中午吃饭了吗","今天中午刚吃了海鲜"))