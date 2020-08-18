def get_num(str1, str2):
    lstr1 = len(str1)
    lstr2 = len(str2)
    record = [[0 for i in range(lstr2 + 1)] for j in range(lstr1 + 1)]  # 多一位
    maxNum = 0  # 最长匹配长度
    p = 0
    q = 0

    for i in range(lstr1):
        for j in range(lstr2):
            if str1[i] == str2[j]:
                # 相同则累加
                record[i + 1][j + 1] = record[i][j] + 1
                if record[i + 1][j + 1] > maxNum:
                    # 获取最大匹配长度
                    maxNum = record[i + 1][j + 1]
                    p = i + 1
                    q = i + i
                # 记录最大匹配长度的终止位置
    return str1[p - maxNum:p], maxNum


def delete_substr_method(in_str, in_substr):
    start_loc = in_str.find(in_substr)
    len_substr = len(in_substr)
    res_str = in_str[:start_loc] + in_str[start_loc + len_substr:]
    return res_str


def get_accuracy(str1, str2):
    le = max(len(str1),len(str2))
    same_str, num1 = get_num(str1, str2)
    print(same_str, num1)
    str1 = delete_substr_method(str1, same_str)
    str2 = delete_substr_method(str2, same_str)
    if len(str1) > 0 and len(str2) > 0:
        same_str, num2 = get_num(str1, str2)
        print(same_str, num2)
        return num1 + num2
    return num1 / le

a = '234567890'
s1 = '1234567890'
s2 = '1234567890'
print(get_accuracy(s1, s2))



str2 = [1,2,3,4,5]
str2 = [str(i) for i in str2]
str2 = ''.join(str2)
print(str2)