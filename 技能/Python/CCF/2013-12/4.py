# n = int(input())
# count = 0
# max_num = ['2']
# for i in range(n - 3):
#     max_num.append('3')
# max_num.append('0')
# max_num.append('1')
# max_num = str(int(''.join(max_num)))
# int_max_num = int(max_num)
# min_num = ['2']
# for i in range(n - 3):
#     min_num.append('0')
# min_num.append('1')
# min_num.append('3')
# min_num = str(int(''.join(min_num)))
# int_min_num = int(min_num)
# for i in range(int_min_num, int_max_num + 1):
#     if '4' in str(i) or '5' in str(i) or '6' in str(i) or '7' in str(i) or '8' in str(i) or '9' in str(i):
#         continue
#     else:
#         if '0' in str(i) and '1' in str(i) and '2' in str(i) and '3' in str(i):
#             if str(i).rindex('2') < str(i).index('3') and str(i).rindex('0') < str(i).index('1'):
#                 count += 1
# print(count % 1000000007)

# 方法超时

# # 优化
n = int(input())
MOD = 10**9 + 7
# 帕斯卡三角形计算组合数
c = [[0 for i in range(1001)] for _ in range(1001)]  # n<=1000
for i in range(0, 1001):
    c[i][0] = 1
    for j in range(1, i+1):
        c[i][j] = (c[i-1][j-1] + c[i-1][j]) % MOD
num = 0
for k in range(2, n-1):  # 连加
    num += c[n-1][k]*(k-1)*(n-k-1)  # c[][]为组合数
print(num % MOD)
#
# 有误的代码，样例77时错误，结果未取模导致溢出？
# def com(a, b):
#     result1 = 1
#     for i in range(2, a + 1):
#         result1 *= i
#     result2 = 1
#     for i in range(2, b + 1):
#         result2 *= i
#         result2 %= 1000000007
#     result3 = 1
#     for i in range(2, b - a + 1):
#         result3 *= i
#     return result2 / (result1 * result3)
# n = int(input())
# MOD = 10 ** 9 + 7
# count = 0
# for k in range(2, n - 1):
#     count += (com(k, n - 1) % MOD) * (k - 1) * (n - k - 1)
# print(int(count % 1000000007))