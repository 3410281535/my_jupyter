# 暴力算法会超时
# 非零段划分
n = int(input())
A0 = list(map(int, input().split()))
A0.insert(0, 0)
A0.append(0)
A = [0] 
temp = 0
for i in range(1, len(A0)):
    if not temp == A0[i]:
        A.append(A0[i])
        temp = A0[i]
a_list = sorted(list(set(A)))
dict = {}
for a_l in a_list:
    dict[a_l] = 0
count = 0
if sum(A) == 0:
    print(0)
else:
    max_d = 0
    for i in range(1, len(A) - 1):
        if not A[i] == 0 and A[i + 1] == 0:
            max_d += 1
        if A[i - 1] < A[i] and A[i + 1] < A[i]:
            dict[A[i]] -= 1
        if A[i - 1] > A[i] and A[i + 1] > A[i]:
            dict[A[i]] += 1
    # print(dict)
    for i in range(1, len(a_list)):
        a0 = a_list[i]
        a1 = a_list[i - 1]
        dict[a0] += dict[a1]
    print(1 + max(dict.values()))
    