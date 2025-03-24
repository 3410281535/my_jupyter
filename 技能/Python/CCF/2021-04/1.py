# 灰度直方图
data = list(map(int, input().split()))
n = data[0]
m = data[1]
L = data[2]
A = []
L0 = {}
for i in range(L):
    if not i in L0:
        L0[i] = 0
for i in range(n):
    A.append(list(map(int, input().split())))
for i in range(n):
    for j in range(m):
        L0[A[i][j]] += 1
for i in range(L):
    print(L0[i], end=" ")