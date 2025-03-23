N = int(input())
m = []
a = []
Drop = [False] * N
T = 0
D = 0
for i in range(N):
    data = list(map(int, input().split()))
    m.append(data[0])
    data.pop(0)
    a.append(data)
for i in range(N):
    for j in range(1, m[i]):
        if a[i][j] > 0:
            if not a[i][j] == a[i][0]:
                if not Drop[i]:
                    D += 1
                    Drop[i] = True
            a[i][0] = a[i][j]
        else:
            a[i][0] += a[i][j]
    T += a[i][0]
E = 0
if Drop[N - 2] and Drop[N - 1] and Drop[0]:
    E += 1
if Drop[N - 1] and Drop[0] and Drop[1]:
    E += 1
for i in range(N - 2):
    if Drop[i] and Drop[i + 1] and Drop[i + 2]:
        E += 1
print(T, D, E)