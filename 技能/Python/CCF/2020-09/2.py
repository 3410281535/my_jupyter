data = list(map(int, input().split()))
n = data[0]
k = data[1]
t = data[2]
xl = data[3]
yd = data[4]
xr = data[5]
yu = data[6]
passing = 0
hang = 0
link = [0] * n
for i in range(n):
    data = list(map(int, input().split()))
    l = []
    flag = True
    for j in range(1, t + 1):
        if xl <= data[2 * j - 2] <= xr and yd <= data[2 * j - 1] <= yu:
            if flag:
                if len(l) > 0:
                    l[-1] += 1
                else:
                    l.append(1)
            else:
                l.append(1)
                flag = True
        elif len(l) > 0:
            flag = False
    link[i] = max(l) if len(l) > 0 else 0
for i in range(n):
    if link[i] > 0:
        passing += 1
        if link[i] >= k:
            hang += 1
print(passing)
print(hang)