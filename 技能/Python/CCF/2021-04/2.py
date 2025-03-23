data = list(map(int, input().split()))
n = data[0]
L = data[1]
r = data[2]
t = data[3]
a = []
count = 0
for i in range(n):
    data = list(map(int, input().split()))
    a.append(data)
pre_sum = [[0 for i in range(n)] for j in range(n)]
for i in range(n):
    for j in range(n):
        pre_sum[i][j] = a[i][j]
        if i > 0:
            pre_sum[i][j] += pre_sum[i - 1][j]
        if j > 0:
            pre_sum[i][j] += pre_sum[i][j - 1]
        if i > 0 and j > 0:
            pre_sum[i][j] -= pre_sum[i - 1][j - 1]

for i in range(n):
    for j in range(n):
        w_min = max(0, j - r)
        w_max = min(n - 1, j + r)
        h_min = max(0, i - r)
        h_max = min(n - 1, i + r)
        area = (w_max - w_min + 1) * (h_max - h_min + 1)
        part1 = pre_sum[h_max][w_max]
        part2 = pre_sum[h_max][w_min - 1] if j - r > 0 else 0
        part3 = pre_sum[h_min - 1][w_max] if i - r > 0 else 0
        part4 = pre_sum[h_min - 1][w_min - 1] if i - r > 0 and j - r > 0 else 0
        sum = part1 - part2 - part3 + part4
        if sum <= t * area:
            count += 1
print(count)