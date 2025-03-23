n = int(input())
a = []
b = []
c = []
d = []
for i in range(n):
    data = list(map(int, input().split()))
    a.append(data[0])
    b.append(data[1])
for i in range(n):
    data = list(map(int, input().split()))
    c.append(data[0])
    d.append(data[1])
max_time = max(max(b), max(d))
time_line = [0] * max_time
for i in range(n):
        for j in range(a[i], b[i]):
            time_line[j] += 1
        for j in range(c[i], d[i]):
            time_line[j] += 1
ind = 0
flag = True    
count = 0
for i in range(max_time):
    if time_line[i] > 1:
        count += 1

print(count)