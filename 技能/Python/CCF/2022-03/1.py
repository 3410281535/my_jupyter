data = list(map(int, input().split()))
n = data[0]
k = data[1]
x = {0: 1}
count = 0
for i in range(k):
    data = list(map(int, input().split()))
    if data[1] not in x:
        count += 1
    else:
        x[data[1]] = 1
    x[data[0]] = 1
print(count)
