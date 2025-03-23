data = list(map(int, input().split()))
n = data[0]
N = data[1]
r = N // (n + 1)
a = [0]
temp = 0
num = [0]
data = list(map(int, input().split()))
count = [0] * N
error = 0
for i in range(len(data)):
    a.append(data[i])
    for i in range(data[i], N):
        count[i] += 1
    # count += N - data[i]
ind = 0
temp = 0
g = [0]
for i in range(1, N): 
    g.append(abs(count[i] - i // r))

print(sum(g))