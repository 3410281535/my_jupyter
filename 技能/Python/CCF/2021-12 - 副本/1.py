data = list(map(int, input().split()))
n = data[0]
N = data[1]
a = [0]
temp = 0
num = [0]
data = list(map(int, input().split()))
count = 0
for i in range(len(data)):
    count += N - data[i]
print(count)