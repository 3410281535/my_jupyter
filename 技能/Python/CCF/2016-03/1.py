n = int(input())
a = list(map(int, input().split()))
count = 0
for i in range(1, n - 1):
    if (a[i - 1] < a[i] and a[i + 1] < a[i]) or (a[i - 1] > a[i] and a[i + 1] > a[i]):
        count += 1
print(count)