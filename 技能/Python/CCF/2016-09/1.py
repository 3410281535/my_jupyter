n = int(input())
a = list(map(int, input().split()))
b = 0
for i in range(n - 1):
    b = max(b, abs(a[i] - a[i + 1]))
print(b)