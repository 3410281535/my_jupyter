n = int(input())
h = list(map(int, input().split()))
max_area = n
for i in range(n):
    max_area = max(max_area, h[i])
    k = 0  # 向右k格
    height = h[i]
    while i + k < n:
        height = min(height, h[i + k])
        max_area = max(max_area, (1 + k) * height)
        k += 1
print(max_area)