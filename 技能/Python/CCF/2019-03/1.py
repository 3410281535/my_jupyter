# 小中大
n = int(input())
a = list(map(int, input().split()))
max_ = max(a)
sorted(a)
med_ = (a[len(a) // 2] + a[(len(a) - 1) // 2]) / 2
min_ = min(a)
if med_ % 1 == 0:
    med_ = int(med_)
else:
    med_ = round(med_, 1)
print(max_, med_, min_)