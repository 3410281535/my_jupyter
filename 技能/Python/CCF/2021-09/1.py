# 数组推导
n = int(input())
B = list(map(int, input().split()))
temp_max = B[0]
sum_ = 0 + B[0]
# 最大
for i in range(1, len(B)):
    temp_max = B[i] if B[i] > temp_max else temp_max
    sum_ += temp_max
print(sum_)

sum_ = 0 + B[0]
temp_max = B[0]
# 最小
temp_min = 0
for i in range(1, len(B)):
    if B[i] > temp_max:
        sum_ += B[i]
        temp_max = B[i]
    else:
        sum_ += temp_min
print(sum_)