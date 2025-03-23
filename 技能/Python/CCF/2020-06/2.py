# 普通做法会超时，输入第一行“1000000000 120492 442944”
data = list(map(int, input().split()))
n = data[0]
a = data[1]
b = data[2]
dict1 = {}
s = 0
for i in range(a):
    data = list(map(int, input().split()))
    dict1[data[0]] = data[1]
for i in range(b):
    data = list(map(int, input().split()))
    if data[0] in dict1:
        dict1[data[0]] *= data[1]
        s += dict1[data[0]]
print(s)