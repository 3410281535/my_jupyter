n = int(input())
a = list(map(int, input().split()))
med = []
a.sort(key= lambda x:x)
# print(a)
dict = {}
dict_less = {}
dict_larger = {}
for i in range(n):
    if a[i] in dict:
        dict[a[i]] += 1
    else:
        dict[a[i]] = 1
temp = 0
num = 0
for i in range(n):
    if not a[i] == num:
        dict_less[a[i]] = temp
        temp += dict[a[i]]
        num = a[i]
# print(dict_less)
a_re = a[::-1]
# print(a_re)
temp = 0
num = 0
for i in range(n):
    if not a_re[i] == num:
        dict_larger[a_re[i]] = temp
        temp += dict[a_re[i]]
        num = a_re[i]
# print(dict_larger)
a0 = list(set(a))
# print(a0)
count = 0
for i in range(len(a0)):
    if dict_larger[a0[i]] == dict_less[a0[i]]:
        med.append(a0[i])
if len(med) > 0:
    for i in range(len(med)):
        print(med[i], end = " ")
else:
    print(-1)