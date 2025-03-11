y = int(input())
d = int(input())
m = [31,28,31,30,31,30,31,31,30,31,30,31]
if y % 4 == 0 and y % 100 != 0 or y % 400 == 0:
    m[1] = 29
for i in range(12):
    if d > m[i]:
        d -= m[i]
    else:
        print(i+1)
        print(d)
        break 