data = list(map(int, input().split()))
r = data[0]
y = data[1]
g = data[2]
T = r + y + g
n = int(input())
k = []
t = []
for i in range(n):
    data = list(map(int, input().split()))
    k.append(data[0])
    t.append(data[1])
time = 0
for i in range(n):
    if k[i] == 0:
        time += t[i]
    else:
        temp_time = time % T
        if k[i] == 1:
            if temp_time <= t[i]:
                time += t[i] - temp_time
            elif temp_time <= t[i] + g:
                time += 0
            elif temp_time <= t[i] + g + y:
                time += T - temp_time + t[i]
            else:
                time += T - temp_time + t[i]
        elif k[i] == 2:
            if temp_time <= t[i]:
                time += t[i] - temp_time + r
            elif temp_time <= t[i] + r:
                time += t[i] + r - temp_time
            elif temp_time <= t[i] + r + g:
                time += 0
            else:
                time += T - temp_time + t[i] + r
        else:
            if temp_time < t[i]:
                time += 0
            elif temp_time <= t[i] + y:
                time += t[i] + y + r - temp_time
            elif temp_time <= t[i] + y + r:
                time += t[i] + y + r - temp_time
print(time)