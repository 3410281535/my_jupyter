data = input().split()
n = int(data[0])
k = int(data[1])
key = []
for i in range(n):
    key.append(i + 1)
wsc = []
time_max = 0
for i in range(k):
    data = list(map(int, input().split()))
    data[2] += data[1]
    wsc.append(data)
wsc.sort(key=lambda x: (x[1], x[0]))
# wsc:钥匙编号，开始时间，结束时间
time_line = []
for i in range(k):
    time_line.append([wsc[i][1], wsc[i][0], 0])
    time_line.append([wsc[i][2], wsc[i][0], 1])
time_line.sort(key=lambda x: (x[0], -x[2], x[1]))
# time_line:时间，钥匙编号，0取钥匙，1还钥匙
for i in range(len(time_line)): 
    if time_line[i][2] == 0:  # 取钥匙
        for j in range(len(key)):
            if key[j] == time_line[i][1]:
                key[j] = 'X'
                break
    else:
        for j in range(len(key)):
            if key[j] == 'X':
                key[j] = time_line[i][1]
                break
for k in key:
    print(k, end=' ')