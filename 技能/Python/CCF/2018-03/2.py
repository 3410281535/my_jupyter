
data = list(map(int, input().split()))
n = data[0]  #小球个数
L = data[1]  #线段长度
t = data[2]  #时间
a = list(map(int, input().split()))
ball = []
for i in range(n):
    ball.append([a[i], 1, i])
    ball.sort(key=lambda x: x[0])  #按照位置排序
# 单个球的情况
if n == 1:
    for time in range(t):
        if ball[0][0] == 0 or ball[0][0] == L:
            ball[0][1] = -ball[0][1]
        ball[0][0] += ball[0][1]
else:
    for time in range(t):
        # 第一颗球
        if ball[0][0] == 0:
            ball[0][1] = -ball[0][1]
        elif ball[0][0] == ball[1][0]:
            ball[0][1] = -ball[0][1]
            ball[1][1] = -ball[1][1]
        # 中间的球    
        for i in range(1, n - 1):
            if ball[i][0] == ball[i + 1][0]:
                ball[i][1] = -ball[i][1]
                ball[i + 1][1] = -ball[i + 1][1]
        if ball[n - 1][0] == L:
            ball[n - 1][1] = -ball[n - 1][1]
        # 移动
        for i in range(n):
            ball[i][0] += ball[i][1]
ball.sort(key=lambda x: x[2])
for i in range(n):
    print(ball[i][0], end=' ')