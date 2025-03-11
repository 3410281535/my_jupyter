grid = []
global level
for i in range(15):
    grid.append(list(map(int, input().split())))
app_grid = []
for i in range(4):
    app_grid.append(list(map(int, input().split())))
n = int(input())
level = 0
flag = True
while flag and not level == 15:
    for i in range(3, -1, -1):
        for j in range(3, -1, -1):
            if i + level < 15 and app_grid[i][j] + grid[i + level][j + n - 1] == 2:
                flag = False
                break
    level += 1
if not flag:
    level -= 2
else:
    for i in range(4):
        for j in range(4):
            if app_grid[i][j] == 1:
                level = 15 - i - 1
                break
    
for i in range(4):
    for j in range(4):
        if i + level < 15:
            grid[i + level][j + n - 1] += app_grid[i][j]
for i in range(15):
    for j in range(10):
        print(grid[i][j], end=' ')
    print()
        