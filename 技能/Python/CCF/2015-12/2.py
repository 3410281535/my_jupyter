data = list(map(int, input().split()))
n = data[0]
m = data[1]
grid = []
for i in range(n):
    data = list(map(int, input().split()))
    grid.append(data)
flag = [[True for i in range(m)] for j in range(n)]
for i in range(n):
    for j in range(m):
        if not grid[i][j] == 0:
            if j + 2 < m and grid[i][j] == grid[i][j + 1] == grid[i][j + 2]:
                j0 = 3
                while j + j0 < m and grid[i][j] == grid[i][j + j0]:
                    j0 += 1
                for k in range(j, j + j0):
                    flag[i][k] = False
            if i + 2 < n and grid[i][j] == grid[i + 1][j] == grid[i + 2][j]:
                i0 = 3
                while i + i0 < n and grid[i][j] == grid[i + i0][j]:
                    i0 += 1
                for k in range(i, i + i0):
                    flag[k][j] = False
for i in range(n):
    for j in range(m):
        if flag[i][j]:
            print(grid[i][j], end=' ')
        else:
            print(0, end=' ')
    print()