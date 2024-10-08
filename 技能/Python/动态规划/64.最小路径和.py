def minPathSum(grid):
    m = len(grid)
    n = len(grid[0])
    for i in range(1, n):
        grid[0][i] = grid[0][i -1] + grid[0][i]
    for i in range(1, m):
        grid[i][0] = grid[i - 1][0] + grid[i][0]
    for i in range(1, m):
        for j in range(1, n):
            grid[i][j] = min(grid[i -1][j], grid[i][j - 1]) + grid[i][j]
    return grid[m - 1][n - 1]

# data = input().split()
# m = int(data[0])
# n = int(data[1])
# grid = [[0 for i in range(n)] for i in range(m)]
# for i in range(m):
#     data = input().split()
#     for j in range(n):
#         grid[i][j] = int(data[j])

# grid = [[1, 3, 1],
#         [1, 5, 1],
#         [4, 2, 1]]
grid = [[1,2,3],[4,5,6]]
print(minPathSum(grid))
