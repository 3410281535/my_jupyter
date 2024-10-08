def uniquePathsWithObstacles(obstacleGrid) -> int:
    m = len(obstacleGrid)
    n = len(obstacleGrid[0])
    if m == 1 and n == 1:
        if obstacleGrid[0][0] == 0:
            return 1
        else:
            return 0
    else:
        for i in range(m):
            if m == 1:
                for j in range(n):
                    if obstacleGrid[i][j] == 1:
                        obstacleGrid[i][j] = 0
                    else:
                        if i == 0 and j == 0:
                            obstacleGrid[i][j] = 1
                        else:
                            obstacleGrid[i][j] = obstacleGrid[i][j - 1]
            elif n == 1:
                for j in range(n):
                    if obstacleGrid[i][j] == 1:
                        obstacleGrid[i][j] = 0
                    else:
                        if i == 0 and j == 0:
                            obstacleGrid[i][j] = 1
                        else:
                            obstacleGrid[i][j] = obstacleGrid[i - 1][j]
            else:
                for j in range(n):
                    if obstacleGrid[i][j] == 1:
                        obstacleGrid[i][j] = 0
                    elif i == 0 and j == 0:
                        obstacleGrid[i][j] = 1
                    elif i == 0:
                        obstacleGrid[i][j] = obstacleGrid[i][j - 1]
                    elif j == 0:
                        obstacleGrid[i][j] = obstacleGrid[i - 1][j]
                    else:
                        obstacleGrid[i][j] = obstacleGrid[i - 1][j] + obstacleGrid[i][j - 1]
        return obstacleGrid[m - 1][n - 1]


obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]
print(uniquePathsWithObstacles(obstacleGrid))
