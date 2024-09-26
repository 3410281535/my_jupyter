def path(m, n):
    # if m == 0 and n == 0:
    #     return 0
    # if m == 0:
    #     return 1
    # if n == 0:
    #     return 1
    # else:
    #     return path(m - 1, n) + path(m, n - 1)
    dp = [[0] * n for _ in range(m)]
    for i in range(n):
        dp[0][i] = 1
    for i in range(m):
        dp[i][0] = 1
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    print(dp[m - 1][n - 1])


m = int(input())
n = int(input())
path(m, n)