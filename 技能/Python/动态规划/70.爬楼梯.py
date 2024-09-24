"""
假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
"""

def stair(n):
    dp = [1] * (n + 1)
    for i in range(n + 1):
        if i == 2:
            dp[i] = 2
        if i > 2:
            dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]

n = int(input())
print(stair(n))