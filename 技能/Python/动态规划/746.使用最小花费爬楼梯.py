def minCostClimbingStairs(cost):
    len1 = len(cost) + 1
    dp = [0] * len1
    dp[0] = 0
    dp[1] = 0
    dp[2] = min(cost[0], cost[1])
    for i in range(3, len1):
        dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2])
    return dp[len1 - 1]


data = input().split()
cost = [0] * len(data)
for i in range(len(data)):
    cost[i] = int(data[i])
print(minCostClimbingStairs(cost))