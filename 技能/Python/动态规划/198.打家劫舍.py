def rob(num):
    dp = [0] * L
    dp[0] = num[0]


    if L == 1:
        return dp[0]
    elif L == 2:
        dp[1] = max(num[0], num[1])
        return dp[1]
    else:
        dp[1] = max(num[0], num[1])
        for i in range(2, L):
            dp[i] = max(num[i] + dp[i - 2], dp[i - 1])
        return dp[L - 1]

data = input().split()
L = len(data)
num = [0] * L
for i in range(L):
    num[i] = int(data[i])

print(rob(num))