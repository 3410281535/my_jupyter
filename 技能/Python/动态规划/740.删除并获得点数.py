def deleteAndEarn(nums):
    L = len(nums)
    dp = [0] * L
    dp[0] = nums[0] + nums[1]
    if L == 1:
        return dp[0]
    elif L == 2:
        return dp[0]
    else:
        dp[L - 1] = nums[L - 2] + nums[L - 1]
        for i in range(1, L - 2):
            dp[i] = nums[i - 1] + nums[i] + nums[i + 1]
        return max(dp)


data = input().split()
nums = [0] * len(data)
for i in range(len(data)):
    nums[i] = int(data[i])

print(deleteAndEarn(nums))