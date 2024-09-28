def deleteAndEarn(nums):
    L = len(nums)
    dp = [0] * L
    for i in range(L):
        dp[i] = nums[i]

    for i in range(L):
        for value in nums:
            if (value == nums[i] - 1) or (value == nums[i] + 1):
                dp[i] -= value
    return dp

data = input().split()
nums = [0] * len(data)
for i in range(len(data)):
    nums[i] = int(data[i])

print(deleteAndEarn(nums))