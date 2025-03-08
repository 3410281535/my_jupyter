n = int(input())
m = list(map(int, input().split()))
count = 1
for i in range(n - 1):
	if m[i] != m[i + 1]:
		count += 1
print(count)