# 10 1 10 20 30 20
n = int(input())
m = list(map(int, input().split()))
d = dict()
for i in range(n):
	if m[i] in d:
		d[m[i]] += 1
	else:
		d[m[i]] = 1
print(d)
d1 = dict(sorted(d.items(), key=lambda x: (-x[1], x[0])))
print(list(d1.keys())[0])