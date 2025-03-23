S = input()
allergic = int(input())
n = int(input())
for i in range(n):
    s = input()
    if allergic == 1:
        if S in s:
            print(s)
    else:
        if S.lower() in s.lower():
            print(s)