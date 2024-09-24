def fibo(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        f = [0] * (n + 1)
        f[1] = 1
        for i in range(2, n + 1):
            f[i] = f[i - 1] + f[i - 2]
        return f[n]

n = int(input())
print(fibo(n))