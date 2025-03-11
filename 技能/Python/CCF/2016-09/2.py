n = int(input())
p = list(map(int, input().split()))
max_row = 5
seat = []
for i in range(20):
    seat.append([0, max_row])
for i in range(n):
    flag = False
    for j in range(20):
        if p[i] <= seat[j][1]:
            flag = True
            temp = seat[j][0]
            seat[j][0] += p[i]
            seat[j][1] -= p[i]
            for i1 in range(p[i]):
                print(5 * j + i1 + 1 + temp, end=' ')
            print()
            break
    if not flag:
        for j in range(20):
            if seat[j][1] > 0:
                if p[i] > seat[j][1]:
                    temp = seat[j][0]
                    p[i] -= seat[j][1]
                    for i1 in range(seat[j][1]):
                        print(5 * j + i1 + 1 + temp, end=' ')
                    seat[j][0] += seat[j][1]
                    seat[j][1] = 0

                    
                else:
                    temp = seat[j][0]
                    seat[j][0] += p[i]
                    seat[j][1] -= p[i]
                    for i1 in range(p[i]):
                        print(5 * j + i1 + 1 + temp, end=' ')
                    p[i] -= 0
                    break
        print()
