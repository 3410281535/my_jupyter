n = int(input())
m = int(input())
p = []
q = []
for i in range(m):
    data = input().split()
    p.append(int(data[0]))
    q.append(int(data[1]))
stu = []
flag = []
for i in range(n):
    stu.append(i)
    flag.append(False)  # 位置
for i in range(m):
    temp = stu[p[i] - 1]
    stu[p[i] - 1] += q[i]
    if q[i] >= 0:
        for i2 in range(n):
            if temp < stu[i2] <= stu[p[i] - 1] and not i2 == p[i] - 1:
                flag[i2] = True
        for i1 in range(n):
            if flag[i1]:
                stu[i1] -= 1
                flag[i1] = False
    else:
        for i2 in range(n):
            if stu[p[i] - 1] <= stu[i2] < temp and not i2 == p[i] - 1:
                flag[i2] = True
        for i1 in range(n):
            if flag[i1]:
                stu[i1] += 1
                flag[i1] = False
for i in range(n):
    for i1 in range(n):
        if i == stu[i1]:
            print(i1 + 1, end=' ')
            break
"""
# AI
def move_student(queue, p, q):
    # 找到学号为 p 的学生在队列中的位置
    index = queue.index(p)
    # 移除学生
    queue.pop(index)
    # 计算新的位置
    new_index = index + q
    # 插入学生到新位置
    queue.insert(new_index, p)

def main():
    n = int(input())
    m = int(input())
    queue = list(range(1, n+1))
    for _ in range(m):
        p, q = map(int, input().split())
        move_student(queue, p, q)
    print(' '.join(map(str, queue)))

if __name__ == "__main__":
    main()
"""