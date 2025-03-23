def OP(a, b, op):
    if op == '+':
        return a + b
    elif op == '-':
        return a - b
    elif op == 'x':
        return a * b
    elif op == '/':
        return a // b
    

n = int(input())
num_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
op_list = ['+', '-', 'x', '/']
for i in range(n):
    data = []
    op = []
    d = input()
    num = []
    for i in range(len(d)):
        data.append(d[i])
    j = 0
    a = 0
    b = 0
    c = 0
    d = 0
    while data[j] in num_list:
        a = 10 * a + int(data[j])
        j += 1
    num.append(a)
    op.append(data[j])
    j += 1
    while data[j] in num_list:
        b = 10 * b + int(data[j])
        j += 1
    num.append(b)
    op.append(data[j])
    j += 1
    while data[j] in num_list:
        c = 10 * c + int(data[j])
        j += 1
    num.append(c)
    op.append(data[j])
    j += 1  
    while data[j] in num_list:
        d = 10 * d + int(data[j])
        if not j == len(data) - 1:  
            j += 1
        else:
            break
    num.append(d)
    while not len(op) == 0:
        # 运算所有'x'、'/'
        while 'x' in op or '/' in op:
            for k in range(len(op)):
                if op[k] == 'x' or op[k] == '/':
                    num[k] = OP(num[k], num[k + 1], op[k])
                    num.pop(k + 1)
                    op.pop(k)
                    break
        if not len(op) == 0:
            num[0] = OP(num[0], num[1], op[0])
            num.pop(1)
            op.pop(0)
    if num[0] == 24:
        print('Yes')
    else:
        print('No')