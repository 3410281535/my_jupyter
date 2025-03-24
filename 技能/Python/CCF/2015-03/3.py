# 节日， 已AC
def run(y):
    return y % 400 == 0 or (y % 4 == 0 and not y % 100 == 0)

data = list(map(int, input().split()))
a = data[0]
b = data[1]
c = data[2]
y1 = data[3]
y2 = data[4]
week_day = 2
dict = {}
flag = True
for y in range(1850, 2051):
    flag = run(y)
    for m in range(1, 13):
        d = 31
        if m in [1, 3, 5, 7, 8, 10, 12]:
            d = 31
        elif m in [4, 6, 9, 11]:
            d = 30
        elif flag:
            d = 29
        else:
            d = 28
        if m < 10:
            m0 = f"0{m}"
        else:
            m0 = f"{m}"
        for d_ in range(1, d + 1):
            if d_ < 10:
                d0 = f"0{d_}"
            else:
                d0 = f"{d_}"
            date = str(y) + '/' + m0 + '/' + d0
            dict[date] = week_day
            if not week_day == 7:
                week_day += 1
            else:
                week_day = 1
if a < 10:
    a0 = f"0{a}"
else:
    a0 = f"{a}"

for y in range(y1, y2 + 1):
    date1 = 1
    date0 = str(y) + '/' + a0 + '/' + "01"
    week_day1 = dict[date0]
    if week_day1 > c:  # 本月第一天的星期几大于要求的c，要先补完本周
        while not week_day1 == 7:
            date1 += 1
            week_day1 += 1
        date1 += 7 * (b - 1) + c
    else:
        date1 += 7 * (b - 1) + c - week_day1
    flag = run(y)
    if a in [1, 3, 5, 7, 8, 10, 12]:
        d = 31
    elif a in [4, 6, 9, 11]:
        d = 30
    elif flag:
        d = 29
    else:
        d = 28
    
    if date1 <= d:
        if date1 < 10:
            date1_ = f"0{date1}"
        else:
            date1_ = f"{date1}"
        date0_ = str(y) + '/' + a0 + '/' + date1_
        print(date0_)
    else:
        print("none")