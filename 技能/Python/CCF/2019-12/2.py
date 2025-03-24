def get_0123(s):
    s_0 = [s[0], s[1] + 1]
    s_1 = [s[0], s[1] - 1]
    s_2 = [s[0] + 1, s[1]]
    s_3 = [s[0] - 1, s[1]]
    return s_0, s_1, s_2, s_3

score = [0] * 5
sta = []
# 回收站选址
n = int(input())
S = []
for i in range(n):
    S.append(list(map(int, input().split())))

# 找回收站
for s in S:
    for s_ in get_0123(s):  # 目标选址地
        if s_ in S:
            s__0, s__1, s__2, s__3 = get_0123(s_)
            if s__0 in S and s__1 in S and s__2 in S and s__3 in S and not s_ in sta:
                sta.append(s_)
                c = 0
                # 评分
                if [s_[0] + 1, s_[1] + 1] in S:
                    c += 1
                if [s_[0] + 1, s_[1] - 1] in S:
                    c += 1
                if [s_[0] - 1, s_[1] + 1] in S:
                    c += 1
                if [s_[0] - 1, s_[1] - 1] in S:
                    c += 1
                score[c] += 1
for sc in score:
    print(sc)
                