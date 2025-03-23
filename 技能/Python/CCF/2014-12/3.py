#wrong
import sys
records = []
price = 0
amount = 0
# ctrl+D stop
ca = []
for line in sys.stdin:
    record = line.split()
    records.append(record)
pop_list = []
for i in range(len(records)):
    if records[i][0] == 'cancel':
        # print(i + 1, int(records[i][1]))
        # print("--------")
        pop_list.append(i + 1)
        pop_list.append(int(records[i][1]))
pop_list = sorted(pop_list, reverse=True)
# print(pop_list)
# for i in range(len(records)):
#     print(records[i])
# print("---------")
for p in pop_list:
    records.pop(p - 1)
# for i in range(len(records)):
#     print(records[i])
# print("---------")
buy = {}
sell = {}
buy_price =[]
sell_price = []
for record in records:
    if record[0] == 'buy':
        if record[1] in buy:
            buy[f"{float(record[1]):.2f}"] += int(record[2])
        else:
            buy[f"{float(record[1]):.2f}"] = int(record[2])
            buy_price.append([round(float(record[1]), 2), int(record[2])])
    if record[0] == 'sell':
        if record[1] in sell:
            sell[f"{float(record[1]):.2f}"] += int(record[2])
        else:
            sell[f"{float(record[1]):.2f}"] = int(record[2])
            sell_price.append([round(float(record[1]), 2), int(record[2])])

dp_b = {}
dp_s = {}
buy_price = sorted(buy_price, reverse=True)
sell_price = sorted(sell_price, reverse=True)
buy_price1 = []
sell_price1 = []
for i in range(len(buy_price)):
    buy_price1.append(buy_price[i][0])
for i in range(len(sell_price)):
    sell_price1.append(sell_price[i][0])
buy_min = min(buy_price1)
buy_max = max(buy_price1)
sell_min = min(sell_price1)
sell_max = max(sell_price1)
price_list = sorted(list(set(buy_price1 + sell_price1)), reverse=True)
temp = 0
temp1 = 0
for p0 in price_list:
    if p0 in buy_price1:
        dp_b[f"{float(p0):.2f}"] = buy[f"{float(p0):.2f}"] + temp
        temp = dp_b[f"{float(p0):.2f}"]
price_list_re = sorted(price_list, reverse=False)
for p0 in price_list_re:
    if p0 in sell_price1:
        dp_s[f"{float(p0):.2f}"] = sell[f"{float(p0):.2f}"] + temp1
        temp1 = dp_s[f"{float(p0):.2f}"]

for p in price_list: # 价格一定会小于最大买价，大于最小卖价->有人买
    if buy_min <= p <= buy_max and sell_min <= p <= sell_max:
        temp_sell_price = 0
        temp_buy_price = 0
        for b in buy_price1:
            if p <= b:
                temp_buy_price = b
            else:
                break
        for s in sell_price1:
            if p <= s:
                temp_sell_price = s
            else:
                break
        amount_temp = min(dp_b[f"{float(temp_buy_price):.2f}"], dp_s[f"{float(temp_sell_price):.2f}"])
        if amount_temp > amount:
            price = p
            amount = amount_temp
    elif p < buy_min:  # 不会大于buy_max，不然没人买
        if p > sell_max: # 大于最大卖价，都卖
            amount_temp = min(dp_b[f"{buy_min:.2f}"], dp_s[f"{sell_max:.2f}"])
            if amount_temp > amount:
                price = p
                amount = amount_temp
        else:
            temp_sell_price = 0
            for s in sell_price1:
                if p <= s:
                    temp_sell_price = s
                else:
                    break
            amount_temp = min(dp_b[f"{buy_min:.2f}"], dp_s[f"{float(p):.2f}"])
            if amount_temp > amount:
                    price = p
                    amount = amount_temp
    elif p > sell_max:  # 不会小于sell_min，不然没人卖
        if p < buy_min:
                amount_temp = min(dp_b[f"{buy_min:.2f}"], dp_s[f"{sell_max:.2f}"])
                if amount_temp > amount:
                    price = p
                    amount = amount_temp
        else:
            temp_buy_price = 0
            for b in buy_price1:
                if p <= b:
                    temp_buy_price = b
                else:
                    break
            amount_temp = min(dp_b[f"{float(p):.2f}"], dp_s[f"{sell_max:.2f}"])
            if amount_temp > amount:
                price = p
                amount = amount_temp
    
# for i in range(len(records)):
#     print(records[i])

print(f"{float(price):.2f}", amount)
# print("---------")
# print(price_list)
# print(buy_price)
# print(sell_price)
# print(buy)
# print(sell)
# print(dp_b)
# print("---------")
# print(dp_s)