import sys
records = []
price = 0
amount = 0
# ctrl+D stop
for line in sys.stdin:
    record = line.split()
    if record[0] == 'cancel':
        records.pop(int(record[1]))
    else:
        records.append(record)

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

dp = {}
buy_price = sorted(buy_price, reverse=True)
sell_price = sorted(sell_price, reverse=True)
price_list = sorted(list(set(buy_price + sell_price)), reverse=True)


print(price_list)
print(buy_price)
print(sell_price)
print(buy)
print(sell)
# print(price, amount)