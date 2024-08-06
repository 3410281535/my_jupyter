from urllib.request import urlopen

"""url = "https://www.baidu.com"

resp = urlopen(url)

content = resp.read().decode("utf-8")
print(content)

with open("Baidu.html", mode="w", encoding="utf-8") as f:
    f.write(content)
"""



url = "http://www.cffex.com.cn/"
resp = urlopen(url)

content = resp.read().decode("utf-8")
print(content)
with open("China Financial Futures Exchange.html", mode="w", encoding="utf-8") as f:
    f.write(content)
