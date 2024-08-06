import requests

q = input("输入搜索内容：")

url = f'https://www.bing.com/search?q={q}'

my_headers = {
    "User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Mobile Safari/537.36 Edg/123.0.0.0"
}

resp = requests.get(url, headers=my_headers)

print(resp)
print(resp.text)  # 直接用有可能被检测到，可像上面加一个请求头。
