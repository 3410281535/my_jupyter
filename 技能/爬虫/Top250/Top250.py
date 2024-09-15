import requests
import re
import openpyxl
import matplotlib.pyplot as plt
import pandas as pd
from bs4 import BeautifulSoup
import os

directory = "./img"
if not os.path.exists(directory):
    os.makedirs(directory)
headers = {
    "User-Agent": "Mozilla/5.0(Windows NT 10.0;Win64;x64)"
}
k = 0
data_list = []

workbook = openpyxl.Workbook()
sheet = workbook.active
sheet.append(['电影名称', '导演', '主演', '年份', '电影类型', '评分', '评语'])
for start_num in range(0, 250, 25):
    response = requests.get(f"https://movie.douban.com/top250?start={start_num}", headers=headers)
    html = response.text
    soup = BeautifulSoup(html, "html.parser")

    # 图片
    pics = soup.findAll('div', 'pic')
    links = []
    for pic in pics:
        links.append(pic.find('img').attrs['src'])
    # for link in links:print(link)
    for i in range(len(links)):
        path = './img/{}.jpg'.format(i+k+1)
        res = requests.get(links[i])
        with open(path, 'wb') as f:
            f.write(res.content)
        # print('{}.jpg has been saved'.format(i+k+1))
    k = k+25

    for item in soup.find_all('div', class_='item'):
        data = []
        data.append(item.find_all_next(class_='title')[0].get_text())
        data.append(item.find_all_next("p", class_='')[0].get_text())
        data.append(item.find_all_next("span", "rating_num")[0].get_text())
        data.append(item.find_all(class_='inq')[0].get_text() if len(item.find_all(class_='inq')) == 1 else "")
        data_list.append(data)
        # print(data)
    # print(data)
for i in range(10):
    for j in range(25):
        # 电影名称
        movie_name = data_list[j][0]

        # 导演和主演信息的正则表达式模式
        director_actor_pattern = r'导演:\s(.*?)\s+主演:\s(.*?)\s+\d{4}\s/\s([\w\s]+)'

        year_genre_pattern = r'(\d{4})[\s\xa0/]+([\w\s]+)[\s\xa0/]+([\w\s]+)'

        # 评分
        rating = data_list[j][2]

        # 评语
        comment = data_list[j][3]

        # 使用正则表达式匹配导演和主演信息
        match = re.search(director_actor_pattern, data_list[j][1])

        if match:
            director = match.group(1)
            actors = match.group(2)
        else:
            director = "未知导演"
            actors = "未知主演"
        year_genre_match = re.search(year_genre_pattern, data_list[j][1])

        if match:
            year = year_genre_match.group(1)
            genre = year_genre_match.group(3)
        sheet.append([movie_name, director, actors, year, genre, rating, comment])

workbook.save('电影信息.xlsx')

df = pd.read_excel(r"D:\Python\Programs\电影信息.xlsx")
print(df.columns)
# 将 "年份" 列转换为字符串
df['年份'] = df['年份'].astype(str)

# 使用正则表达式从电影信息中提取年份并创建一个新的列
df['年份'] = df['年份'].str.extract(r'(\d{4})')

# 按年份统计电影数量
year_counts = df['年份'].value_counts().sort_index()

# 创建柱状图
plt.figure(figsize=(12, 6))
year_counts.plot(kind='bar', color='skyblue')
plt.xlabel('Year')
plt.ylabel('Movie_Num')
plt.title('Number of movies per year')
plt.xticks(rotation=45)
plt.tight_layout()

# 显示图表
plt.show()
