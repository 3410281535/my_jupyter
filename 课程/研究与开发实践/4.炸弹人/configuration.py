import pygame
import random


# 参数
# 颜色
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
BROWN = (165, 42, 42)
GRAY = (190, 190, 190)

BACKGROUND_COLOR = (105, 139, 105)  # DarkSeaGreen4
# 屏幕
width = 950
height = 750


# 砖块类
class Brick:
    def __init__(self, left, top, brick_type):
        self.length = 50
        self.left = left
        self.top = top
        self.right = left + self.length
        self.bottom = top + self.length
        self.type = brick_type  # 0：空，1：砖（可打破），2：混凝土（不可打破）
        self.color = GREEN if self.type == 0 else (BROWN if self.type == 1 else GRAY)
        self.rect = (self.left, self.top, self.length, self.length)


# Mike类，方的
class Mike:
    def __init__(self, x, y):
        self.length = 30
        self.left = x
        self.top = y
        self.right = x + self.length
        self.bottom = self.top + self.length
        self.color = BLUE
        self.brick_mike_in_x = x % 50
        self.brick_mike_in_y = y % 50
        self.rect = pygame.Rect(self.left, self.top, self.length, self.length)


# 炸弹类
class Bomb:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = 30  # 爆炸半径
        self.active = True
        self.last_time = pygame.time.get_ticks()  # 放置炸弹的时间


# 怪物类
class Monster:
    def __init__(self, x, y):
        self.length = 20
        self.left = x
        self.top = y
        self.right = x + self.length
        self.bottom = self.top + self.length
        self.speed = 2  # 移动速度
        self.direction = random.randint(0, 3)  # 0: 左, 1: 上, 2: 右, 3: 下
        self.rect = pygame.Rect(self.left, self.top, self.length, self.length)

    def move(self, brick_grid):
        # 根据方向移动
        if self.direction == 0:
            self.left -= self.speed
        elif self.direction == 1:
            self.top -= self.speed
        elif self.direction == 2:
            self.left += self.speed
        elif self.direction == 3:
            self.top += self.speed

        # 更新位置和碰撞矩形
        self.rect = pygame.Rect(self.left, self.top, self.length, self.length)

        # 检测碰撞
        if self.check_collision(brick_grid):
            self.change_direction()

    def check_collision(self, brick_grid):
        # 获取怪物所在的格子
        i = self.top // 50
        j = self.left // 50
        max_row = 14  # 最大行
        max_col = 18  # 最大列

        # 根据方向检测相邻的砖块
        if self.direction == 0 and j > 0 and brick_grid[i * 19 + (j - 1)].type != 0:  # 左
            return True
        if self.direction == 1 and i > 0 and brick_grid[(i - 1) * 19 + j].type != 0:  # 上
            return True
        if self.direction == 2 and j < max_col and brick_grid[i * 19 + (j + 1)].type != 0:  # 右
            return True
        if self.direction == 3 and i < max_row and brick_grid[(i + 1) * 19 + j].type != 0:  # 下
            return True
        return False

    def change_direction(self):
        # 反方向移动
        if self.direction == 0:
            self.direction = 2
        elif self.direction == 1:
            self.direction = 3
        elif self.direction == 2:
            self.direction = 0
        elif self.direction == 3:
            self.direction = 1


# 道具类
class PowerUp:
    def __init__(self, x, y, shape):
        self.x = x
        self.y = y
        self.active = True
        self.shape = shape
        self.rect = pygame.Rect(self.x, self.y, 30, 30)


# 碰撞函数
def mike_brick_col(mike, brick_grid):
    # 根据 Mike 的位置计算 Mike 所在的格子
    i = mike.top // 50
    j = mike.left // 50

    # 初始化结果数组，4个方向，默认为可以移动（1）
    result = [1, 1, 1, 1]  # [left, up, right, down]

    # 获取 Mike 四个边的可能砖块坐标
    # 左方
    if mike.top + mike.length > (i + 1) * 50:
        left_bricks = [(i, j - 1), (i + 1, j - 1)]
    else:
        left_bricks = [(i, j - 1)]

    # 上方
    # print((mike.left + mike.length,(j + 1) * 50))
    if mike.left + mike.length > (j + 1) * 50:
        up_bricks = [(i - 1, j), (i - 1, j + 1)]
    else:
        up_bricks = [(i - 1, j)]

    # 右方
    if mike.top + mike.length > (i + 1) * 50:
        right_bricks = [(i, j + 1), (i + 1, j + 1)]
    else:
        right_bricks = [(i, j + 1)]

    # 下方
    if mike.left + mike.length > (j + 1) * 50:
        down_bricks = [(i + 1, j), (i + 1, j + 1)]
    else:
        down_bricks = [(i + 1, j)]


        # 定义行、列的最大值和最小值
    max_row = 14  # 最大行索引
    max_col = 18  # 最大列索引

    # 检测左方的砖块
    for bi, bj in left_bricks:
        if 0 <= bi <= max_row and 0 <= bj <= max_col:
            # print(brick_grid[bi * 19 + bj].type)
            if brick_grid[bi * 19 + bj].type == 0:
                result[0] = 1
            else:
                print((mike.left, brick_grid[bi * 19 + bj].right))
                if mike.left - brick_grid[bi * 19 + bj].right < 5:
                    result[0] = 0
                    break
                else:
                    result[0] = 1
    # 检测上方的砖块
    # print(up_bricks)
    for bi, bj in up_bricks:
        if 0 <= bi <= max_row and 0 <= bj <= max_col:
            if brick_grid[bi * 19 + bj].type == 0:
                result[1] = 1
            else:
                if mike.top - brick_grid[bi * 19 + bj].bottom < 5:
                    result[1] = 0
                    break
                else:
                    result[1] = 1
    # 检测右方的砖块
    for bi, bj in right_bricks:
        if 0 <= bi <= max_row and 0 <= bj <= max_col:
            if brick_grid[bi * 19 + bj].type == 0:
                result[2] = 1
            else:
                if brick_grid[bi * 19 + bj].left - (mike.left + mike.length) < 5:
                    result[2] = 0
                    break
                else:
                    result[2] = 1

    # 检测下方的砖块
    for bi, bj in down_bricks:
        if 0 <= bi <= max_row and 0 <= bj <= max_col:
            if brick_grid[bi * 19 + bj].type == 0:
                result[3] = 1
            else:
                if brick_grid[bi * 19 + bj].top - (mike.top + mike.length) < 5:
                    result[3] = 0
                    break
                else:
                    result[3] = 1

    # 砖块四角
    if mike.left == brick_grid[(i - 1) * 19 + (j - 1)].right - 1:
        if mike.top == brick_grid[(i - 1) * 19 + (j - 1)].bottom - 1 and brick_grid[(i - 1) * 19 + (j - 1)].type != 0:  # 左上
            result[0] = 0
            result[1] = 0
        if mike.bottom == brick_grid[(i + 1) * 19 + (j - 1)].top - 1 and brick_grid[(i + 1) * 19 + (j - 1)].type != 0:  # 左下
            result[0] = 0
            result[3] = 0
    if mike.right == brick_grid[(i - 1) * 19 + (j + 1)].left - 1:
        if mike.top == brick_grid[(i - 1) * 19 + (j + 1)].bottom - 1 and brick_grid[(i - 1) * 19 + (j + 1)].type != 0:  # 右上
            result[1] = 0
            result[2] = 0
        if mike.bottom == brick_grid[(i + 1) * 19 + (j + 1)].top - 1 and brick_grid[(i + 1) * 19 + (j + 1)].type != 0:  # 右下
            result[2] = 0
            result[3] = 0

    return result


def collision(mike, monster):
    return (
            (
                    (mike.left < monster.left + monster.length < mike.left + mike.length)
                    or
                    (mike.left + mike.length > monster.left > mike.left)
            )
            and
            (
                    (mike.top < monster.top + monster.length < mike.top + mike.length)
                    or
                    (mike.top + mike.length > monster.top > mike.top)
            )
    )


def bomb_collision(bomb, monster):
    bomb_center_x = bomb.x + 25
    bomb_center_y = bomb.y + 25
    return abs(monster.left - bomb_center_x) < bomb.radius and abs(monster.top - bomb_center_y) < bomb.radius


# 初始化怪物的函数，确保怪物在空砖块上出现且远离Mike,mike(1, 1)
def initialize_monsters(num_monsters, brick_grid):
    monsters = []
    possible_positions = []

    # 收集砖块类型为0的所有坐标
    for i in range(15):
        for j in range(19):
            if i <= 3 and j <= 3:
                continue
            elif brick_grid[i * 19 + j].type == 0:
                possible_positions.append((i, j))

    # 从收集到的位置中随机选择怪物起始点
    while len(monsters) < num_monsters and possible_positions:
        (i, j) = random.choice(possible_positions)
        monster = Monster(j * 50 + 15, i * 50 + 15)  # 怪物在空砖块的中心
        monsters.append(monster)
        possible_positions.remove((i, j))

    return monsters


# 炸弹爆炸逻辑
def explode_bomb(bomb, brick_grid, monsters, screen, power_up_type=None):
    if power_up_type == 1:
        # 方形道具：爆炸范围扩大
        explosion_rect = pygame.Rect(bomb.x - 100, bomb.y - 100, 200, 200)
    elif power_up_type == 2:
        # 十字形道具：十字爆炸
        explode_cross(bomb, brick_grid, monsters, screen)
        return
    else:
        # 默认爆炸范围
        explosion_rect = pygame.Rect(bomb.x - 50, bomb.y - 50, 100, 100)

    for brick in brick_grid:
        if brick.type == 1 and explosion_rect.colliderect(pygame.Rect(brick.rect)):
            brick.type = 0
            brick.color = GREEN

    for monster in monsters[:]:
        monster_rect = pygame.Rect(monster.left, monster.top, monster.length, monster.length)
        if explosion_rect.colliderect(monster_rect):
            monsters.remove(monster)
    pygame.draw.rect(screen, (255, 255, 0), explosion_rect)


# 十字形爆炸的辅助函数
def explode_cross(bomb, brick_grid, monsters, screen):
    explosion_segments = []
    i = bomb.y // 50
    j = bomb.x // 50

    # 向四个方向扩展
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上、下、左、右
    for di, dj in directions:
        step = 1
        while True:
            ni = i + di * step
            nj = j + dj * step
            if 0 <= ni < 15 and 0 <= nj < 19:
                brick = brick_grid[ni * 19 + nj]
                if brick.type == 2:  # 碰到混凝土块，停止
                    break
                elif brick.type == 1:  # 砖块，炸掉并停止
                    brick.type = 0
                    brick.color = GREEN
                else:
                    explosion_segments.append((nj * 50 + 25, ni * 50 + 25))
            else:
                break
            step += 1

    # 绘制十字爆炸
    for segment in explosion_segments:
        pygame.draw.circle(screen, (255, 255, 0), segment, 25)

    # 检查十字爆炸对怪物的影响
    for monster in monsters[:]:
        for segment in explosion_segments:
            if pygame.Rect(monster.left, monster.top, monster.length, monster.length).collidepoint(segment):
                monsters.remove(monster)
                break