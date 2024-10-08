"""
下方为等级，条每次升级可获得不同效果（子弹增大、伤害增加，副武器等）
小怪出现完后boss出现，请尽快升级
boss二阶段伤害增加
"""

import math
import time
import pygame
import random
import sys


# 参数
# 设置屏幕大小
width = 500
height = 800

# 定义颜色
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 97, 0)
PURPLE = (162, 32, 240)

# 经验条
C = [WHITE, GREEN, YELLOW, ORANGE, BLUE, PURPLE]

# 敌机和子弹的最大数量
MAX_ENEMY = 50
MAX_BULLET = 100

# 敌机数量
enemy_num = 0
# 飞机长宽
my_plane_width = 50
my_plane_height = 70
enemy_plane_width = 30
enemy_plane_height = 50

# 飞机血量
playerHealth = 100

# 飞机等级
level = 0
# 击杀数量
kill_num = 0

# 初始化pygame
pygame.init()

screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("飞机大战")


# 定义敌机和子弹
class Bullet:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# 飞机
class Plane:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.width = w
        self.height = h
        self.radius = 5
        self.bullet_active = [False] * MAX_BULLET
        self.bullet = [Bullet(x + w / 2, y + h + 10) for _ in range(MAX_BULLET)]  # bullet位置后续要更新
        self.active = False
        self.hurt = 5
        self.left_bullet_active = [False] * MAX_BULLET
        self.left_bullet = [Bullet(x + w / 2, y + h + 10) for _ in range(MAX_BULLET)]  # bullet位置后续要更新
        self.right_bullet_active = [False] * MAX_BULLET
        self.right_bullet = [Bullet(x + w / 2, y + h + 10) for _ in range(MAX_BULLET)]  # bullet位置后续要更新


class BOSS:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.blood = 5000
        self.radius = 5
        self.radius0 = 10
        self.radius1 = 20
        self.radius2 = 50
        self.radius3 = 100
        self.shoot_interval = 1000  # 5秒发射一次
        self.angle = 0
        self.bullet_angle = [[0 for _ in range(5)] for _ in range(18)]
        self.bullet = [[Bullet(x, y) for _ in range(5)] for _ in range(18)]
        self.bullet_active = [[False for _ in range(5)] for _ in range(18)]
        self.flag = False

    def draw(self):
        # 外框
        pygame.draw.circle(screen, WHITE, (self.x, self.y), self.radius3)
        # 瞳孔
        pygame.draw.circle(screen, BLACK, (self.x, self.y), self.radius2)
        pupil_color = RED if self.blood < 2500 else WHITE

        pygame.draw.circle(screen, pupil_color, (self.x, self.y), self.radius1)
        pygame.draw.rect(screen, BLACK, (width / 2 - 10, 0, 20, 200))
        # 核心
        pygame.draw.circle(screen, pupil_color, (width / 2, 100), self.radius0)

    def shoot(self):
        for i in range(18):
            self.angle = random.randint(0, 29)
            for j in range(5):
                if not boss.bullet_active[i][j]:
                    boss.bullet_angle[i][j] = math.radians((self.angle + i * 20))
                    bullet_x = self.x + math.cos(boss.bullet_angle[i][j]) * (self.radius3 + 20)
                    bullet_y = self.y + math.sin(boss.bullet_angle[i][j]) * (self.radius3 + 20)
                    boss.bullet[i][j].x = bullet_x
                    boss.bullet[i][j].y = bullet_y
                    boss.bullet_active[i][j] = True
                    break



    def update_position(self):
        for i in range(18):
            angle = math.radians((self.angle + i * 20))
            for j in range(5):
                if boss.bullet_active[i][j]:
                    boss.bullet[i][j].x += math.cos(angle) * 2
                    boss.bullet[i][j].y += math.sin(angle) * 2

    def check_collision(self):
        global playerHealth
        # boss子弹打我机
        for i in range(18):
            for j in range(5):
                if boss.bullet_active[i][j]:
                    if my_plane.x < boss.bullet[i][j].x < my_plane.x + my_plane_width and my_plane.y < boss.bullet[i][j].y < my_plane.y + my_plane_height:
                        boss.bullet_active[i][j] = False
                        if boss.blood <= 2500:
                            playerHealth -= 2 * my_plane.hurt
                        else:
                            playerHealth -= my_plane.hurt

        # 我机碰boss
        if my_plane.y <= 223:
            playerHealth -= 10
            my_plane.y += 200

        # 我机子弹碰boss
        for i in range(MAX_BULLET):
            if my_plane.bullet_active[i]:
                bullet_x = my_plane.bullet[i].x
                bullet_y = my_plane.bullet[i].y
                distance_to_boss = math.sqrt((bullet_x - boss.x) ** 2 + (bullet_y - boss.y) ** 2)

                # 打核心
                if width / 2 - boss.radius0 < my_plane.bullet[i].x - my_plane.radius and my_plane.bullet[i].x + my_plane.radius < width / 2 + boss.radius0:
                    if my_plane.bullet[i].y + my_plane.radius < 100 + boss.radius0:
                        my_plane.bullet_active[i] = False
                        if 2 <= level <= 4:
                            self.blood -= 2 * (my_plane.hurt + level - 2)
                        elif level == 5:
                            self.blood -= 2 * 2 * my_plane.hurt
                        else:
                            self.blood -= 2 * my_plane.hurt
                elif distance_to_boss < self.radius3:  # Hitting the outer shell
                    my_plane.bullet_active[i] = False
                    if 2 <= level <= 4:
                        self.blood -= my_plane.hurt + level - 2
                    elif level == 5:
                        self.blood -= 2 * my_plane.hurt
                    else:
                        self.blood -= my_plane.hurt
        if 3 <= level <= 4:
            for i in range(MAX_BULLET):
                if my_plane.left_bullet_active[i]:
                    bullet_x = my_plane.left_bullet[i].x
                    bullet_y = my_plane.left_bullet[i].y
                    distance_to_boss = math.sqrt((bullet_x - boss.x) ** 2 + (bullet_y - boss.y) ** 2)

                    # 打核心
                    if width / 2 - boss.radius0 < my_plane.left_bullet[i].x - my_plane.radius and my_plane.left_bullet[
                        i].x + my_plane.radius < width / 2 + boss.radius0:
                        if my_plane.left_bullet[i].y + my_plane.radius < 100 + boss.radius0:
                            my_plane.left_bullet_active[i] = False
                            self.blood -= 2 * my_plane.hurt
                    elif distance_to_boss < self.radius3:  # Hitting the outer shell
                        my_plane.left_bullet_active[i] = False
                        self.blood -= my_plane.hurt
            for i in range(MAX_BULLET):
                if my_plane.right_bullet_active[i]:
                    bullet_x = my_plane.right_bullet[i].x
                    bullet_y = my_plane.right_bullet[i].y
                    distance_to_boss = math.sqrt((bullet_x - boss.x) ** 2 + (bullet_y - boss.y) ** 2)

                    # 打核心
                    if width / 2 - boss.radius0 < my_plane.right_bullet[i].x - my_plane.radius and my_plane.right_bullet[
                        i].x + my_plane.radius < width / 2 + boss.radius0:
                        if my_plane.right_bullet[i].y + my_plane.radius < 100 + boss.radius0:
                            my_plane.right_bullet_active[i] = False
                            self.blood -= 2 * my_plane.hurt
                    elif distance_to_boss < self.radius3:  # Hitting the outer shell
                        my_plane.right_bullet_active[i] = False
                        self.blood -= my_plane.hurt


def extra_bullets():
    for i in range(MAX_BULLET):
        # 左侧子弹
        if not my_plane.left_bullet_active[i]:
            my_plane.left_bullet[i] = Bullet(my_plane.x, my_plane.y - 10)
            my_plane.left_bullet_active[i] = True
            break

    for i in range(MAX_BULLET):
        # 右侧子弹
        if not my_plane.right_bullet_active[i]:
            my_plane.right_bullet[i] = Bullet(my_plane.x + my_plane.width, my_plane.y - 10)
            my_plane.right_bullet_active[i] = True
            break


def extra_collision_left():
    global kill_num
    for i in range(MAX_ENEMY):
        if enemy_planes[i].active:
            for j in range(MAX_BULLET):
                if my_plane.left_bullet_active[j]:
                    if enemy_planes[i].x <= my_plane.left_bullet[j].x + my_plane.radius and enemy_planes[i].x + enemy_plane_width > my_plane.left_bullet[j].x - my_plane.radius:
                        if my_plane.left_bullet[j].y < enemy_planes[i].y + enemy_plane_height:
                            kill_num += 1
                            enemy_planes[i].active = False
                            my_plane.left_bullet_active[j] = False
                            break


def extra_collision_right():
    global kill_num
    for i in range(MAX_ENEMY):
        if enemy_planes[i].active:
            for j in range(MAX_BULLET):
                if my_plane.right_bullet_active[j]:
                    if enemy_planes[i].x <= my_plane.right_bullet[j].x + my_plane.radius and enemy_planes[i].x + enemy_plane_width > my_plane.right_bullet[j].x - my_plane.radius:
                        if my_plane.right_bullet[j].y < enemy_planes[i].y + enemy_plane_height:
                            kill_num += 1
                            enemy_planes[i].active = False
                            my_plane.right_bullet_active[j] = False
                            break


def update_positions_left():
    for i in range(MAX_BULLET):
        if my_plane.left_bullet_active[i]:
            my_plane.left_bullet[i].y -= 5


def update_positions_right():
    for i in range(MAX_BULLET):
        if my_plane.right_bullet_active[i]:
            my_plane.right_bullet[i].y -= 5


def draw_left():
    global C
    for i in range(MAX_BULLET):
        if my_plane.left_bullet_active[i]:
            pygame.draw.circle(screen, C[level], (int(my_plane.left_bullet[i].x), int(my_plane.left_bullet[i].y)), my_plane.radius)
            if my_plane.left_bullet[i].y < 0:
                my_plane.left_bullet_active[i] = False


def draw_right():
    global C
    for i in range(MAX_BULLET):
        if my_plane.right_bullet_active[i]:
            pygame.draw.circle(screen, C[level], (int(my_plane.right_bullet[i].x), int(my_plane.right_bullet[i].y)), my_plane.radius)
            if my_plane.right_bullet[i].y < 0:
                my_plane.right_bullet_active[i] = False


my_plane = Plane(width / 2 - my_plane_width / 2, 740, my_plane_width, my_plane_height)
my_plane.active = True

enemy_planes = [Plane(0, 0, enemy_plane_width, enemy_plane_height) for _ in range(MAX_ENEMY)]

boss = BOSS(width / 2, 100)

# 检测按键
def is_press_key():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    keys = pygame.key.get_pressed()

    if keys[pygame.K_LEFT]:
        my_plane.x -= 4
    if keys[pygame.K_UP]:
        my_plane.y -= 6
    if keys[pygame.K_RIGHT]:
        my_plane.x += 4
    if keys[pygame.K_DOWN]:
        my_plane.y += 6


def add_enemy():
    for i in range(MAX_ENEMY):
        if not enemy_planes[i].active:
            enemy_planes[i] = Plane(random.randint(0, width - enemy_plane_width), 0, enemy_plane_width,
                                    enemy_plane_height)
            enemy_planes[i].active = True
            enemy_planes[i].bullet[0] = Bullet(enemy_planes[i].x + enemy_plane_width / 2,
                                               enemy_planes[i].y + enemy_plane_height + 10)
            enemy_planes[i].bullet_active[0] = True
            break



# 发射子弹
def shoot():
    if 3 <= level <= 4:
        extra_bullets()
    for i in range(MAX_BULLET):
        if not my_plane.bullet_active[i]:
            my_plane.bullet[i] = Bullet(my_plane.x + my_plane.width / 2, my_plane.y - 10)
            my_plane.bullet_active[i] = True
            break

# 敌机发射子弹
def enemy_shoot():
    for i in range(len(enemy_planes)):
        if enemy_planes[i].active:
            for j in range(MAX_ENEMY):
                if not enemy_planes[i].bullet_active[j]:  # 第i架敌机的第j发子弹
                    enemy_planes[i].bullet_active[j] = True
                    enemy_planes[i].bullet[j] = Bullet(enemy_planes[i].x + enemy_plane_width / 2, enemy_planes[i].y + enemy_plane_height + 10)
                    break


# 碰撞检测
def check_collision():
    global kill_num
    global playerHealth
    # 子弹打敌机
    if 3 <= level <= 4:
        extra_collision_left()
        extra_collision_right()
    for i in range(MAX_ENEMY):
        if enemy_planes[i].active:
            for j in range(MAX_BULLET):
                if my_plane.bullet_active[j]:
                    if enemy_planes[i].x <= my_plane.bullet[j].x + my_plane.radius and enemy_planes[i].x + enemy_plane_width > my_plane.bullet[j].x - my_plane.radius:
                        if my_plane.bullet[j].y < enemy_planes[i].y + enemy_plane_height:
                            kill_num += 1
                            enemy_planes[i].active = False
                            my_plane.bullet_active[j] = False
                            break


    # 检测敌机、敌机子弹和我机的碰撞
    for i in range(MAX_ENEMY):
        # 敌机与我机碰撞-20血
        if enemy_planes[i].active:
            for x, y in [(my_plane.x, my_plane.y),
                         (my_plane.x + my_plane_width, my_plane.y),
                         (my_plane.x, my_plane.y + my_plane_height),
                         (my_plane.x + my_plane_width, my_plane.y + my_plane_height)]:
                if enemy_planes[i].x < x < enemy_planes[i].x + enemy_plane_width and enemy_planes[i].y < y < enemy_planes[i].y + enemy_plane_height:
                    kill_num += 1
                    playerHealth -= 20
                    enemy_planes[i].active = False

        # 敌机子弹与我机碰撞-5血
        for j in range(MAX_BULLET):
            if enemy_planes[i].bullet_active[j]:
                if my_plane.x < enemy_planes[i].bullet[j].x < my_plane.x + my_plane.width:
                    if my_plane.y < enemy_planes[i].bullet[j].y + enemy_planes[i].radius < my_plane.y + my_plane.height:
                        enemy_planes[i].bullet_active[j] = False
                        playerHealth -= 5


    # boss
    if boss.flag:
        boss.check_collision()

# 更新敌机和子弹位置
def update_positions():
    # 敌机
    for i in range(MAX_ENEMY):
        if enemy_planes[i].active:
            enemy_planes[i].y += 2
    # 我机子弹
    if 3 <= level <= 4:
        update_positions_left()
        update_positions_right()
    for i in range(MAX_BULLET):
        if my_plane.bullet_active[i]:
            my_plane.bullet[i].y -= 5
    # 敌机子弹
    for i in range(MAX_ENEMY):
        for j in range(MAX_BULLET):
            if enemy_planes[i].bullet_active[j]:  # 敌机被摧毁子弹仍存在
                enemy_planes[i].bullet[j].y += 4
    if boss.flag:
        boss.update_position()


# 绘制对象
def draw_objects():
    global C
    screen.fill(BLACK)
    # 小怪进度条
    pygame.draw.rect(screen, WHITE, (width - 2, 0, 2, height * (MAX_ENEMY - enemy_num) / MAX_ENEMY))
    pygame.draw.rect(screen, C[level], (0, height - 3, width * (kill_num % 10) / 10, 3))
    # 我机
    pygame.draw.rect(screen, C[level], (my_plane.x, my_plane.y, my_plane.width, my_plane.height))


    if boss.flag:
        # boss自身
        boss.draw()
        # boss血条
        pygame.draw.rect(screen, RED, (0, 0, width * boss.blood / 5000, 4))
        # boss子弹
        for i in range(18):
            for j in range(5):
                if boss.bullet_active[i][j]:
                    pygame.draw.circle(screen, RED, (boss.bullet[i][j].x, boss.bullet[i][j].y), boss.radius)
                    if boss.bullet[i][j].x < 0 or boss.bullet[i][j].x > width or boss.bullet[i][j].y < 0 or boss.bullet[i][j].y > height:
                        boss.bullet_active[i][j] = False

        # 禁区
        pygame.draw.rect(screen, RED, (0, 220, width, 3))


    # 我机子弹
    if 3 <= level <= 4:
        draw_left()
        draw_right()
    for i in range(MAX_BULLET):
        if my_plane.bullet_active[i]:
            pygame.draw.circle(screen, C[level], (int(my_plane.bullet[i].x), int(my_plane.bullet[i].y)), my_plane.radius)
            if my_plane.bullet[i].y < 0:
                my_plane.bullet_active[i] = False

    # 敌机
    for i in range(MAX_ENEMY):
        if enemy_planes[i].active and enemy_planes[i].y <= height:
            pygame.draw.rect(screen, RED, (enemy_planes[i].x, enemy_planes[i].y, enemy_plane_width, enemy_plane_height))

    # 敌机子弹
    for i in range(MAX_ENEMY):
        for j in range(MAX_BULLET):
            if enemy_planes[i].bullet_active[j] and enemy_planes[i].bullet[j].y <= height:
                pygame.draw.circle(screen, RED, (int(enemy_planes[i].bullet[j].x), int(enemy_planes[i].bullet[j].y)), enemy_planes[i].radius)



    font = pygame.font.SysFont(None, 35)
    health_text = font.render(f"Health: {playerHealth}", True, WHITE)
    screen.blit(health_text, (10, 10))
    Boss_text = font.render(f"BoSS: {boss.blood}", True, WHITE)
    screen.blit(Boss_text, (10, 30))
    pygame.display.flip()

clock = pygame.time.Clock()



# 游戏主循环
def game_loop():
    global enemy_num
    global level
    t1 = pygame.time.get_ticks()  # 初始敌机时间
    tt1 = pygame.time.get_ticks()  # 初始子弹时间
    ttt1 = pygame.time.get_ticks()  # 敌机子弹时间
    tttt1 = pygame.time.get_ticks() # boss子弹时间

    running = True
    while running:
        t2 = pygame.time.get_ticks()
        if enemy_num < MAX_ENEMY and not boss.flag:
            if t2 - t1 >= 1000:
                add_enemy()
                enemy_num += 1
                t1 = t2
        else:
            boss.flag = True


        tt2 = pygame.time.get_ticks()
        if tt2 - tt1 >= 150:
            shoot()
            tt1 = tt2

        ttt2 = pygame.time.get_ticks()
        if ttt2 - ttt1 >= 1600:
            enemy_shoot()
            ttt1 = ttt2

        tttt2 = pygame.time.get_ticks()
        if tttt2 - tttt1 >= boss.shoot_interval:
            boss.shoot()
            tttt1 = tttt2


        is_press_key()
        check_collision()
        update_positions()
        draw_objects()

        level = kill_num // 10
        print(my_plane.radius)
        if 1 <= level <= 4:
            my_plane.radius = 6


        if boss.blood <= 0:
            screen.fill(BLACK)
            font = pygame.font.SysFont(None, 35)
            win_text = font.render("You Win", True, WHITE)
            screen.blit(win_text, (width / 2 - 50, height / 2))
            pygame.display.update()
            time.sleep(5)
            running = False
        elif playerHealth <= 0:
            screen.fill(BLACK)
            font = pygame.font.SysFont(None, 35)
            die_text = font.render("You Die", True, WHITE)
            screen.blit(die_text, (width / 2 - 50, height / 2))
            pygame.display.update()
            time.sleep(5)
            running = False

        clock.tick(60)
    pygame.time.delay(300)


if __name__ == "__main__":
    game_loop()



