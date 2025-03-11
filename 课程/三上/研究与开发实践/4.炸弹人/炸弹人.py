from configuration import *
import time

# 初始化
pygame.init()
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("炸弹人")

# 初始化道具效果标志
has_power_up = False
current_power_up = None

# 实例化对象
mike = Mike(50, 50)
brick_grid = []
brick_1 = []
bombs = []
powerUps = []
font = pygame.font.SysFont(None, 36)

# 设置游戏计时
game_start = pygame.time.get_ticks()
game_duration = 100  # 游戏持续时间（秒）

# 创建砖块
for i in range(15):
    for j in range(19):
        brick_type = 0
        if (i % 2 == 0 and j % 2 == 0) or (i == 0 or i == 14) or (j == 0 or j == 18):
            brick_type = 2
        elif random.randint(0, 7) == 0 and i>=3 and j>=3:
            brick_type = 1
        new_brick = Brick(50 * j, 50 * i, brick_type)
        brick_grid.append(new_brick)
        if brick_type == 1:
            brick_1.append(new_brick)

# 初始化 Mike 的位置
mike.left = brick_grid[20].left
mike.top = brick_grid[20].top
mike.rect = pygame.Rect(mike.left, mike.top, mike.length, mike.length)


# 初始化怪物
mike_position = (mike.top // 50, mike.left // 50)
monsters = initialize_monsters(5, brick_grid)


# 初始化道具
# 确保brick_1列表有足够的砖块
if len(brick_1) >= 2:
    # 随机选取两个砖块作为道具一和道具二所在的砖块
    selected_bricks_for_powerups = random.sample(brick_1, 2)

    # 创建道具一和道具二
    powerUp1 = PowerUp(selected_bricks_for_powerups[0].left + 10, selected_bricks_for_powerups[0].top + 10, 1)  # 方形道具
    powerUp2 = PowerUp(selected_bricks_for_powerups[1].left + 10, selected_bricks_for_powerups[1].top + 10, 2)  # 十字形道具

    # 将道具加入道具列表
    powerUps.append(powerUp1)
    powerUps.append(powerUp2)


# 主循环
running = True
while running:
    screen.fill(BACKGROUND_COLOR)

    # 检测事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()

    # 获取碰撞检测数组
    can_move = mike_brick_col(mike, brick_grid)

    # 更新 Mike 的位置
    if keys[pygame.K_LEFT] and can_move[0]:
        mike.left -= 5
    if keys[pygame.K_UP] and can_move[1]:
        mike.top -= 5
    if keys[pygame.K_RIGHT] and can_move[2]:
        mike.left += 5
    if keys[pygame.K_DOWN] and can_move[3]:
        mike.top += 5

    # 更新 Mike 的 rect
    mike.rect = pygame.Rect(mike.left, mike.top, mike.length, mike.length)

    if keys[pygame.K_SPACE]:
        bombs.append(Bomb(mike.left // 50 * 50 + 25, mike.top // 50 * 50 + 25))

    # 更新炸弹状态
    for bomb in bombs:
        if bomb.active:
            pygame.draw.circle(screen, RED, (bomb.x, bomb.y), 10)
            if pygame.time.get_ticks() - bomb.last_time > 3000:  # 爆炸时间为3秒
                bomb.active = False
                explode_bomb(bomb, brick_grid, monsters, screen,
                             power_up_type=current_power_up if has_power_up else None)
                # # 检查Mike是否在爆炸范围内，后续再改进吧
                # if bomb_collision(bomb, mike):
                #     screen.fill(BACKGROUND_COLOR)
                #     text = font.render("Game Over!", True, (255, 255, 255))
                #     screen.blit(text, (width // 2 - 100, height // 2))
                #     pygame.display.update()
                #     time.sleep(5)
                #     running = False
                #     break  # 如果游戏结束，退出炸弹更新循环
                bombs.remove(bomb)
                mike.color = BLUE
                has_power_up = False  # 道具效果只生效一次
                current_power_up = None

    # 更新怪物位置
    for monster in monsters:
        monster.move(brick_grid)
        pygame.draw.rect(screen, RED, (monster.left, monster.top, monster.length, monster.length))
        if collision(mike, monster):
            screen.fill(BACKGROUND_COLOR)
            text = font.render("Game Over!", True, (255, 255, 255))
            screen.blit(text, (width // 2 - 100, height // 2))
            pygame.display.update()
            time.sleep(5)
            running = False



    # 道具效果碰撞检测
    for powerUp in powerUps[:]:
        if powerUp.active:
            # 绘制道具（黄色）
            if powerUp.shape == 1:
                pygame.draw.rect(screen, (255, 255, 0), powerUp.rect)  # 方形道具
            else:
                pygame.draw.polygon(screen, (255, 255, 0),
                                    [(powerUp.x + 15, powerUp.y), (powerUp.x + 30, powerUp.y + 15),
                                     (powerUp.x + 15, powerUp.y + 30), (powerUp.x, powerUp.y + 15)])  # 十字形道具

            # 检查Mike是否碰到道具
            if mike.rect.colliderect(powerUp.rect):
                powerUp.active = False
                has_power_up = True
                current_power_up = powerUp.shape  # 记录道具类型
                mike.color = (255, 255, 0)  # Mike变成黄色


    # # 绘制道具并检查
    # for powerUp in powerUps:
    #     if powerUp.active:
    #         pygame.draw.circle(screen, GREEN, (powerUp.x + 25, powerUp.y + 25), 10)
    #         if mike.left < powerUp.x + 10 and mike.left + mike.length * 2 > powerUp.x and \
    #                 mike.top < powerUp.y + 10 and mike.top + mike.length * 2 > powerUp.y:
    #             powerUp.active = False

    # 绘制砖块
    for brick in brick_grid:
        if brick.type != 0:
            pygame.draw.rect(screen, brick.color, brick.rect)
            pygame.draw.rect(screen, (0, 0, 0), brick.rect, 1)  # 黑色边框

    # 绘制 Mike
    pygame.draw.rect(screen, mike.color, mike.rect)

    if not monsters:
        screen.fill(BACKGROUND_COLOR)
        text = font.render("You Win!", True, (255, 255, 255))
        screen.blit(text, (width // 2 - 100, height // 2))
        pygame.display.update()
        time.sleep(5)
        running = False

    # 绘制时间
    elapsed_time = (pygame.time.get_ticks() - game_start) // 1000
    remaining_time = game_duration - elapsed_time
    text = font.render(f"Time: {remaining_time}", True, (255, 255, 255))
    screen.blit(text, (10, 10))

    if remaining_time <= 0:
        screen.fill(BACKGROUND_COLOR)
        win = len(monsters) == 0
        text = font.render("You Win!" if win else "Game Over!", True, (255, 255, 255))
        screen.blit(text, (width // 2 - 100, height // 2))
        pygame.display.update()
        time.sleep(5)
        running = False

    pygame.display.update()
    pygame.time.delay(30)

pygame.quit()