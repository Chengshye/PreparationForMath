"""元胞自动机模板（生命游戏）"""

import pygame, sys, time, random

width = 102  # 面板的宽度（外围有一层墙）
high = 102  # 面板的高度（外围有一层墙）
size = 6  # 设置绘制的单方格大小


def initialization(arr):  # 初始化
    for i in range(high):
        for j in range(width):
            ran = random.random()
            if ran > 0.9:
                arr[i][j] = 1
            else:
                pass
    return arr


def nextmultiply(arr):  # 下一代繁衍
    newarr = [([0] * width) for n in range(high)]
    for i in range(high):
        for j in range(width):
            num = 0
            if (i == 0 or i == high - 1) or (j == 0 or j == width - 1):
                newarr[i][j] = 0
            else:
                num = arr[i - 1][j - 1] + arr[i - 1][j] + arr[i - 1][j + 1] + arr[i][j - 1] + arr[i][j + 1] + \
                      arr[i + 1][j - 1] + arr[i + 1][j] + arr[i + 1][j + 1]
                if arr[i][j] == 0:  # 若原细胞为死亡状态
                    if num == 3:
                        newarr[i][j] = 1
                else:  # 若原细胞为存活状态
                    if num == 2 or num == 3:
                        newarr[i][j] = 1
                    else:
                        newarr[i][j] = 0
    return newarr


if __name__ == '__main__':
    color_white = pygame.Color(255, 255, 255)
    color_LightSkyBlue = pygame.Color(135, 206, 250)
    color_black = pygame.Color(0, 0, 0)
    pygame.init()
    screen = pygame.display.set_mode((width * size, high * size))
    screen.fill(color_white)
    pygame.display.set_caption("生命游戏Game of Life")
    arr = [([0] * width) for i in range(high)]  # 创建一个二维数组
    arr = initialization(arr)
    while (True):
        screen.fill(color_white)
        time.sleep(0.1)
        for i in range(high):
            for j in range(width):
                if arr[i][j] == 1:
                    pygame.draw.rect(screen, color_black, (i * size, j * size, size, size))
                elif (i == 0 or i == high - 1) or (j == 0 or j == width - 1):
                    pygame.draw.rect(screen, color_LightSkyBlue, (i * size, j * size, size, size))
                else:
                    pass
        for event in pygame.event.get():  # 监听器
            if event.type == pygame.QUIT:
                sys.exit()
        arr = nextmultiply(arr)
        pygame.display.update()
