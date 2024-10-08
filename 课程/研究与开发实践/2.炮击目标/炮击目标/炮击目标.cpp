#include <graphics.h>
#include <stdio.h>
#include <conio.h>
#define _USE_MATH_DEFINES 
#include <cmath>


const float width = 600, height = 800;
const float r = 15; // 球的半径
const float g = 1;  // 重力加速度
const int angle_max = 90; // 最大角度
const int angle_min = -90; // 最小角度


// 计算球的未来位置，给定当前速度和时间
void predictPosition(float& px, float& py, float vx, float vy, float g, int time) {
    px = (float)(vx * time); // x = vx * t
    py = (float)(height / 2 + vy * time - 0.5 * g * time * time); // y = 400 + vy * t - 0.5 * g * t^2
}


// 绘制炮管
void drawCannon(int angle) {
    int cx = 0, cy = height / 2; // 炮管基点
    int length = 50; // 炮管长度
    double radians = angle * M_PI / 180.0;
    int x2 = cx + length * cos(radians);
    int y2 = cy - length * sin(radians); // y轴方向是向下的，所以加上

    setcolor(WHITE);
    line(cx, cy, x2, y2); // 画炮管
}


// 绘制炮弹的预测路径
void drawPredictionPath(float angle) {
    double radians = angle * M_PI / 180.0;
    float vx = 30 * cos(radians);
    float vy = 23 * sin(radians);

    for (int t = 1; vx * t <= 0.75 * width; t++) {
        float px, py;
        predictPosition(px, py, vx, vy, g, t);
        if (px >= width) {
            break;
        }// 如果超出屏幕范围，退出循环
        py = height - py;
        putpixel(px, py, WHITE); // 绘制炮弹路径
    }
}


//输出得分
void printscore(TCHAR s[], int balls_left, int score) {
    wsprintf(s, TEXT("剩余球数：%i      得分：%i/8"), balls_left, score);
    outtextxy(100, 100, s);
}


int main() {
    int x = 0, y = height / 2;
    int angle = 0; // 初始炮管角度，0代表正右方
    float vx = 0, vy = 0;
    bool f = true; // 允许开炮标志
    bool suc = false; // 新增：靶子是否被击中的标记
    int score = 0; // 积分
    bool score_plus = true;//允许加分标志
    int color = WHITE;
    int balls_left = 8;


    initgraph(width, height);
    setbkcolor(RGB(50, 50, 50));


    int p = 50; // 靶子初始位置
    int vp = 3; // 靶子移动速度


    printf("通过“w”、“s”来调整炮管角度\n空格键发射");
    while (1) {
        cleardevice(); // 清屏


        // 根据靶子是否被击中来设置靶子的颜色
        if (suc) {
            setfillcolor(GREEN);
            setlinecolor(GREEN);
        }
        else {
            setfillcolor(WHITE);
            setlinecolor(WHITE);
        }


        // 绘制靶子
        fillrectangle(580, p, 600, p + 60);


        p += vp;
        if (p > height || p < 0) {
            vp = -vp;
        }


        // 处理按键输入来发射炮弹和调整炮管角度
        if (_kbhit()) {
            char input = _getch();
            if (input == ' ' && f == true && balls_left >0) {
                f = false;
                balls_left--;
                // 计算发射速度分量
                double radians = angle * M_PI / 180.0;
                vx = 30 * cos(radians);
                vy = -23 * sin(radians);
                suc = false; // 每次发射时重置靶子击中状态
            }
            else if (input == 'w' && angle <= angle_max - 5) { // 检测到特殊键 (上下箭头)转动炮管，角度限制
                angle = angle + 5; // 增加角度
            }
            else if (input == 's' && angle >= angle_min + 5) {
                angle = angle - 5; // 减少角度
            }
        }


        // 绘制炮管
        drawCannon(angle);


        // 开炮！
        if (!f) {
            vy += g; // 加速
            x += vx; // 更新位置
            y += vy;


            // 检查是否击中靶子
            if (x + r >= 580 && x - r <= 600 && y + r >= p && y - r <= p + 60) {
                suc = true; // 靶子被击中
                if (score_plus) {
                    score++;
                    score_plus = false;
                }
            }
            else {
                suc = false;
            }


            // 如果炮弹超过屏幕，重置位置
            if (x - r >= width || y - r >= height) {
                x = 0;
                y = height / 2;
                f = true; // 超过后允许再次开火
                score_plus = true;
            }
            
        }
        else {
            // 绘制炮弹的预测路径
            drawPredictionPath(angle);
            x = 0; // 当未发射炮弹时，重置位置
            y = height / 2;
        }


        // 绘制球
        setfillcolor(WHITE); // 确保球的颜色不绿
        fillcircle(x, y, r);


        TCHAR s[50];
        printscore(s, balls_left, score);


        // 刷新屏幕
        Sleep(40);
    }


    _getch();
    closegraph();
    return 0;
}