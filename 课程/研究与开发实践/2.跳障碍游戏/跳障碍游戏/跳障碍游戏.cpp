#include <graphics.h>
#include <stdio.h>
#include <conio.h>
#include <windows.h>  // For Sleep()


void printscore(TCHAR s[], int score) {
    wsprintf(s, TEXT("%i"), score);
    outtextxy(100, 100, s);
}


int main() {
    float width = 600, height = 400;
    float ball_x, ball_y, radius = 20;
    float x = 200, y = 380;
    float vy = 0, g = 1;
    ball_x = width / 4;
    ball_y = height - radius;
    int suc = 1;
    int score_plus = 1;
    TCHAR s[5];
    int score = 0;
    int rec = 0;
    int color = WHITE;
    int jump = 2;//只允许二段跳
    int min_rex_height = 100, max_rec_height = 300, rec_height = 300;
    int fail = 0;


    initgraph(width, height);


    //开始游戏控制
     printf("按“s”开始游戏...\n");
    while (_getch() != 's') {
        // 等待 's' 键被按下
    }
    printf("游戏开始，退出请按“q”...");
    int time = 0;


    //游戏中
    while (1) {
        setfillcolor(WHITE);
        setlinecolor(WHITE);
        //setfillstyle(SOLID_FILL, WHITE);

        //设置二段跳限制的跳跃操控
        if (_kbhit() && jump > 0) {
            char input = _getch();
            if (input == ' ') {
                vy = -20;
                jump--;
            }
            else if (input == 'q') {
                break;
            }
        }
        vy = vy + g;
        y = y + vy;

        if (y >= height - radius) {
            vy = -0.95 * vy;
            y = height - radius;
            if (y == height - radius) {
                jump = 2;
            }
        }

        cleardevice();
        fillcircle(x, y, radius);
        
        //每个新矩形独立速度、高度控制
        int speed = 5;
        if (rec == 0) {
            speed = rand() % (5 - 1 + 1) + 1;
            rec_height = rand() % (max_rec_height - min_rex_height + 1) + min_rex_height;
            printf("new rec\nspeed:%d\nrec_height:%d\n", speed, rec_height);
            rec = 1;
        }


        //是否碰撞
        //左:570 - speed * time,右:600 - speed * time,上:y + radius
        if (suc == 1) {
            if (570 - speed * time <= x + radius && 600 - speed * time >= x - radius && height - rec_height <= y + radius) {
                suc = 0;
            }
        }


        // 碰撞后矩形变红
        if (suc == 0) {
            setfillcolor(RED);
            setlinecolor(RED);
        }
        else {
            setfillstyle(SOLID_FILL, color);  // 使用当前颜色绘制矩形
        }
        

        fillrectangle(570 - speed * time, height - rec_height, 600 - speed * time, height); //矩形
        
        
        time += 1;
        //球最左侧过矩形最右侧后进行成功判定，不要等矩形到最左
        if (600 - speed * time < x - radius && score_plus == 1) {
            if (suc == 1) {
                score += suc;
                score_plus = 0;
            }
            else {
                printf("touch!\n");
                score_plus = 0;
            }
        }


        //矩形到最左侧
        if (570 - speed * time == 0) {
            if (suc == 0) {
                fail += 1;
            }
            time = 0, rec = 0, suc = 1, score_plus = 1;
        }
        printscore(s, score);
        Sleep(10);
    }

    
    closegraph();
    printf("fail:%d", fail);
    return 0;
}