#include <graphics.h>  
#include <conio.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

// 板参数
int board_width = 100, board_height = 10; // bw板长，bh板高
int speed = 1;
int x, y = 590;

// Mike移动参数
int v = 3;
int drop_v = 5;

int type[6];

struct Board {
    int left;
    int top;
    int type; // 0:常规、1:一踩就破、2:踩了掉血
    int round; // 当前轮数
};

struct Mike {
    int left;
    int top;
    int right;
    int bottom;
};

void printstart(TCHAR text[], int w, int h) {
    wsprintf(text, TEXT("Press 's' to start the game"));
    outtextxy(w / 2, h / 2, text);
}

int mike_on_board(Mike mike, Board board[6]) {
    for (int i = 0; i < 6; i++) {
        if (board[i].top < mike.bottom && mike.bottom <= board[i].top + board_height) {
            if (mike.left > board[i].left && mike.left < board[i].left + board_width) {
                return i;
            }
            else if (mike.right > board[i].left && mike.right < board[i].left + board_width) {
                return i;
            }
        }
    }
    return -1;
}

void type_change(int type[]) {
    for (int i = 0; i < 6; i++) {
        type[i] = 0; // 初始化数组元素
    }
    type[0] = 0; // 默认常规板
    int type1_index = rand() % 6; // 0到5之间
    int type2_index;
    do {
        type2_index = rand() % 6; // 0到5之间
    } while (type2_index == type1_index);

    type[type1_index] = 1;
    type[type2_index] = 2;
}

void printlife(TCHAR s[], int life) {
    wsprintf(s, TEXT("life:%i"), life);
    outtextxy(500, 500, s);
}

void printfloor(TCHAR s[], int floor) {
    wsprintf(s, TEXT("floor:%i"), floor);
    outtextxy(500, 400, s);
}



int main() {
    // graph参数
    int width = 600, height = 600;
    int index;

    // mike参数
    int mike_width = 20;
    int mike_height = 30;

    int i = 0;
    TCHAR text1[20];
    char S;

    bool start = false;
    bool change = true;

    // 随机种子
    srand((unsigned int)time(NULL));
    type_change(type);

    //命
    int life = 3;
    bool life_dec = true;
    TCHAR s[5];

    int floor = 0;
    int last_board = 0;
    bool on_other_board = false;
    bool win = false;

    Board board[6];
    Mike mike;
    initgraph(width, height);

    // 等待用户按 's' 开始游戏
    printstart(text1, width, height);
    do {
        
        S = _getch(); // 获取用户输入，但不显示在屏幕上
    } while (S != 's' && S != 'S'); // 如果不是 's' 或 'S'，则继续等待

    for (i = 0; i < 6; i++) {
        board[i].left = rand() % 500;
        board[i].top = 590;
        board[i].type = 0; // 初始化板的类型
    }

    int type1_index = 1 + rand() % 5; // 0到5之间
    int type2_index;
    do {
        type2_index = 1 + rand() % 5; // 0到5之间
    } while (type2_index == type1_index);

    board[type1_index].type = 1;
    board[type2_index].type = 2;

    BeginBatchDraw();
    
    while (1) {
        //结束检测
        if (floor > 100) {
            win = true;
        }
        if (win) {
            cleardevice(); // 只在循环开始时清屏
            settextstyle(100, 0, _T("Arial"));
            outtextxy(200, 200, _T("Win")); // 输出 Win
            Sleep(100); // 暂停100秒
        }
        else if (life == 0 && !win) {
            cleardevice(); // 只在循环开始时清屏
            settextstyle(100, 0, _T("Arial"));
            outtextxy(200, 200, _T("Lose")); // 输出 Lose
            Sleep(100); // 暂停100秒
        }
        
        board[0].type = 0;
        // 画顶上的尖刺
        for (i = 0; i < 20; i++) {
            line(30 * i, 0, 30 * i + 15, 30);
            line(30 * i + 15, 30, 30 * (i + 1), 0);
        }

        // 对每个板进行处理
        for (i = 0; i < 6; i++) {
            // 如果是第一个板子或者与上一个板子间距大于100，或者到达顶部
            if (i == 0 || board[i].top - board[i - 1].top > 100 || board[i].top <= 100) {
                if (change) {
                    type_change(type);
                    change = false;
                }
                if (board[i].type == 1) {
                    setfillcolor(YELLOW);
                }
                else if (board[i].type == 2) {
                    setfillcolor(RED);
                }
                else {
                    setfillcolor(WHITE);
                }
                fillrectangle(board[i].left, board[i].top, board[i].left + board_width, board[i].top + board_height);
                board[i].top -= speed;
                if (i == 0 && !start) {
                    start = true;
                    mike.left = board[i].left + (board_width - mike_width) / 2;
                    mike.top = board[i].top - mike_height;
                    mike.right = mike.left + mike_width;
                    mike.bottom = mike.top + mike_height;
                }
            }
            if (board[i].top < 0) {
                board[i].top = height - board_height; // 重置到底部
                board[i].left = rand() % 500; // 随机位置
                if (i == 0) {
                    type1_index = rand() % 6; // 0到5之间
                    do {
                        type2_index = rand() % 6; // 0到5之间
                    } while (type2_index == type1_index);
                }
                else {
                    if (i == type1_index) {
                        board[i].type = 1;
                    }
                    else if (i == type2_index) {
                        board[i].type = 2;
                    }
                    else {
                        board[i].type = 0;
                    }
                }
                
            }
        }

        // 更新和绘制 Mike
        index = mike_on_board(mike, board);
        
        
        if (index != last_board && index >= 0) {
            if (index > last_board) {
                floor += index - last_board;
            }
            else {
                floor += 6 - last_board + index;
            }
            
        }
        if (index >= 0) {
            last_board = index;
        }
        
        if (index >= 0 && last_board >= 0) {
            if (board[index].type == 1) {
                mike.top += drop_v;
                mike.bottom += drop_v;
                life_dec = true;
            }
            else if (board[index].type == 2) {
                if (life_dec) {
                    life_dec = false;
                    life--;
                }
                mike.top -= speed;
                mike.bottom -= speed;
            }
            else {
                mike.top -= speed;
                mike.bottom -= speed;
                life_dec = true;
            }
        }
        else {
            mike.top += drop_v;
            mike.bottom += drop_v;
        }

        if (mike.top < 30 || mike.top >= height) {
            life = 0;
        }
        setfillcolor(WHITE);
        fillrectangle(mike.left, mike.top, mike.right, mike.bottom);

        

        // 检测输入
        if (_kbhit()) {
            char input = _getch();
            if (GetAsyncKeyState(VK_LEFT)) {
                mike.left -= v;
                mike.right -= v;
            }
            if (GetAsyncKeyState(VK_RIGHT)) {
                mike.left += v;
                mike.right += v;
            }
        }
        if (floor < 100 && life != 0) {
            printlife(text1, life);
            printfloor(text1, floor);
        }
        
        FlushBatchDraw();
        Sleep(20);
        cleardevice(); // 清除屏幕
    }
    cleardevice();
    
    _getch();
    closegraph();
    return 0;
}