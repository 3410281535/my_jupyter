#include <graphics.h>
#include <conio.h>
#include <stdio.h>
#include <vector>
#include <stdlib.h>
#include <time.h>

struct Brick {
	bool drawable; // 是否可绘制
	bool collidable; // 是否可碰撞
	int left, top, right, bottom;
	bool prop1 = false;//道具1：板子变长
	bool prop2 = false;//道具2：变两个球
};
struct Ball {
	int x;
	int y;
	int r;
};

struct Board {
	int left;
	int len;
	int thick;
	int top;
};

void printlife1(TCHAR s[], int life) {
	wsprintf(s, TEXT("%i"), life);
	outtextxy(500, 500, s);
}

int main()
{
	//砖块参数
	int w = 90, h = 40;
	int gap = 5;

	//graph参数
	int width = 6 * w + 7 * gap, height = 600;//width=575

	//板参数
	Board board;

	board.left = 220;
	board.len = (width / 2 - board.left) * 2;
	board.thick = 15;
	board.top = 560;

	//球参数
	int r = 10;
	int x = board.left + board.len / 2, y = board.top - r;
	int vx = 3, vy = -3;//球速

	Ball ball;
	ball.x = x;
	ball.y = y;
	ball.r = r;
	
	int i, j;
	
	
	//碰撞检测所需参数
	int index;
	//砖块剩余
	int brick_left = 0;

	//命
	int life = 3;

	TCHAR s[5];


	//道具序号
	int num1, num2;
	srand(time(0));
	num1 = rand() % 30;
	while (1) {
		num2 = rand() % 30;
		if (num1 != num2) {
			break;
		}
	}


	Brick brick[30];
	for (i = 0; i < 6; i++) {
		for (j = 0; j < 5; j++) {
			brick[j * 6 + i].drawable = true;
			brick[j * 6 + i].collidable = false;//内层无法被碰撞
			brick[j * 6 + i].left = (i + 1) * gap + i * w;
			brick[j * 6 + i].top = (j + 1) * gap + j * h;
			brick[j * 6 + i].right = (i + 1) * gap + (i + 1) * w;
			brick[j * 6 + i].bottom = (j + 1) * gap + (j + 1) * h;
			brick[j * 6 + i].prop1 = false;
			brick[j * 6 + i].prop2 = false;
		}
	}
	brick[num1].prop1 = true;
	brick[num2].prop2 = true;


	// 存储可碰撞砖块索引的数组
	std::vector<int> Brick_Colliadable;

	for (i = 0; i < 6; i++) {
		for (j = 4; j < 5; j++) {
			brick[j * 6 + i].collidable = true;
		}
	}

	
	bool start = false;//游戏开始标志，初始为false

	initgraph(width, height);

	setbkcolor(RGB(200, 200, 200));
	setlinecolor(RGB(0, 128, 128));
	setfillcolor(RGB(0, 128, 128));
	cleardevice();
	//批量绘制
	BeginBatchDraw();
	while (1)
	{
		//画 砖
		setfillcolor(RGB(0, 128, 128));
		//七列五层，记得改graph的width
		brick_left = 0;
		for (i = 0; i < 6; i++) {
			for (j = 0; j < 5; j++) {
				if (brick[j * 6 + i].drawable) {
					brick_left += 1;
					if (brick[j * 6 + i].collidable) {
						//本轮可碰撞砖块组
						Brick_Colliadable.push_back(j * 6 + i);
						setfillcolor(RGB(0, 128, 128));
						if (brick[j * 6 + i].prop1) {
							setfillcolor(YELLOW);
						}
						if (brick[j * 6 + i].prop2) {
							setfillcolor(GREEN);
						}
					}
					else if (brick[j * 6 + i].prop1) {
						setfillcolor(YELLOW);
					}
					else if (brick[j * 6 + i].prop2) {
						setfillcolor(GREEN);
					}
					else {
						setfillcolor(RGB(0, 128, 128));
					}
					fillrectangle(brick[j * 6 + i].left, brick[j * 6 + i].top, brick[j * 6 + i].right, brick[j * 6 + i].bottom);
				}
				else {
					brick[j * 6 + i].collidable = false;//可画才可撞
				}
			}
		}
		if (!brick[num1].prop1) {
			board.len = 150;
		}
		if (!brick[num2].prop2) {
			;
		}
		//画 板
		setfillcolor(RGB(0, 0, 128));
		fillrectangle(board.left, board.top, board.left + board.len, board.top + board.thick);

		if (_kbhit()) {
			char input = _getch();
			//发射
			if ((input == ' ') && (!start))
			{
				//游戏开始、球已发射标志
				start = true; 
			}
			if (GetAsyncKeyState(VK_LEFT))
			{
				if (board.left >= 4) {//这样比较好追
					board.left = board.left - 4;
				}
			}
			if (GetAsyncKeyState(VK_RIGHT))
			{
				if (board.left + board.len <= width - 4) {
					board.left = board.left + 4;
				}
			}
		}
		if (!start) {//未开始时球跟随板中间移动
			ball.x = board.left + board.len / 2, ball.y = board.top - ball.r;
			setfillcolor(RGB(128, 0, 0));
			fillcircle(ball.x, ball.y, ball.r);
		}
		else {//游戏开始后
			//碰撞判断以确定球运动方向
			//碰板
			if (((ball.y + ball.r > board.top) && (ball.x > board.left) && (ball.x < board.left + board.len) || (ball.y - ball.r < 0))) {
				vy = -vy;
			}

			//碰左右边界
			if ((ball.x + r >= width) || (ball.x - r <= 0)) {
				vx = -vx;
			}

			//检测可碰撞砖块碰撞检测先左右后上下
			for (size_t i = 0; i < Brick_Colliadable.size(); i++) {
				 index = Brick_Colliadable[i];
				 //最左侧砖块，检测砖块底、右、顶
				 if (index % 6 == 0) {
					 if ((ball.x - ball.r < brick[index].right) && (ball.x >brick[index].right) && (brick[index].bottom > ball.y) && (ball.y > brick[index].top)) {//撞右（其右无砖块）
						 brick[index].drawable = false;
						 brick[index].prop1 = false;
						 brick[index].prop2 = false;
 
						 vx = -vx;

						 if (index >= 6 && brick[index - 6].drawable) {//上可撞
							 brick[index - 6].collidable = true;
						 }
						 if (index <= 21 && brick[index + 6].drawable) {//下可撞
							 brick[index + 6].collidable = true;
						 }
						 break;
					 }
					 if ((ball.y - ball.r < brick[index].bottom) && (ball.y > brick[index].bottom) && (ball.x < brick[index].right && (ball.x > brick[index].left))) {//撞底（其下无砖块）
						 brick[index].drawable = false;
						 brick[index].prop1 = false;
						 brick[index].prop2 = false;
 
						 vy = -vy;

						 if (index >= 6 && brick[index - 6].drawable) {//上可撞
							 brick[index - 6].collidable = true;
						 }
						 if (brick[index + 1].drawable) {//右可撞
							 brick[index + 1].collidable = true;
						 }
						 break;
					 }
					 if ((ball.y + ball.r > brick[index].top) && (ball.y < brick[index].top) && (ball.x < brick[index].right)) {//撞上（其上无砖块）
						 brick[index].drawable = false;
						 brick[index].prop1 = false;
						 brick[index].prop2 = false;
 
						 vy = -vy;

						 if (brick[index + 1].drawable) {//右可撞
							 brick[index + 1].collidable = true;
						 }
						 if (index <= 21 && brick[index + 6].drawable) {//下有则下可撞
							 brick[index + 6].collidable = true;
						 }
						 break;
					 }
				 }//////////////////////昨晚到这
				 //最右侧砖块，检测砖块底、左、顶
				 else if (index % 6 == 5) {
					 if ((ball.x + ball.r > 6 * gap + 5 * w) && (brick[index].bottom < ball.y) && (ball.y < brick[index].top)) {//撞左（其左无砖块）
						 brick[index].drawable = false;
						 brick[index].prop1 = false;
						 brick[index].prop2 = false;
 
						 vx = -vx;

						 if (index >= 11 && brick[index - 6].drawable) {//上可撞
							 brick[index - 6].collidable = true;
						 }
						 if (index <= 23 && brick[index + 6].drawable) {//下可撞
							 brick[index + 6].collidable = true;
						 }
						 break;
					 }
					 if ((ball.y - ball.r < brick[index].bottom) && (ball.y > brick[index].bottom) && (ball.x > brick[index].left) && ball.x < brick[index].right) {//撞底
						 brick[index].drawable = false;
						 brick[index].prop1 = false;
						 brick[index].prop2 = false;
 
						 vy = -vy;

						 if (index >= 11 && brick[index - 6].drawable) {//上可撞
							 brick[index - 6].collidable = true;
						 }
						 if (brick[index - 1].drawable) {//左可撞
							 brick[index - 1].collidable = true;
						 }
						 break;
					 }
					 if ((ball.y + ball.r > brick[index].top) && (ball.y < brick[index].top) && (brick[index].left < ball.x) && (ball.x < brick[index].right)) {//撞上（其上无砖块）
						 brick[index].drawable = false;
						 brick[index].prop1 = false;
						 brick[index].prop2 = false;
 
						 vy = -vy;

						 if (brick[index - 1].drawable) {//左可撞
							 brick[index - 1].collidable = true;
						 }
						 if (index <= 21 && brick[index + 6].drawable) {//下可撞
							 brick[index + 6].collidable = true;
						 }
						 break;
					 }
				 }
				 //中间砖块，由于尖角问题，直接一个砖块消除后上下左右都可碰
				 else
				 {
					 if ((ball.x + ball.r > brick[index].left) && (ball.x < brick[index].left) && (brick[index].top < ball.y) && (ball.y < brick[index].bottom)) {//撞左（其左无砖块）
						 brick[index].drawable = false;
						 brick[index].prop1 = false;
						 brick[index].prop2 = false;
 
						 vx = -vx;

						 if (index / 6 >= 1 && brick[index - 6].drawable) {//上可撞
							 brick[index - 6].collidable = true;
						 }
						 if (index / 6 < 4 && brick[index + 6].drawable) {//下可撞
							 brick[index + 6].collidable = true;
						 }
						 if (brick[index + 1].drawable) {//右可撞
							 brick[index + 1].collidable = true;
						 }
						 if (brick[index - 1].drawable) {
							 brick[index - 1].collidable = true;
						 }
						 break;
					 }
					 if ((ball.x - ball.r < brick[i].right) && (ball.x > brick[i].right) && (brick[index].top < ball.y) && (ball.y < brick[index].bottom)) {//撞右（其右无砖块）
						 brick[index].drawable = false;
						 brick[index].prop1 = false;
						 brick[index].prop2 = false;
 
						 vx = -vx;

						 if (index / 6 >= 1 && brick[index - 6].drawable) {//上可撞
							 brick[index - 6].collidable = true;
						 }
						 if (brick[index - 1].drawable) {//左可撞
							 brick[index - 1].collidable = true;
						 }
						 if (brick[index + 1].drawable) {//右可撞
							 brick[index + 1].collidable = true;
						 }
						 if (index / 6 < 4 && brick[index + 6].drawable) {//下可撞
							 brick[index + 6].collidable = true;
						 }
						 break;
					 }
					 if ((ball.y - ball.r < brick[index].bottom) && (ball.y > brick[index].bottom) && (brick[index].left < ball.x) && (ball.x < brick[index].right)) {//撞底
						 brick[index].drawable = false;
						 brick[index].prop1 = false;
						 brick[index].prop2 = false;
 
						 vy = -vy;

						 if (index / 6 >= 1 && brick[index - 6].drawable) {//上可撞
							 brick[index - 6].collidable = true;
						 }
						 if (index / 6 < 4 && brick[index + 6].drawable) {//下可撞
							 brick[index + 6].collidable = true;
						 }
						 if (brick[index - 1].drawable) {//左可撞
							 brick[index - 1].collidable = true;
						 }
						 if (brick[index + 1].drawable) {//右可撞
							 brick[index + 1].collidable = true;
						 }
						 break;
					 }
					 if ((ball.y + ball.r > brick[index].top) && (ball.y < brick[index].top) && (brick[index].left < ball.x) && (ball.x < brick[index].right)) {//撞上（其上无砖块）
						 brick[index].drawable = false;
						 brick[index].prop1 = false;
						 brick[index].prop2 = false;
 
						 vy = -vy;

						 if (index / 6 >= 1 && brick[index - 6].drawable) {//上可撞
							 brick[index - 6].collidable = true;
						 }
						 if (brick[index - 1].drawable) {//左可撞
							 brick[index - 1].collidable = true;
						 }
						 if (index / 6 < 4 && brick[index + 6].drawable) {//下可撞
							 brick[index + 6].collidable = true;
						 }
						 if (brick[index + 1].drawable) {//右可撞
							 brick[index + 1].collidable = true;
						 }
						 break;
					 }
				 }
				 // 从 Brick_Colliadable 中移除
				 Brick_Colliadable.erase(Brick_Colliadable.begin() + i);
				 i--;
			}



			//判断完运动方向后，绘制游戏中球的位置
			ball.x = ball.x + vx;
			ball.y = ball.y + vy;
			
			if (brick_left == 0) {
				cleardevice();
				outtextxy(230, 400, _T("Win"));
				
			}
			if (ball.y - ball.r >= height) {
				settextcolor(RGB(0, 0, 255));
				settextstyle(60, 0, _T("Arial"));
				if (start) {
					life--;
				}
				if (life == 0) {
					outtextxy(230, 400, _T("Die"));
				}
				else {
					outtextxy(230, 400, _T("Lose"));
				}
				
				ball.x = board.left + board.len / 2, ball.y = board.top - ball.r;
				vx = 3, vy = -3;//重置球速
				start = false;
				//  break;
				}
			setfillcolor(RGB(128, 0, 0));
			fillcircle(ball.x, ball.y, ball.r);
		}
		if (life == 0) {
			Sleep(3000);

		}

		settextstyle(60, 0, _T("Arial"));
		printlife1(s, life);


		FlushBatchDraw();
		Sleep(3);//Sleep时间从30->3，速度缩小十倍
		cleardevice();
	}
	closegraph();
	return 0;
}