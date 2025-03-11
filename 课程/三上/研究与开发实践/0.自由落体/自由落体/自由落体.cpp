#include <graphics.h>
#include <stdio.h>
#include <conio.h>
#include <windows.h>  // For Sleep()

int main() {
    float y = 100;  
    float vy = 0;   
    float g = 0.5;   
    int w = 600;     
    int h = 600;     
    int r = 20;     

    initgraph(w, h);

    while (1) {
        cleardevice();  

        vy = vy + g;
        y = y + vy;

        if (y >= h - r) {
            vy = -0.95 * vy; 
            y = h - r;       
        }

        

       
        fillcircle(300, (int)y, r);

       
        Sleep(1); 
    }

    _getch();
    closegraph();
    return 0;
}