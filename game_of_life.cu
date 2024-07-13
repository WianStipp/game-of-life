#include <stdio.h>

#define NUM_ROWS 10
#define NUM_COLS 10
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

__global__
void add(int a, int b, int *c) {
    *c = a + b;
}

int cuda_test(void) {
    int c;
    int *dev_c;
    cudaMalloc( (void**) &dev_c, sizeof(int) );
    add<<<1,1>>>(2, 7, dev_c);
    // cudaDeviceSynchronize();
    cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
    printf("2+7=%d\n", c);
    cudaFree(dev_c);
    return 0;
}


int count_neighbors(int a[NUM_ROWS][NUM_COLS], int x, int y) {
    int i,j,sum;
    for (i=-1;i<2;i++) {
        for (j=-1;j<2;j++) {
            sum = sum + a[MIN(NUM_ROWS,MAX(0,x+i))][MIN(NUM_COLS,MAX(0,y+j))];
        }
    }
    return sum - a[x][y];
}

void update_state(int a[NUM_ROWS][NUM_COLS]) {
    int next_state[NUM_ROWS][NUM_COLS];
    int x, y;
    for (x=0;x<NUM_ROWS;x++) {
        for (y=0;y<NUM_COLS;y++) {
            printf("Num neighbors: %d\n", count_neighbors(a, x, y));
        }
    }
}


void set_blinker(int a[NUM_ROWS][NUM_COLS]) {
    a[5][4] = 1;
    a[5][5] = 1;
    a[5][6] = 1;
}

void set_blank_state(int a[NUM_ROWS][NUM_COLS]) {
    int i,j;
    for (i=0;i<NUM_ROWS;i++) {
        for (j=0;j<NUM_COLS;j++) {
            a[i][j] = 0;
        }
    }
}

void render_state(int a[NUM_ROWS][NUM_COLS]) {
    int i,j;
    for (i=0;i<NUM_ROWS;i++) {
        for (j=0;j<NUM_COLS;j++) {
            printf("%d   ", a[i][j]);
        }
        printf("\n");
    }
}


int main(void) {
    int state[NUM_ROWS][NUM_COLS];

    set_blank_state(state);
    // set_blinker(state);
    render_state(state);
    update_state(state);
    return 0;
}