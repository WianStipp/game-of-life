#include <stdio.h>

#define GRID_SIZE 10
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

__device__
int count_neighbors(int a[GRID_SIZE][GRID_SIZE], int x, int y) {
    int i,j;
    int sum = 0;
    for (i=-1;i<2;i++) {
        for (j=-1;j<2;j++) {
            if ((x+i >= 0) && (x+i < GRID_SIZE) && (y+j >= 0) && (y+j<GRID_SIZE)) {
                sum = sum + a[x+i][y+j];
            }
        }
    }
    return sum - a[x][y];
}

__global__
void update_next_state(int a[][GRID_SIZE], int next_state[][GRID_SIZE]) {
    int x = threadIdx.x;
    int y = threadIdx.y;
    int num_neighbors;
    num_neighbors = count_neighbors(a, x, y);
    if (num_neighbors < 2) {
        next_state[x][y] = 0;
    }
    else if (num_neighbors == 3) {
        next_state[x][y] = 1;
    }
    else {
        next_state[x][y] = a[x][y];
    }
}


void set_blinker(int a[GRID_SIZE][GRID_SIZE]) {
    a[5][4] = 1;
    a[5][5] = 1;
    a[5][6] = 1;
}

void set_blank_state(int a[GRID_SIZE][GRID_SIZE]) {
    int i,j;
    for (i=0;i<GRID_SIZE;i++) {
        for (j=0;j<GRID_SIZE;j++) {
            a[i][j] = 0;
        }
    }
}

void render_state(int a[GRID_SIZE][GRID_SIZE]) {
    int i,j;
    for (i=0;i<GRID_SIZE;i++) {
        for (j=0;j<GRID_SIZE;j++) {
            printf("%d   ", a[i][j]);
        }
        printf("\n");
    }
}

void set_state_as_next(int a[GRID_SIZE][GRID_SIZE], int b[GRID_SIZE][GRID_SIZE]) {
    int i,j;
    for (i=0;i<GRID_SIZE;i++) {
        for (j=0;j<GRID_SIZE;j++) {
            a[i][j] = b[i][j];
        }
    }
}


int main(void) {
    int i;
    int state[GRID_SIZE][GRID_SIZE];
    int next_state[GRID_SIZE][GRID_SIZE];
    dim3 grid(GRID_SIZE, GRID_SIZE);
    int (*dev_state)[GRID_SIZE], (*dev_next_state)[GRID_SIZE];

    set_blank_state(state);
    set_blinker(state);
    render_state(state);
    cudaMalloc((void**) &dev_state, sizeof(int)*GRID_SIZE*GRID_SIZE);
    cudaMalloc((void**) &dev_next_state, sizeof(int)*GRID_SIZE*GRID_SIZE);

    cudaMemcpy(dev_state, state, sizeof(int)*GRID_SIZE*GRID_SIZE, cudaMemcpyHostToDevice);
    update_next_state<<<1,grid>>>(dev_state, dev_next_state);

    cudaMemcpy(next_state, dev_next_state, sizeof(int)*GRID_SIZE*GRID_SIZE, cudaMemcpyDeviceToHost);
    render_state(next_state);
    cudaFree(dev_state);
    cudaFree(dev_next_state);
    return 0;
}