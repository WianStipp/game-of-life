#include <stdio.h>
#include <time.h>

#define GRID_SIZE 32
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
void fast_update_next_state(int a[][GRID_SIZE], int next_state[][GRID_SIZE]) {
    int x = threadIdx.x;
    int y = threadIdx.y;
    int num_neighbors;

    num_neighbors = count_neighbors(a, x, y);
    if (num_neighbors < 2 || num_neighbors > 3) {
        next_state[x][y] = 0;
    }
    else if (num_neighbors == 3) {
       next_state[x][y] = 1;
    }
    else {
        next_state[x][y] = a[x][y];
    }
    __syncthreads();
    a[x][y] = next_state[x][y];
}

__global__
void update_next_state(int a[][GRID_SIZE], int next_state[][GRID_SIZE]) {
    int x = threadIdx.x;
    int y = threadIdx.y;
    int num_neighbors;
    num_neighbors = count_neighbors(a, x, y);
    if (num_neighbors < 2 || num_neighbors > 3) {
        next_state[x][y] = 0;
    }
    else if (num_neighbors == 3) {
        next_state[x][y] = 1;
    }
    else {
        next_state[x][y] = a[x][y];
    }
}


void set_blinker(int state[GRID_SIZE][GRID_SIZE]) {
    state[5][4] = 1;
    state[5][5] = 1;
    state[5][6] = 1;
}

void set_interesting_state(int state[GRID_SIZE][GRID_SIZE]) {
    state[5][5] = 1;
    state[5][6] = 1;
    state[6][5] = 1;
    state[5][7] = 1;
    state[4][6] = 1;
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

__global__
void device_set_state_as_next(int a[][GRID_SIZE], int b[][GRID_SIZE]) {
    int x = threadIdx.x;
    int y = threadIdx.y;
    a[x][y] = b[x][y];
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
    float start_time, end_time;
    int state[GRID_SIZE][GRID_SIZE];
    int next_state[GRID_SIZE][GRID_SIZE];
    dim3 grid(GRID_SIZE, GRID_SIZE);
    int (*dev_state)[GRID_SIZE], (*dev_next_state)[GRID_SIZE];

    set_blank_state(state);
    // set_blinker(state);
    set_interesting_state(state);
    render_state(state);
    cudaMalloc((void**) &dev_state, sizeof(int)*GRID_SIZE*GRID_SIZE);
    cudaMalloc((void**) &dev_next_state, sizeof(int)*GRID_SIZE*GRID_SIZE);

    cudaMemcpy(dev_state, state, sizeof(int)*GRID_SIZE*GRID_SIZE, cudaMemcpyHostToDevice);
    start_time = (float)clock()/CLOCKS_PER_SEC;
    for (i=0;i<100001;i++) {
        fast_update_next_state<<<1,grid>>>(dev_state, dev_next_state);
        cudaMemcpyAsync(next_state, dev_next_state, sizeof(int)*GRID_SIZE*GRID_SIZE, cudaMemcpyDeviceToHost);
        // render_state(next_state);
        // device_set_state_as_next<<<1,grid>>>(dev_state, dev_next_state);
    }
    cudaMemcpy(next_state, dev_next_state, sizeof(int)*GRID_SIZE*GRID_SIZE, cudaMemcpyDeviceToHost);
    end_time = (float)clock()/CLOCKS_PER_SEC;
    render_state(next_state);
    cudaFree(dev_state);
    cudaFree(dev_next_state);
    printf("Simulation took %f seconds\n", end_time-start_time);
    return 0;
}