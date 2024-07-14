#include <stdio.h>
#include <time.h>

#define NUM_ROWS 32
#define NUM_COLS 32
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))


int count_neighbors(int a[NUM_ROWS][NUM_COLS], int x, int y) {
    int i,j;
    int sum = 0;
    for (i=-1;i<2;i++) {
        for (j=-1;j<2;j++) {
            if ((x+i >= 0) && (x+i < NUM_ROWS) && (y+j >= 0) && (y+j<NUM_COLS)) {
                sum = sum + a[x+i][y+j];
            }
        }
    }
    return sum - a[x][y];
}

void update_next_state(int a[NUM_ROWS][NUM_COLS], int next_state[NUM_ROWS][NUM_COLS]) {
    int x, y, num_neighbors;
    for (x=0;x<NUM_ROWS;x++) {
        for (y=0;y<NUM_COLS;y++) {
            num_neighbors = count_neighbors(a, x, y);
            if ((num_neighbors < 2) || (num_neighbors > 3)) {
                next_state[x][y] = 0;
            }
            else if (num_neighbors == 3) {
                next_state[x][y] = 1;
            }
            else {
                next_state[x][y] = a[x][y];
            }
        }
    }
}

void set_blinker(int a[NUM_ROWS][NUM_COLS]) {
    a[5][4] = 1;
    a[5][5] = 1;
    a[5][6] = 1;
}

void set_interesting_state(int state[NUM_ROWS][NUM_COLS]) {
    state[5][5] = 1;
    state[5][6] = 1;
    state[6][5] = 1;
    state[5][7] = 1;
    state[4][6] = 1;
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

void set_state_as_next(int a[NUM_ROWS][NUM_COLS], int b[NUM_ROWS][NUM_COLS]) {
    int i,j;
    for (i=0;i<NUM_ROWS;i++) {
        for (j=0;j<NUM_COLS;j++) {
            a[i][j] = b[i][j];
        }
    }
}


int main(void) {
    int state[NUM_ROWS][NUM_COLS];
    int next_state[NUM_ROWS][NUM_COLS];
    int i;
    float start_time, end_time;
    set_blank_state(state);
    // set_blinker(state);
    set_interesting_state(state);
    render_state(state);

    start_time = (float)clock()/CLOCKS_PER_SEC;
    for (i=0;i<100001;i++) {
        update_next_state(state, next_state);
        // render_state(next_state);
        set_state_as_next(state, next_state);
        // printf("\n");
    }
    end_time = (float)clock()/CLOCKS_PER_SEC;
    render_state(next_state);
    printf("Simulation took %f seconds\n", end_time-start_time);
    return 0;
}