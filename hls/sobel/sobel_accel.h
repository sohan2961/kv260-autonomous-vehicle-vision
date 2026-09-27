#ifndef SOBEL_ACCEL_H
#define SOBEL_ACCEL_H

#define MAX_WIDTH   1920
#define MAX_HEIGHT  1080
#define MAX_PIXELS  (MAX_WIDTH * MAX_HEIGHT)

void sobel_accel(
    const unsigned char *input,
    unsigned char *output,
    int width,
    int height
);

#endif
