#include "sobel_accel.h"


/*
 * Simple absolute-value function.
 * HLS will synthesize this into hardware logic.
 */
static int abs_int(int value)
{
    return (value < 0) ? -value : value;
}


/*
 * ================================================================
 *  Full-HD Sobel Edge Detection Accelerator
 *
 *  Maximum resolution:
 *      1920 x 1080
 *
 *  Architecture:
 *      DDR input
 *          |
 *          v
 *      Two line buffers
 *          |
 *          v
 *      Sliding 3x3 window
 *          |
 *          v
 *      Sobel Gx / Gy
 *          |
 *          v
 *      DDR output
 *
 *  Target:
 *      AMD Kria KV260 / K26
 *
 *  Goal:
 *      Main processing loop II = 1
 * ================================================================
 */

void sobel_accel(
    const unsigned char *input,
    unsigned char *output,
    int width,
    int height
)
{

    /*
     * ============================================================
     * AXI MASTER INTERFACES
     *
     * input:
     *      Reads image pixels from external DDR.
     *
     * output:
     *      Writes Sobel edge image to external DDR.
     *
     * Maximum number of pixels:
     *
     *      1920 x 1080 = 2,073,600
     * ============================================================
     */

#pragma HLS INTERFACE m_axi port=input  offset=slave bundle=gmem0 depth=2073600
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem1 depth=2073600


    /*
     * ============================================================
     * AXI-LITE CONTROL INTERFACE
     *
     * ARM processor will later control:
     *
     * input address
     * output address
     * width
     * height
     * start/done
     * ============================================================
     */

#pragma HLS INTERFACE s_axilite port=input  bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=width  bundle=control
#pragma HLS INTERFACE s_axilite port=height bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control


    /*
     * ============================================================
     * INPUT VALIDATION
     * ============================================================
     */

    if (width <= 0 || height <= 0)
    {
        return;
    }

    if (width > MAX_WIDTH || height > MAX_HEIGHT)
    {
        return;
    }


    /*
     * ============================================================
     * LINE BUFFERS
     *
     * We do NOT store the full 1920x1080 frame inside FPGA BRAM.
     *
     * Instead:
     *
     * line0 = older image row
     * line1 = previous image row
     *
     * Each line contains maximum 1920 pixels.
     *
     * Since each pixel is 8-bit:
     *
     *      line0 = 1920 bytes
     *      line1 = 1920 bytes
     *
     * Total line-buffer data:
     *
     *      3840 bytes
     *
     * This dramatically reduces repeated DDR accesses.
     * ============================================================
     */

    unsigned char line0[MAX_WIDTH];
    unsigned char line1[MAX_WIDTH];


    /*
     * Force the line buffers into dual-port BRAM.
     */

#pragma HLS BIND_STORAGE variable=line0 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=line1 type=ram_2p impl=bram


    /*
     * ============================================================
     * SMALL IMAGE CASE
     *
     * Sobel requires a 3x3 window.
     *
     * If width or height is smaller than 3,
     * there is no valid Sobel pixel.
     *
     * Therefore output is zero.
     * ============================================================
     */

    if (width < 3 || height < 3)
    {

SMALL_IMAGE_LOOP:

        for (int i = 0; i < width * height; i++)
        {
#pragma HLS PIPELINE II=1

            output[i] = 0;
        }

        return;
    }


    /*
     * ============================================================
     * STEP 1
     *
     * LOAD FIRST IMAGE ROW
     *
     * Important optimization:
     *
     * This is separate from LOAD_ROW1.
     *
     * Previously we tried:
     *
     * line0[x] = input[x];
     * line1[x] = input[width + x];
     *
     * inside one loop iteration.
     *
     * That required TWO reads from gmem0 in one cycle and caused:
     *
     * HLS 200-885
     * II Violation
     *
     * Now each loop requests only ONE AXI read per iteration.
     * ============================================================
     */

LOAD_ROW0:

    for (int x = 0; x < width; x++)
    {
#pragma HLS PIPELINE II=1

        line0[x] = input[x];
    }


    /*
     * ============================================================
     * STEP 2
     *
     * LOAD SECOND IMAGE ROW
     * ============================================================
     */

LOAD_ROW1:

    for (int x = 0; x < width; x++)
    {
#pragma HLS PIPELINE II=1

        line1[x] = input[width + x];
    }


    /*
     * ============================================================
     * STEP 3
     *
     * TOP BORDER
     *
     * Sobel cannot calculate the first image row because a 3x3
     * window would require pixels above the image.
     *
     * Therefore row 0 = zero.
     * ============================================================
     */

TOP_BORDER:

    for (int x = 0; x < width; x++)
    {
#pragma HLS PIPELINE II=1

        output[x] = 0;
    }


    /*
     * ============================================================
     * STEP 4
     *
     * MAIN IMAGE PROCESSING
     *
     * We start reading at input row 2.
     *
     * At any time:
     *
     * line0 = row y-2
     * line1 = row y-1
     * DDR   = row y
     *
     * Together these provide the three rows required
     * for the Sobel 3x3 kernel.
     * ============================================================
     */

ROW_LOOP:

    for (int y = 2; y < height; y++)
    {

        /*
         * ========================================================
         * SLIDING 3x3 WINDOW
         *
         * Conceptually:
         *
         *       win00  win01  win02
         *       win10  win11  win12
         *       win20  win21  win22
         *
         * The left and middle columns are kept in registers.
         *
         * The newest right-hand column comes from:
         *
         * win02 <- line0
         * win12 <- line1
         * win22 <- current input pixel
         * ========================================================
         */

        unsigned char win00 = 0;
        unsigned char win01 = 0;

        unsigned char win10 = 0;
        unsigned char win11 = 0;

        unsigned char win20 = 0;
        unsigned char win21 = 0;


COLUMN_LOOP:

        for (int x = 0; x < width; x++)
        {

            /*
             * Main performance target.
             *
             * We want to start processing a new pixel
             * every clock cycle.
             */

#pragma HLS PIPELINE II=1


            /*
             * ====================================================
             * READ ONE NEW PIXEL FROM DDR
             *
             * Unlike V1, we are not requesting eight neighboring
             * pixels from DDR.
             *
             * Only one new pixel is needed.
             * ====================================================
             */

            unsigned char new_pixel =
                input[y * width + x];


            /*
             * ====================================================
             * BUILD NEW RIGHT-HAND WINDOW COLUMN
             * ====================================================
             */

            unsigned char win02 = line0[x];

            unsigned char win12 = line1[x];

            unsigned char win22 = new_pixel;


            /*
             * ====================================================
             * A complete 3x3 window becomes available
             * when x >= 2.
             * ====================================================
             */

            if (x >= 2)
            {

                /*
                 * =================================================
                 * SOBEL X GRADIENT
                 *
                 * Kernel:
                 *
                 *      -1   0  +1
                 *      -2   0  +2
                 *      -1   0  +1
                 * =================================================
                 */

                int gx =
                    -(int)win00
                    +(int)win02

                    -2 * (int)win10
                    +2 * (int)win12

                    -(int)win20
                    +(int)win22;


                /*
                 * =================================================
                 * SOBEL Y GRADIENT
                 *
                 * Kernel:
                 *
                 *      -1  -2  -1
                 *       0   0   0
                 *      +1  +2  +1
                 * =================================================
                 */

                int gy =
                    -(int)win00
                    -2 * (int)win01
                    -(int)win02

                    +(int)win20
                    +2 * (int)win21
                    +(int)win22;


                /*
                 * =================================================
                 * GRADIENT MAGNITUDE
                 *
                 * Instead of:
                 *
                 * sqrt(Gx*Gx + Gy*Gy)
                 *
                 * we use:
                 *
                 * |Gx| + |Gy|
                 *
                 * This is much cheaper to synthesize.
                 * =================================================
                 */

                int magnitude =
                    abs_int(gx) +
                    abs_int(gy);


                /*
                 * =================================================
                 * SATURATE RESULT TO 8-BIT
                 * =================================================
                 */

                if (magnitude > 255)
                {
                    magnitude = 255;
                }


                /*
                 * =================================================
                 * OUTPUT POSITION
                 *
                 * Current 3x3 window:
                 *
                 * rows:
                 *      y-2
                 *      y-1   <- center
                 *      y
                 *
                 * columns:
                 *      x-2
                 *      x-1   <- center
                 *      x
                 *
                 * Therefore:
                 *
                 * output_x = x - 1
                 * output_y = y - 1
                 * =================================================
                 */

                int output_x = x - 1;

                int output_y = y - 1;


                /*
                 * Write Sobel result.
                 */

                output[
                    output_y * width +
                    output_x
                ] =
                    (unsigned char)magnitude;
            }


            /*
             * ====================================================
             * SLIDE WINDOW HORIZONTALLY
             *
             * Before:
             *
             * A B C
             *
             * Next cycle:
             *
             * B C NEW
             * ====================================================
             */

            win00 = win01;
            win01 = win02;

            win10 = win11;
            win11 = win12;

            win20 = win21;
            win21 = win22;


            /*
             * ====================================================
             * UPDATE VERTICAL LINE BUFFERS
             *
             * Before:
             *
             * line0 = y-2
             * line1 = y-1
             *
             * After:
             *
             * line0 = y-1
             * line1 = y
             *
             * The next image row can therefore reuse these pixels.
             * ====================================================
             */

            line0[x] = line1[x];

            line1[x] = new_pixel;
        }


        /*
         * ========================================================
         * LEFT BORDER
         *
         * Column zero has no valid 3x3 neighborhood.
         * ========================================================
         */

        output[
            (y - 1) * width
        ] = 0;


        /*
         * ========================================================
         * RIGHT BORDER
         * ========================================================
         */

        output[
            (y - 1) * width +
            (width - 1)
        ] = 0;
    }


    /*
     * ============================================================
     * STEP 5
     *
     * BOTTOM BORDER
     *
     * Last row cannot have a valid 3x3 neighborhood.
     * ============================================================
     */

BOTTOM_BORDER:

    for (int x = 0; x < width; x++)
    {
#pragma HLS PIPELINE II=1

        output[
            (height - 1) * width +
            x
        ] = 0;
    }
}
