#include <iostream>
#include <vector>

#include "sobel_accel.h"


static int abs_ref(int value)
{
    return (value < 0) ? -value : value;
}


void sobel_reference(
    const unsigned char *input,
    unsigned char *output,
    int width,
    int height
)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            /*
             * Border pixels.
             */
            if (x == 0 ||
                y == 0 ||
                x == width - 1 ||
                y == height - 1)
            {
                output[y * width + x] = 0;
            }
            else
            {
                int p00 =
                    input[(y - 1) * width + (x - 1)];

                int p01 =
                    input[(y - 1) * width + x];

                int p02 =
                    input[(y - 1) * width + (x + 1)];


                int p10 =
                    input[y * width + (x - 1)];

                int p12 =
                    input[y * width + (x + 1)];


                int p20 =
                    input[(y + 1) * width + (x - 1)];

                int p21 =
                    input[(y + 1) * width + x];

                int p22 =
                    input[(y + 1) * width + (x + 1)];


                int gx =
                    -p00
                    +p02
                    -2 * p10
                    +2 * p12
                    -p20
                    +p22;


                int gy =
                    -p00
                    -2 * p01
                    -p02
                    +p20
                    +2 * p21
                    +p22;


                int magnitude =
                    abs_ref(gx) +
                    abs_ref(gy);


                if (magnitude > 255)
                    magnitude = 255;


                output[y * width + x] =
                    (unsigned char)magnitude;
            }
        }
    }
}


int main()
{
    /*
     * FULL-HD test.
     */
    const int WIDTH  = 1920;
    const int HEIGHT = 1080;

    const int PIXELS = WIDTH * HEIGHT;


    std::cout
        << "Testing Full HD Sobel accelerator"
        << std::endl;

    std::cout
        << "Resolution: "
        << WIDTH
        << " x "
        << HEIGHT
        << std::endl;

    std::cout
        << "Pixels: "
        << PIXELS
        << std::endl;


    /*
     * Allocate complete Full-HD frames.
     */
    std::vector<unsigned char>
        input(PIXELS, 0);

    std::vector<unsigned char>
        hw_output(PIXELS, 0);

    std::vector<unsigned char>
        reference(PIXELS, 0);


    /*
     * ------------------------------------------------------------
     * Generate an artificial road-like image.
     *
     * Background = dark gray.
     * Two bright diagonal lane markings.
     * ------------------------------------------------------------
     */

    for (int y = 0; y < HEIGHT; y++)
    {
        for (int x = 0; x < WIDTH; x++)
        {
            unsigned char pixel = 30;


            /*
             * Left lane.
             */
            int left_lane =
                WIDTH / 2 -
                300 +
                y / 4;


            /*
             * Right lane.
             */
            int right_lane =
                WIDTH / 2 +
                300 -
                y / 4;


            if (x >= left_lane - 4 &&
                x <= left_lane + 4)
            {
                pixel = 220;
            }


            if (x >= right_lane - 4 &&
                x <= right_lane + 4)
            {
                pixel = 220;
            }


            input[
                y * WIDTH + x
            ] = pixel;
        }
    }


    std::cout
        << "Running HLS Sobel..."
        << std::endl;


    /*
     * Hardware-oriented implementation.
     */
    sobel_accel(
        input.data(),
        hw_output.data(),
        WIDTH,
        HEIGHT
    );


    std::cout
        << "Running software reference..."
        << std::endl;


    /*
     * Golden software reference.
     */
    sobel_reference(
        input.data(),
        reference.data(),
        WIDTH,
        HEIGHT
    );


    /*
     * ------------------------------------------------------------
     * Compare every pixel.
     * ------------------------------------------------------------
     */

    int errors = 0;


    for (int i = 0; i < PIXELS; i++)
    {
        if (hw_output[i] != reference[i])
        {
            errors++;


            /*
             * Print only the first 20 errors.
             */
            if (errors <= 20)
            {
                int y = i / WIDTH;
                int x = i % WIDTH;


                std::cout
                    << "Mismatch at x="
                    << x

                    << " y="
                    << y

                    << " HW="
                    << (int)hw_output[i]

                    << " REF="
                    << (int)reference[i]

                    << std::endl;
            }
        }
    }


    if (errors == 0)
    {
        std::cout
            << "FULL HD SOBEL TEST PASSED"
            << std::endl;

        std::cout
            << "1920x1080 processing verified."
            << std::endl;

        return 0;
    }


    std::cout
        << "FULL HD SOBEL TEST FAILED"
        << std::endl;

    std::cout
        << "Total errors = "
        << errors
        << std::endl;


    return 1;
}
