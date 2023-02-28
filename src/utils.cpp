#include "utils.hpp"

#include <cstdio>

namespace utils {
    auto load_image(
        FIBITMAP* bitmap,
        size_t width,
        size_t height,
        size_t pitch
    ) -> std::vector<uint8_t> {
        std::vector<uint8_t> img(3 * width * height * sizeof(uint8_t));

        BYTE* bits = (BYTE*)(FreeImage_GetBits(bitmap));
        for (size_t y = 0; y < height; ++y) {
            BYTE* pixel = (BYTE*)(bits);
            for (size_t x = 0; x < width; ++x) {
                size_t idx = (y * width + x) * 3;
                img[idx + 0] = pixel[FI_RGBA_RED];
                img[idx + 1] = pixel[FI_RGBA_GREEN];
                img[idx + 2] = pixel[FI_RGBA_BLUE];
                pixel += 3;
            }
            bits += pitch;
        }

        return img;
    }

    auto store_image(
        std::vector<uint8_t> const& img,
        FIBITMAP* bitmap,
        size_t width,
        size_t height,
        size_t pitch
    ) -> int32_t {
        int32_t ret = 0;

        BYTE* bits = (BYTE*)(FreeImage_GetBits(bitmap));
        for (size_t y = 0; y < height; y++) {
            BYTE* pixel = (BYTE*)bits;
            for (size_t x = 0; x < width; x++) {
                RGBQUAD newcolor;

                size_t idx = (y * width + x) * 3;
                newcolor.rgbRed = img[idx + 0];
                newcolor.rgbGreen = img[idx + 1];
                newcolor.rgbBlue = img[idx + 2];

                if (!FreeImage_SetPixelColor(bitmap, x, y, &newcolor)) {
                    ret = -1;
                    fprintf(
                        stderr,
                        "\033[1;31merror:\033[0m failed to read set pixel color at (%zu, %zu)\n",
                        x,
                        y
                    );
                }

                pixel += 3;
            }
            bits += pitch;
        }

        return ret;
    }
    
    auto imgcpy(
        uint8_t* output,
        uint8_t const* input,
        size_t dim,
        size_t idx_output, 
        size_t idx_input
    ) -> void {
        for (size_t x = 0; x < dim; ++x) {
            output[idx_output + x * 3 + 0] = input[idx_input + x * 3 + 0];
            output[idx_output + x * 3 + 1] = input[idx_input + x * 3 + 1];
            output[idx_output + x * 3 + 2] = input[idx_input + x * 3 + 2];
        }
    }
}

