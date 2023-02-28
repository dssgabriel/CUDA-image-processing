#include "config.hpp"
#include "driver.cuh"
#include "kernels.cuh"
#include "utils.hpp"

#include "FreeImage.h"

#include <cstdint>
#include <cuda.h>
#include <iostream>
#include <vector>

auto main(int32_t argc, char** argv) -> int32_t {
    Config cfg = Config::parse_args(argc, argv);
    if (!cfg.quiet) {
        cfg.display();
    }

    FreeImage_Initialise();

    // Load and decode a regular file
    FREE_IMAGE_FORMAT fif = FreeImage_GetFileType(cfg.input_filename.c_str());
    FIBITMAP* bitmap = FreeImage_Load(fif, cfg.input_filename.c_str(), 0);
    if (!bitmap) { return EXIT_FAILURE; }
    if (!cfg.quiet) {
        fprintf(
            stderr,
            "\033[1;34m==>\033[0m Successfully loaded \033[32m`%s`\033[0m\n",
            cfg.input_filename.c_str()
        );
    }

    // Get image dimensions
    size_t width = FreeImage_GetWidth(bitmap);
    size_t height = FreeImage_GetHeight(bitmap);
    size_t pitch = FreeImage_GetPitch(bitmap);
    if (!cfg.quiet) {
        fprintf(
            stderr,
            "\033[1;34m==>\033[0m Processing image of size \033[34m%zu\033[0mx\033[34m%zu\033[0m\n",
            width, height
        );
    }

    int32_t ret = 0;
    std::vector<uint8_t> img = utils::load_image(bitmap, width, height, pitch);

    ret = driver::run_on_target(cfg, bitmap, img, width, height, pitch);
    if (ret != 0) { return EXIT_FAILURE; }

    ret = utils::store_image(img, bitmap, width, height, pitch);
    if (ret != 0) { return EXIT_FAILURE; }

    FreeImage_Save(fif, bitmap, cfg.output_filename.c_str(), 0);
    if (!cfg.quiet) {
        fprintf(
            stderr,
            "\033[1;34m==>\033[0m Successfully saved \033[32m`%s`\033[0m\n",
            cfg.output_filename.c_str()
        );
    }

    FreeImage_DeInitialise();

    return EXIT_SUCCESS;
}
