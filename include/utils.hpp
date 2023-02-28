#pragma once

#include "FreeImage.h"

#include <cstdint>
#include <vector>

namespace utils {
    auto load_image(
        FIBITMAP* bitmap,
        size_t width,
        size_t height,
        size_t pitch
    ) -> std::vector<uint8_t>;

    auto store_image(
        std::vector<uint8_t> const& img,
        FIBITMAP* bitmap,
        size_t width,
        size_t height,
        size_t pitch
    ) -> int32_t;

    auto imgcpy(
        uint8_t* output,
        uint8_t const* input,
        size_t dim,
        size_t idx_output, 
        size_t idx_input
    ) -> void;
}
