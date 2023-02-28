#pragma once

#include "config.hpp"

#include "FreeImage.h"

#include <cstdint>
#include <vector>

namespace driver {
    auto run_on_target(
        Config const& cfg,
        FIBITMAP* bitmap,
        std::vector<uint8_t>& img,
        size_t width,
        size_t height,
        size_t pitch
    ) -> int32_t;

    namespace host {
        auto run(
            Config const& cfg,
            std::vector<uint8_t>& img,
            size_t width,
            size_t height,
            size_t pitch
        ) -> int32_t;
    }

    namespace device {
        auto run(
            Config const& cfg,
            FIBITMAP* bitmap,
            std::vector<uint8_t>& img,
            size_t width,
            size_t height,
            size_t pitch
        ) -> int32_t;
    }
}
