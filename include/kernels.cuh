#pragma once

#include "config.hpp"

#include <cstdint>
#include <cuda.h>
#include <vector>

namespace kernels {
    namespace host {
        auto saturate_color(
            uint8_t* img,
            size_t width,
            size_t height,
            ColorKind color
        ) -> void;

        auto flip_horizontally(
            uint8_t* img,
            uint8_t const* tmp,
            size_t width,
            size_t height
        ) -> void;

        auto grayscale(uint8_t* img, size_t width, size_t height) -> void;

        auto blur(uint8_t* img, size_t width, size_t height) -> void;

        auto iterative_blur(
            uint8_t* img,
            size_t width,
            size_t height,
            size_t iterations
        ) -> void;

        auto sobel(
            uint8_t* img,
            uint8_t const* tmp,
            size_t width,
            size_t height,
            uint8_t threshold
        ) -> void;

        auto diapositive(uint8_t* img, size_t height, size_t width) -> void;
    } // namespace host

    namespace device {
        __global__
        auto saturate_color(uint8_t* img, size_t width, size_t height, ColorKind color) -> void;

        __global__
        auto grayscale(uint8_t* img, size_t width, size_t height) -> void;

        __global__
        auto flip_horizontally(
            uint8_t* img,
            uint8_t const* tmp,
            size_t width,
            size_t height
        ) -> void;

        __global__
        auto blur(uint8_t* img, size_t width, size_t height) -> void;

        __global__
        auto sobel(
            uint8_t* img,
            uint8_t const* tmp,
            size_t width,
            size_t height,
            uint8_t threshold
        ) -> void;

        __global__
        auto diapositive(uint8_t* img, size_t width, size_t height) -> void;
    } // namespace device
} // namespace kernels
