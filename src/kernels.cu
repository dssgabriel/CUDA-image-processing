#include "kernels.cuh"

#include <omp.h>

namespace kernels {
    namespace host {
        auto saturate_color(
            uint8_t* img,
            size_t width,
            size_t height,
            ColorKind color
        ) -> void {
            #pragma omp parallel for collapse(2)
            for (size_t y = 0; y < height; ++y) {
                for (size_t x = 0; x < width; ++x) {
                    size_t idx = (y * width + x) * 3;
                    img[idx + color] = INT8_MAX + INT8_MAX / 2;
                }
            }
        }

        auto flip_horizontally(
            uint8_t* img,
            uint8_t const* tmp,
            size_t width,
            size_t height
        ) -> void {
            #pragma omp parallel for collapse(2)
            for (size_t y = 0; y < height; ++y) {
                for (size_t x = 0; x < width; ++x) {
                    size_t ida = (y * width + x) * 3;
                    size_t idb = (width * height - (y * width + x)) * 3;
                    img[ida] = tmp[idb];
                    img[ida + 1] = tmp[idb + 1];
                    img[ida + 2] = tmp[idb + 2];
                }
            }
        }

        auto grayscale(uint8_t* img, size_t width, size_t height) -> void {
            #pragma omp parallel for collapse(2)
            for (size_t y = 0; y < height; ++y) {
                for (size_t x = 0; x < width; ++x) {
                    size_t idx = (y * width + x) * 3;

                    size_t gray = 
                        (uint8_t)(0.299F * (float)(img[idx]))
                        + (uint8_t)(0.587F * (float)(img[idx + 1]))
                        + (uint8_t)(0.114F * (float)(img[idx + 2]));

                    img[idx] = gray;
                    img[idx + 1] = gray;
                    img[idx + 2] = gray;
                }
            }
        }

        auto blur(uint8_t* img, size_t width, size_t height) -> void {
            #pragma omp parallel for collapse(2)
            for (size_t y = 0; y < height; ++y) {
                for (size_t x = 0; x < width; ++x) {
                    size_t idx = (y * width + x) * 3;
                    size_t idl = (y * width + x - 1) * 3;
                    size_t idr = (y * width + x + 1) * 3;
                    size_t idu = ((y - 1) * width + x) * 3;
                    size_t idd = ((y + 1) * width + x) * 3;

                    uint16_t r_mean =
                        img[idx] + img[idl] + img[idr] + img[idu] + img[idd];
                    uint16_t g_mean =
                        img[idx + 1] + img[idl + 1] + img[idr + 1] + img[idu + 1] + img[idd + 1];
                    uint16_t b_mean =
                        img[idx + 2] + img[idl + 2] + img[idr + 2] + img[idu + 2] + img[idd + 2];

                    img[idx] = (uint8_t)(r_mean / 5);
                    img[idx + 1] = (uint8_t)(g_mean / 5);
                    img[idx + 2] = (uint8_t)(b_mean / 5);
                }
            }
        }

        auto iterative_blur(uint8_t* img, size_t width, size_t height, size_t iterations) -> void {
            for (size_t _ = 0; _ < iterations; _++) {
                blur(img, width, height);
            }
        }

        auto sobel(
            uint8_t* img,
            uint8_t const* tmp,
            size_t width,
            size_t height,
            uint8_t threshold
        ) -> void {
            uint32_t threshold_sq = threshold * threshold;

            #pragma omp parallel for collapse(2)
            for (size_t y = 0; y < height; ++y) {
                for (size_t x = 0; x < width; ++x) {
                    size_t idx = (y * width + x) * 3;
                    size_t idl = (y * width + (x - 1)) * 3;
                    size_t idr = (y * width + (x + 1)) * 3;
                    size_t idu = ((y - 1) * width + x) * 3;
                    size_t idd = ((y + 1) * width + x) * 3;
                    size_t idul = ((y - 1) * width + (x - 1)) * 3;
                    size_t idur = ((y - 1) * width + (x + 1)) * 3;
                    size_t iddl = ((y + 1) * width + (x - 1)) * 3;
                    size_t iddr = ((y + 1) * width + (x + 1)) * 3;

                    int16_t gy = -tmp[idul] + tmp[idur] - 2 * tmp[idl] +
                                 2 * tmp[idr] - tmp[iddl] + tmp[iddr];
                    int16_t gx = -tmp[idul] - 2 * tmp[idu] - tmp[idur] +
                                 tmp[iddl] + 2 * tmp[idd] + tmp[iddr];

                    uint32_t mag = (gy * gy + gx * gx) / threshold_sq * UINT8_MAX;
                    img[idx] = mag;
                    img[idx + 1] = mag;
                    img[idx + 2] = mag;
                }
            }
        }

        auto diapositive(uint8_t* img, size_t height, size_t width) -> void {
            #pragma omp parallel for collapse(2)
            for (size_t y = 0; y < height; ++y) {
                for (size_t x = 0; x < width; ++x) {
                    size_t idx = (y * width + x) * 3;
                    if (idx < height * width * 3) {
                        img[idx] = UINT8_MAX - img[idx];
                        img[idx + 1] = UINT8_MAX - img[idx + 1];
                        img[idx + 2] = UINT8_MAX - img[idx + 2];
                    }
                }
            }
        }
    } // namespace host

    namespace device {
        __global__
        auto saturate_color(uint8_t* img, size_t width, size_t height, ColorKind color) -> void {
            size_t y = blockIdx.y * blockDim.y + threadIdx.y;
            size_t x = blockIdx.x * blockDim.x + threadIdx.x;
            size_t idx = (y * width + x) * 3;

            if (idx < height * width * 3) {
                img[idx + color] = INT8_MAX + INT8_MAX / 2;
            }
        }

        __global__
        auto grayscale(uint8_t* img, size_t width, size_t height) -> void {
            size_t y = blockIdx.y * blockDim.y + threadIdx.y;
            size_t x = blockIdx.x * blockDim.x + threadIdx.x;
            size_t idx = (y * width + x) * 3;

            if (idx < height * width * 3) {
                size_t gray = 
                    (uint8_t)(0.299F * (float)(img[idx]))
                    + (uint8_t)(0.587F * (float)(img[idx + 1]))
                    + (uint8_t)(0.114F * (float)(img[idx + 2]));

                img[idx] = gray;
                img[idx + 1] = gray;
                img[idx + 2] = gray;
            }
        }

        __global__
        auto flip_horizontally(
            uint8_t* img,
            uint8_t const* tmp,
            size_t width,
            size_t height
        ) -> void {
            size_t y = blockIdx.y * blockDim.y + threadIdx.y;
            size_t x = blockIdx.x * blockDim.x + threadIdx.x;
            size_t ida = (y * width + x) * 3;
            size_t idb = (width * height - (y * width + x)) * 3;

            if (ida < height * width * 3 && idb < height * width * 3) {
                img[ida] = tmp[idb];
                img[ida + 1] = tmp[idb + 1];
                img[ida + 2] = tmp[idb + 2];
            }
        }

        __global__
        auto blur(uint8_t* img, size_t width, size_t height) -> void {
            size_t y = blockIdx.y * blockDim.y + threadIdx.y;
            size_t x = blockIdx.x * blockDim.x + threadIdx.x;
            size_t idx = (y * width + x) * 3;
            size_t idl = (y * width + x - 1) * 3;
            size_t idr = (y * width + x + 1) * 3;
            size_t idu = ((y - 1) * width + x) * 3;
            size_t idd = ((y + 1) * width + x) * 3;
            size_t count = 1;
            size_t size = height * width * 3;

            if (idx < size) {
                uint16_t r_mean = img[idx];
                uint16_t g_mean = img[idx + 1];
                uint16_t b_mean = img[idx + 2];

                if (idl < size) {
                    r_mean += img[idl];
                    g_mean += img[idl + 1];
                    b_mean += img[idl + 2];
                    ++count;
                }
                if (idr < size) {
                    r_mean += img[idr];
                    g_mean += img[idr + 1];
                    b_mean += img[idr + 2];
                    ++count;
                }
                if (idu < size) {
                    r_mean += img[idu];
                    g_mean += img[idu + 1];
                    b_mean += img[idu + 2];
                    ++count;
                }
                if (idd < size) {
                    r_mean += img[idd];
                    g_mean += img[idd + 1];
                    b_mean += img[idd + 2];
                    ++count;
                }

                img[idx] = (uint8_t)(r_mean / count);
                img[idx + 1] = (uint8_t)(g_mean / count);
                img[idx + 2] = (uint8_t)(b_mean / count);
            }
        }

        __global__
        auto sobel(
            uint8_t* img,
            uint8_t const* tmp,
            size_t width,
            size_t height,
            uint8_t threshold
        ) -> void {
            uint32_t threshold_sq = threshold * threshold;

            size_t y = blockIdx.y * blockDim.y + threadIdx.y;
            size_t x = blockIdx.x * blockDim.x + threadIdx.x;
            size_t idx = (y * width + x) * 3;
            size_t idl = (y * width + (x - 1)) * 3;
            size_t idr = (y * width + (x + 1)) * 3;
            size_t idu = ((y - 1) * width + x) * 3;
            size_t idd = ((y + 1) * width + x) * 3;
            size_t idul = ((y - 1) * width + (x - 1)) * 3;
            size_t idur = ((y - 1) * width + (x + 1)) * 3;
            size_t iddl = ((y + 1) * width + (x - 1)) * 3;
            size_t iddr = ((y + 1) * width + (x + 1)) * 3;

            size_t size = height * width * 3;
            if (idx < size) {
                int16_t gy = 0;
                int16_t gx = 0;
                if (idul < size) {
                    gy -= tmp[idul];
                    gx -= tmp[idul];
                }
                if (idur < size) {
                    gy += tmp[idur];
                    gx -= tmp[idur];
                }
                if (iddl < size) {
                    gy -= tmp[iddl];
                    gx += tmp[iddl];
                }
                if (iddr < size) {
                    gy += tmp[iddr];
                    gx += tmp[iddr];
                }
                if (idl < size) {
                    gy -= 2 * tmp[idl];
                }
                if (idr < size) {
                    gy += 2 * tmp[idr];
                }
                if (idu < size) {
                    gx -= 2 * tmp[idu];
                }
                if (idd < size) {
                    gx += 2 * tmp[idd];
                }

                uint32_t mag = (gy * gy + gx * gx) / threshold_sq * UINT8_MAX;
                img[idx] = mag;
                img[idx + 1] = mag;
                img[idx + 2] = mag;
            }
        }

        __global__
        auto diapositive(uint8_t* img, size_t height, size_t width) -> void {
            size_t y = blockIdx.y * blockDim.y + threadIdx.y;
            size_t x = blockIdx.x * blockDim.x + threadIdx.x;
            size_t idx = (y * width + x) * 3;
            if (idx < height * width * 3) {
                img[idx] = UINT8_MAX - img[idx];
                img[idx + 1] = UINT8_MAX - img[idx + 1];
                img[idx + 2] = UINT8_MAX - img[idx + 2];
            }
        }
    } // namespace device
} // namespace kernels
