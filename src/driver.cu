#include "config.hpp"
#include "driver.cuh"
#include "kernels.cuh"
#include "utils.hpp"

#include <cassert>

static constexpr size_t BLOCK_SIZE = 32;

namespace driver {
    auto run_on_target(
        Config const& cfg,
        FIBITMAP* bitmap,
        std::vector<uint8_t>& img,
        size_t width,
        size_t height,
        size_t pitch
    ) -> int32_t {
        int32_t ret = 0;

        switch (cfg.target) {
            case TargetKind::Host:
                ret = host::run(cfg, img, width, height, pitch);
                break;
            case TargetKind::Device:
                ret = device::run(cfg, bitmap, img, width, height, pitch);
                break;
        }

        return ret;
    }

    namespace host {
        auto run(
            Config const& cfg,
            std::vector<uint8_t>& img,
            size_t width,
            size_t height,
            size_t pitch
        ) -> int32_t {
            int32_t ret = 0;

            size_t size = 3 * width * height * sizeof(uint8_t);

            // Allocate memory for images on host
            auto h_img = (uint8_t*)(malloc(size));
            auto h_tmp = (uint8_t*)(malloc(size));

            // Copy original image on host-allocated images
            memcpy(h_img, img.data(), size);
            memcpy(h_tmp, img.data(), size);

            for (auto const& f: cfg.filters) {
                switch (f.kind) {
                    case FilterKind::Blur:
                        kernels::host::iterative_blur(
                            h_img,
                            width,
                            height,
                            f.associated_val.nb_iterations
                        );
                        break;
                    case FilterKind::Diapositive:
                        kernels::host::diapositive(h_img, width, height);
                        break;
                    case FilterKind::Grayscale:
                        kernels::host::grayscale(h_img, width, height);
                        break;
                    case FilterKind::HorizontalFlip:
                        kernels::host::flip_horizontally(h_img, h_tmp, width, height);
                        break;
                    case FilterKind::PopArt:
                        fprintf(
                            stderr,
                            "\033[1;33mwarning:\033[0m filter `Pop-Art` is not available on "
                            "host/CPU. Skipping...\n"
                        );
                        break;
                    case FilterKind::Saturate:
                        kernels::host::saturate_color(
                            h_img,
                            width,
                            height,
                            f.associated_val.color_to_saturate
                        );
                        break;
                    case FilterKind::Sobel:
                        kernels::host::grayscale(h_img, width, height);
                        memcpy(h_tmp, h_img, size);
                        kernels::host::sobel(
                            h_img,
                            h_tmp,
                            width,
                            height,
                            f.associated_val.threshold
                        );
                        break;
                }
            }

            memcpy(img.data(), h_img, size);

            free(h_img);
            free(h_tmp);

            return ret;
        }
    } // namespace host

    namespace device {
        auto run(
            Config const& cfg,
            FIBITMAP* bitmap,
            std::vector<uint8_t>& img,
            size_t width,
            size_t height,
            size_t pitch
        ) -> int32_t {
            cudaError_t ret = cudaSuccess;

            size_t size = 3 * width * height * sizeof(uint8_t);
            assert(img.size() == size);
            size_t width_split = width / 2;
            size_t height_split = height / 2;
            size_t size_split = 3 * width_split * height_split * sizeof(uint8_t);

            // Allocate memory for images on host
            uint8_t* h_img;
            cudaMallocHost((void**)(&h_img), size);
    
            // Allocate memory for images on device
            uint8_t* d_img;
            cudaMalloc((void**)(&d_img), size);
    
            uint8_t* d_tmp;
            cudaMallocHost((void**)(&d_tmp), size);

            // Copy original image on host-allocated images
            memcpy(h_img, img.data(), size);

            // Copy original image on host-allocated images
            cudaMemcpy(d_img, h_img, size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_tmp, h_img, size, cudaMemcpyHostToDevice);

            dim3 grid(width / BLOCK_SIZE + 1, height / BLOCK_SIZE + 1);
            dim3 block(BLOCK_SIZE, BLOCK_SIZE);
            dim3 grid_popart(
                width_split / BLOCK_SIZE + 1,
                height_split / BLOCK_SIZE + 1
            );

            // Resize image for pop-art
            FIBITMAP* shrank_bitmap = FreeImage_Rescale(
                bitmap,
                width_split,
                height_split,
                FILTER_BOX
            );
            std::vector<uint8_t> tmp = utils::load_image(
                shrank_bitmap,
                FreeImage_GetWidth(shrank_bitmap),
                FreeImage_GetHeight(shrank_bitmap),
                FreeImage_GetPitch(shrank_bitmap)
            );

            for (auto const& f: cfg.filters) {
                switch (f.kind) {
                    case FilterKind::Blur:
                        for (size_t _ = 0; _ < f.associated_val.nb_iterations; ++_) {
                            kernels::device::blur<<<grid, block>>>(d_img, width, height);
                        }
                        break;
                    case FilterKind::Diapositive:
                        kernels::device::diapositive<<<grid, block>>>(d_img, width, height);
                        break;
                    case FilterKind::Grayscale:
                        kernels::device::grayscale<<<grid, block>>>(d_img, width, height);
                        break;
                    case FilterKind::HorizontalFlip:
                        kernels::device::flip_horizontally<<<grid, block>>>(
                            d_img,
                            d_tmp,
                            width,
                            height
                        );
                        break;
                    case FilterKind::PopArt:
                        // Allocate small images on host
                        uint8_t* h_img_tl;
                        cudaMallocHost((void**)(&h_img_tl), size_split);

                        uint8_t* h_img_tr;
                        cudaMallocHost((void**)(&h_img_tr), size_split);

                        uint8_t* h_img_bl;
                        cudaMallocHost((void**)(&h_img_bl), size_split);

                        uint8_t* h_img_br;
                        cudaMallocHost((void**)(&h_img_br), size_split);

                        // Allocate small images on device
                        uint8_t* d_img_tl;
                        cudaMalloc((void**)(&d_img_tl), size_split);

                        uint8_t* d_img_tr;
                        cudaMalloc((void**)(&d_img_tr), size_split);

                        uint8_t* d_img_bl;
                        cudaMalloc((void**)(&d_img_bl), size_split);

                        uint8_t* d_img_br;
                        cudaMalloc((void**)(&d_img_br), size_split);

                        // Create streams
                        cudaStream_t stream[4];
                        for (size_t i = 0; i < 4; ++i) {
                            cudaStreamCreate(&stream[i]);
                        }

                        // Load small images
                        memcpy(h_img_tl, tmp.data(), size_split);
                        memcpy(h_img_tr, h_img_tl, size_split);
                        memcpy(h_img_bl, h_img_tl, size_split);
                        memcpy(h_img_br, h_img_tl, size_split);

                        // Copy host-allocated small images on device
                        cudaMemcpyAsync(
                            d_img_tl,
                            h_img_tl,
                            size_split,
                            cudaMemcpyHostToDevice,
                            stream[0]
                        );
                        cudaMemcpyAsync(
                            d_img_tr,
                            h_img_tr,
                            size_split,
                            cudaMemcpyHostToDevice,
                            stream[1]
                        );
                        cudaMemcpyAsync(
                            d_img_bl,
                            h_img_bl,
                            size_split,
                            cudaMemcpyHostToDevice,
                            stream[2]
                        );
                        cudaMemcpyAsync(
                            d_img_br,
                            h_img_br,
                            size_split,
                            cudaMemcpyHostToDevice,
                            stream[3]
                        );

                        // Launch kernels
                        kernels::device::saturate_color<<<grid_popart, block, 0, stream[0]>>>(
                            d_img_tl,
                            width_split,
                            height_split,
                            Red
                        );
                        kernels::device::saturate_color<<<grid_popart, block, 0, stream[1]>>>(
                            d_img_tr,
                            width_split,
                            height_split,
                            Green
                        );
                        kernels::device::saturate_color<<<grid_popart, block, 0, stream[2]>>>(
                            d_img_bl,
                            width_split,
                            height_split,
                            Blue
                        );
                        kernels::device::grayscale<<<grid_popart, block, 0, stream[3]>>>(
                            d_img_br,
                            width_split,
                            height_split
                        );

                        // Copy small images from device to host
                        cudaMemcpyAsync(
                            h_img_tl,
                            d_img_tl,
                            size_split,
                            cudaMemcpyDeviceToHost,
                            stream[0]
                        );
                        cudaMemcpyAsync(
                            h_img_tr,
                            d_img_tr,
                            size_split,
                            cudaMemcpyDeviceToHost,
                            stream[1]
                        );
                        cudaMemcpyAsync(
                            h_img_bl,
                            d_img_bl,
                            size_split,
                            cudaMemcpyDeviceToHost,
                            stream[2]
                        );
                        cudaMemcpyAsync(
                            h_img_br,
                            d_img_br,
                            size_split,
                            cudaMemcpyDeviceToHost,
                            stream[3]
                        );
                        cudaDeviceSynchronize();

                        for (size_t y = 0; y < height_split; y++) {
                            size_t idx_sub_img = y * width_split * 3;
                            size_t idx_img_ul = y * width * 3 + height_split * width * 3;
                            size_t idx_img_ur = y * width * 3 + height_split * width * 3 + width_split * 3;
                            size_t idx_img_dl = y * width * 3;
                            size_t idx_img_dr = y * width * 3 + width_split * 3;

                            utils::imgcpy(h_img, h_img_tl, width_split, idx_img_ul, idx_sub_img);
                            utils::imgcpy(h_img, h_img_tr, width_split, idx_img_ur, idx_sub_img);
                            utils::imgcpy(h_img, h_img_bl, width_split, idx_img_dl, idx_sub_img);
                            utils::imgcpy(h_img, h_img_br, width_split, idx_img_dr, idx_sub_img);
                        }

                        for (size_t i = 0; i < 4; ++i) {
                            cudaStreamDestroy(stream[i]);
                        }

                        cudaFree(d_img_tl);
                        cudaFree(d_img_tr);
                        cudaFree(d_img_bl);
                        cudaFree(d_img_br);

                        cudaFreeHost(h_img_tl);
                        cudaFreeHost(h_img_tr);
                        cudaFreeHost(h_img_bl);
                        cudaFreeHost(h_img_br);

                        cudaMemcpy(d_img, h_img, size, cudaMemcpyHostToDevice);
                        break;
                    case FilterKind::Saturate:
                        kernels::device::saturate_color<<<grid, block>>>(
                            d_img,
                            width,
                            height,
                            f.associated_val.color_to_saturate
                        );
                        break;
                    case FilterKind::Sobel:
                        kernels::device::grayscale<<<grid, block>>>(d_img, width, height);
                        cudaMemcpy(d_tmp, d_img, size, cudaMemcpyDeviceToDevice);
                        kernels::device::sobel<<<grid, block>>>(
                            d_img,
                            d_tmp,
                            width,
                            height,
                            f.associated_val.threshold
                        );
                        break;
                }
            }

            cudaMemcpy(img.data(), d_img, size, cudaMemcpyDeviceToHost);

            cudaFreeHost(h_img);
            cudaFree(d_img);
            cudaFree(d_tmp);

            return ret;
        }
    } // namespace device
} // namespace driver
