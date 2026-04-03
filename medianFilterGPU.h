#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstddef>
#include <sycl/sycl.hpp>
#include "utils.h"

class MedianFilterGPU {
private:
    static float median_7(float arr[7]);
    static uint8_t median_9(uint8_t window[9]);

    static inline int clamp_int(int v, int lo, int hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }

public:
    static void median_filter_7(const float* input, float* output, size_t length);
    static void median_filter_3x3(const uint8_t* input, uint8_t* output,
                                  size_t width, size_t height, size_t stride);
};

// -------------------------- 1D МЕДИАННЫЙ ФИЛЬТР ДЛЯ ОКНА 1x7 --------------------------

float MedianFilterGPU::median_7(float arr[7]) {
    cond_swap(arr[0], arr[6]);
    cond_swap(arr[2], arr[3]);
    cond_swap(arr[4], arr[5]);

    cond_swap(arr[0], arr[2]);
    cond_swap(arr[1], arr[4]);
    cond_swap(arr[3], arr[6]);

    arr[1] = get_max(arr[0], arr[1]);
    cond_swap(arr[2], arr[5]);
    cond_swap(arr[3], arr[4]);

    arr[2] = get_max(arr[1], arr[2]);
    arr[4] = get_min(arr[4], arr[6]);

    arr[3] = get_max(arr[2], arr[3]);
    arr[4] = get_min(arr[4], arr[5]);

    arr[3] = get_min(arr[3], arr[4]);

    return arr[3];
}

void MedianFilterGPU::median_filter_7(const float* input, float* output, size_t length) {
    sycl::queue q;

    size_t N = length;
    float* d_input = sycl::malloc_shared<float>(N, q);
    float* d_output = sycl::malloc_shared<float>(N, q);
    std::cout<<"1618\n";
    q.memcpy(d_input, input, N * sizeof(float)).wait();
    std::cout<<"1618\n";


    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(N - 6), [=](sycl::id<1> idx) {
            size_t i = idx[0] + 3;
            float window[7];

            for (int j = -3; j <= 3; ++j) window[j + 3] = d_input[i + j];

            d_output[i] = median_7(window);
        });
    });
    q.wait();


    float window[7];

    for (size_t i = 0; i < 3 && i < N; ++i) {
        for (int j = -3; j <= 3; ++j) {
            int idx = static_cast<int>(i) + j;
            if (idx < 0) window[j + 3] = d_input[0];
            else if (idx >= static_cast<int>(N)) window[j + 3] = d_input[N - 1];
            else window[j + 3] = d_input[idx];
        }
        d_output[i] = median_7(window);
    }

    for (size_t i = (N > 3 ? N - 3 : 0); i < N; ++i) {
        for (int j = -3; j <= 3; ++j) {
            int idx = static_cast<int>(i) + j;
            if (idx < 0) window[j + 3] = d_input[0];
            else if (idx >= static_cast<int>(N)) window[j + 3] = d_input[N - 1];
            else window[j + 3] = d_input[idx];
        }
        d_output[i] = median_7(window);
    }

    q.memcpy(output, d_output, N * sizeof(float)).wait();

    sycl::free(d_input, q);
    sycl::free(d_output, q);
}

// -------------------------- 2D МЕДИАННЫЙ ФИЛЬТР ДЛЯ ОКНА 3x3 --------------------------

uint8_t MedianFilterGPU::median_9(uint8_t window[9]) {
    cond_swap(window[0], window[3]);
    cond_swap(window[1], window[7]);
    cond_swap(window[2], window[5]);
    cond_swap(window[4], window[8]);

    cond_swap(window[0], window[7]);
    cond_swap(window[2], window[4]);
    cond_swap(window[3], window[8]);
    cond_swap(window[5], window[6]);

    window[2] = get_max(window[0], window[2]);
    cond_swap(window[1], window[3]);
    cond_swap(window[4], window[5]);
    window[7] = get_min(window[7], window[8]);

    window[4] = get_max(window[1], window[4]);
    window[3] = get_min(window[3], window[6]);
    window[5] = get_min(window[5], window[7]);

    cond_swap(window[2], window[4]);
    cond_swap(window[3], window[5]);

    window[3] = get_max(window[2], window[3]);
    window[4] = get_min(window[4], window[5]);

    window[4] = get_max(window[3], window[4]);

    return window[4];
}

void MedianFilterGPU::median_filter_3x3(const uint8_t* input, uint8_t* output,
                                        size_t width, size_t height, size_t stride) {
    sycl::queue q{sycl::default_selector_v};

    constexpr size_t WG_X = 16;
    constexpr size_t WG_Y = 16;

    const size_t global_x = (width  + WG_X - 1) / WG_X * WG_X;
    const size_t global_y = (height + WG_Y - 1) / WG_Y * WG_Y;

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint8_t, 2> tile(
            sycl::range<2>(WG_Y + 2, WG_X + 2), h
        );

        h.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(global_y, global_x),
                              sycl::range<2>(WG_Y, WG_X)),
            [=](sycl::nd_item<2> it) {
                const int lx = static_cast<int>(it.get_local_id(1));
                const int ly = static_cast<int>(it.get_local_id(0));
                const int gx = static_cast<int>(it.get_global_id(1));
                const int gy = static_cast<int>(it.get_global_id(0));

                const int group_x = static_cast<int>(it.get_group(1));
                const int group_y = static_cast<int>(it.get_group(0));

                const int base_x = group_x * static_cast<int>(WG_X);
                const int base_y = group_y * static_cast<int>(WG_Y);

                const int tile_w = static_cast<int>(WG_X + 2);
                const int tile_h = static_cast<int>(WG_Y + 2);
                const int local_size = static_cast<int>(WG_X * WG_Y);
                const int lid = ly * static_cast<int>(WG_X) + lx;

                for (int idx = lid; idx < tile_w * tile_h; idx += local_size) {
                    int ty = idx / tile_w;
                    int tx = idx % tile_w;

                    int img_x = clamp_int(base_x + tx - 1, 0, static_cast<int>(width) - 1);
                    int img_y = clamp_int(base_y + ty - 1, 0, static_cast<int>(height) - 1);

                    tile[ty][tx] = input[img_y * stride + img_x];
                }

                it.barrier(sycl::access::fence_space::local_space);

                if (gx < static_cast<int>(width) && gy < static_cast<int>(height)) {
                    uint8_t window[9];

                    const int tx = lx + 1;
                    const int ty = ly + 1;

                    window[0] = tile[ty - 1][tx - 1];
                    window[1] = tile[ty - 1][tx];
                    window[2] = tile[ty - 1][tx + 1];
                    window[3] = tile[ty][tx - 1];
                    window[4] = tile[ty][tx];
                    window[5] = tile[ty][tx + 1];
                    window[6] = tile[ty + 1][tx - 1];
                    window[7] = tile[ty + 1][tx];
                    window[8] = tile[ty + 1][tx + 1];

                    output[gy * stride + gx] = median_9(window);
                }
            }
        );
    }).wait();
}

