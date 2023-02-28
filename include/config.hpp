#pragma once

#include <cstdint>
#include <cstring>
#include <getopt.h>
#include <string>
#include <vector>

enum ColorKind : size_t {
    Red,
    Green,
    Blue
};

enum class TargetKind {
    Host,
    Device,
};

enum class FilterKind {
    Blur,
    Diapositive,
    Grayscale,
    HorizontalFlip,
    PopArt,
    Saturate,
    Sobel,
};

struct Filter {
    FilterKind kind;
    union {
        ColorKind color_to_saturate;
        size_t nb_iterations;
        uint8_t threshold;
    } associated_val;
};

struct Config {
    std::string input_filename;
    std::string output_filename;
    std::vector<Filter> filters;
    TargetKind target;
    bool quiet;

    Config(
        std::string input,
        std::string output,
        std::vector<Filter> filters,
        TargetKind target,
        bool quiet
    ) : input_filename(input),
        output_filename(output),
        filters(std::move(filters)),
        target(target),
        quiet(quiet)
    {} 

    static auto parse_args(int32_t argc, char** argv) -> Config;

    static auto help(char const* bin) -> void;

    auto display() -> void;
};

