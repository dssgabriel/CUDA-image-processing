#include "config.hpp"

static constexpr size_t BUFFER_MAX_LEN = 64;
static constexpr char VERSION[] = "v0.1.0";

auto Config::parse_args(int32_t argc, char** argv) -> Config {
    std::string input("img.jpg");
    std::string output("new_img.jpg");
    std::vector<Filter> filters;
    TargetKind target = TargetKind::Device;
    bool quiet = false;

    int32_t curr_opt = 0;
    for (;;) {
        static struct option long_opts[] = {
            { "help", no_argument, nullptr, 'h' },
            { "version", no_argument, nullptr, 'v' },
            { "quiet", no_argument, nullptr, 'q' },
            { "target", required_argument, nullptr, 't' },
            { "input-filename", required_argument, nullptr, 'i' },
            { "output-filename", required_argument, nullptr, 'o' },
            { "blur", optional_argument, nullptr, 'b' },
            { "diapositive", no_argument, nullptr, 'd' },
            { "horizontal-flip", no_argument, nullptr, 'f' },
            { "grayscale", no_argument, nullptr, 'g' },
            { "pop-art", no_argument, nullptr, 'p' },
            { "saturate", optional_argument, nullptr, 's' },
            { "sobel", optional_argument, nullptr, 'e' },
            { nullptr, 0, nullptr, 0 },
        };

        int32_t opt_idx = 0;
        curr_opt = getopt_long(argc, argv, "hvqt:i:o:dfgpb::s::e::", long_opts, &opt_idx);
        if (curr_opt == -1) { break; }

        Filter filter;
        switch (curr_opt) {
            case 'h':
                Config::help(argv[0]);
                std::exit(EXIT_SUCCESS);
            case 'v':
                fprintf(
                    stderr,
                    "\033[1mAPM - CPU & CUDA accelerated image filters\033[0m\n%s\n",
                    VERSION
                );
                std::exit(EXIT_SUCCESS);
            case 'q':
                quiet = true;
                break;
            case 't':
                if (!strcmp(optarg, "cpu")) {
                    target = TargetKind::Host;
                } else if (!strcmp(optarg, "gpu")) {
                    target = TargetKind::Device;
                } else {
                    fprintf(stderr, "\033[1;31merror:\033[0m invalid target.\nSee help below.\n\n");
                    Config::help(argv[0]);
                    std::exit(EXIT_FAILURE);
                }
                break;
            case 'i':
                input = optarg;
                break;
            case 'o':
                output = optarg;
                break;
            
            case 'b':
                filter.kind = FilterKind::Blur;
                if (!optarg) {
                    filter.associated_val.nb_iterations = 16;
                } else {
                    filter.associated_val.nb_iterations = (size_t)(atoi(optarg));
                }
                filters.push_back(filter);
                break;
            case 'd':
                filter.kind = FilterKind::Diapositive;
                filters.push_back(filter);
                break;
            case 'e':
                filter.kind = FilterKind::Sobel;
                if (!optarg) {
                    filter.associated_val.threshold = UINT8_MAX / 2;
                } else {
                    int32_t tmp = atoi(optarg);
                    if (tmp < 0 || tmp > UINT8_MAX) {
                        fprintf(
                            stderr,
                            "\033[1;31merror:\033[0m invalid threshold value "
                            "(must be between 0 and 255).\n"
                            );
                        std::exit(EXIT_FAILURE);
                    }
                    filter.associated_val.threshold = tmp;
                }
                filters.push_back(filter);
                break;
            case 'f':
                filter.kind = FilterKind::HorizontalFlip;
                filters.push_back(filter);
                break;
            case 'g':
                filter.kind = FilterKind::Grayscale;
                filters.push_back(filter);
                break;
            case 'p':
                filter.kind = FilterKind::PopArt;
                filters.push_back(filter);
                break;
            case 's':
                filter.kind = FilterKind::Saturate;
                if (!optarg) {
                    filter.associated_val.color_to_saturate = ColorKind(Red);
                } else if (!strcmp(optarg, "red") || !strcmp(optarg, "r")) {
                    filter.associated_val.color_to_saturate = ColorKind(Red);
                } else if (!strcmp(optarg, "green") || !strcmp(optarg, "g")) {
                    filter.associated_val.color_to_saturate = ColorKind(Green);
                } else if (!strcmp(optarg, "blue") || !strcmp(optarg, "b")) {
                    filter.associated_val.color_to_saturate = ColorKind(Blue);
                } else {
                    fprintf(stderr, "\033[1;31merror:\033[0m invalid color.\nSee help below.\n\n");
                    Config::help(argv[0]);
                    std::exit(EXIT_FAILURE);
                }

                filters.push_back(filter);
                break;
            default:
                fprintf(stderr, "See help below.\n\n");
                Config::help(argv[0]);
                std::exit(EXIT_FAILURE);
        }
    }

    if (filters.empty()) {
        fprintf(stderr, "\033[1;31merror:\033[0m no filters to apply.\n");
        std::exit(EXIT_FAILURE);
    }

    return Config(input, output, filters, target, quiet);
}

auto Config::help(char const* bin) -> void {
    fprintf(stderr, "\033[1mAPM - CPU & CUDA accelerated image filters\033[0m\n");

    fprintf(stderr, "\n\033[1mUsage:\033[0m %s [FLAGS] [OPTIONS]\n", bin);

    fprintf(stderr, "\n\033[1mFlags:\033[0m\n");
    fprintf(stderr, "  -h, --help              Print this help.\n");
    fprintf(stderr, "  -v, --version           Print program version.\n");
    fprintf(stderr, "  -q, --quiet             Be quiet.\n");
    fprintf(
        stderr,
        "  -t, --target \033[33m<cpu|gpu>\033[0m  Define target on which to run (default: `gpu`).\n"
    );
    fprintf(
        stderr,
        "  -i, --input-file \033[33m<VAL>\033[0m  Path to input image file "
        "(default: `img.jpg`).\n"
    );
    fprintf(
        stderr,
        "  -o, --output-file \033[33m<VAL>\033[0m Path to output image file "
        "(default: `new_img.jpg`).\n"
    );

    fprintf(stderr, "\n\033[1mOptions:\033[0m\n");
    fprintf(
        stderr,
        "  -b, --blur \033[33m[VAL]\033[0m        "
        "Apply iterative blur filter to image (default: `16` iterations).\n"
    );
    fprintf(stderr, "  -d, --diapositive       Apply diapositive filter to image.\n");
    fprintf(stderr, "  -f, --horizontal-flip   Flip image horizontally.\n");
    fprintf(stderr, "  -g, --grayscale         Apply grayscale filter to image.\n");
    fprintf(
        stderr,
        "  -p, --pop-art           Apply Pop-Art filter to image (unavailable on host CPU).\n"
    );
    fprintf(
        stderr,
        "  -s, --saturate \033[33m[VAL]\033[0m    Apply saturation filter to image (default: "
        "`\033[31mred\033[0m`).\n                            "
        "Available options: `\033[31mred\033[0m`, `\033[32mgreen\033[0m`, `\033[34mblue\033[0m`.\n"
    );
    fprintf(
        stderr,
        "  -e, --sobel \033[33m[VAL]\033[0m       Apply Sobel edge detection filter to "
        "image (default threshold: `127`).\n"
    );
}

auto Config::display() -> void {
    char* filter_name = (char*)(malloc(32));
    char* filter_option = (char*)(malloc(32));

    fprintf(stderr, "\033[1mConfiguration:\033[0m\n");
    fprintf(
        stderr, 
        "   Input image: %s\n  Output image: %s\n",
        this->input_filename.c_str(),
        this->output_filename.c_str()
    );

    fprintf(stderr, "       Filters: ");
    for (auto f: this->filters) {
        switch (f.kind) {
            case FilterKind::Saturate:
                sprintf(filter_name, "saturation");
                sprintf(filter_option, "%s", f.associated_val.color_to_saturate == Red ? 
                    " (red)" : f.associated_val.color_to_saturate == Green ?
                        " (green)" : " (blue)"
                );
                break;
            case FilterKind::Blur:
                sprintf(filter_name, "blur");
                sprintf(filter_option, " (%zu times)", f.associated_val.nb_iterations);
                break;
            case FilterKind::Diapositive:
                sprintf(filter_name, "diapositive");
                break;
            case FilterKind::Grayscale:
                sprintf(filter_name, "grayscale");
                break;
            case FilterKind::HorizontalFlip:
                sprintf(filter_name, "horizontal flip");
                break;
            case FilterKind::Sobel:
                sprintf(filter_name, "Sobel");
                sprintf(filter_option, " (threshold: %hhu)", f.associated_val.threshold);
                break;
            case FilterKind::PopArt:
                sprintf(filter_name, "Pop-Art");
                break;
        }
        fprintf(stderr, "%s%s, ", filter_name, filter_option);
    }

    if (!this->filters.empty()) {
        fprintf(stderr, "\b\b  \b\b\n");
    } else {
        fprintf(stderr, "none\n");
    }
    fprintf(stderr, "        Target: ");
    switch (this->target) {
        case TargetKind::Host:
            fprintf(stderr, "Host CPU\n");
            break;
        case TargetKind::Device:
            fprintf(stderr, "Device GPU\n");
            break;
    }

    free(filter_name);
    free(filter_option);
}
