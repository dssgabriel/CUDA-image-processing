# APM - Mini CUDA project: Image processing

## Build
### Dependencies
- A C++ compiler supporting ISO C++11 ;
- NVIDIA's CUDA compiler `nvcc` and NVIDIA's CUDA runtime libraries supporting ISO C++ 11 ;
- The FreeImage C++ library.

The provided Makefile assumes that the FreeImage library has been installed at `./ext/FreeImage/build` from the root of the project.
If this is not the case, please update the relevant parts of the Makefile as follows:
```make
CXXFLAGS := -std=c++11 -Wall -Wextra -fopenmp -g -I $(INCDIR)/ \
            -I path/to/FreeImage/include/                      \
            -L path/to/FreeImage/lib/
NVCXXFLAGS := -std=c++11 -O3 -use_fast_math -extra-device-vectorization -Xcompiler -fopenmp -g -I $(INCDIR)/ \
              -I path/to/FreeImage/include/                                                                  \
              -L path/to/FreeImage/lib/
```
Do not forget to re-export the correct location of the library with `LD_LIBRARY_PATH`:
```sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:path/to/FreeImage/lib/
```

The user must specify the default **Compute Capability** for the CUDA using the `COMPUTE_CAPABILITY` variable.
Once these prerequisites are matched the project can be built using the provided Makefile, e.g. for NVIDIA A100 GPUs:
```sh
COMPUTE_CAPABILITY=80 make
```

## Features
The project provides multiple runtime options and flags that are listed in the help command:
```
target/cuda-image-transform [-h|--help]
```
Arguments in between angle brackets `< >` are required, those in between square brackets `[ ]` are optional.
> *Note:* Because of `getopt`/`getopt_long` special treatment of optional arguments (see their manpage), flags that take optional arguments are required to have no space between the flag and the argument, e.g. when specifying a threshold for the Sobel filter:
> ```sh
> target/cuda-image-transform --sobel 100 # BAD: argument `100` is ignored
> target/cuda-image-transform --sobel100  # OK!: Threshold is set to `100`
> ```

The implemented filters are:
- Blur and iterative blur ;
- Diapositive ;
- Grayscale ;
- Horizontal flip ;
- Pop-Art ;
- Saturation (red, green and blue) ;
- Sobel filter.

Apart from the `pop-art` filter which is GPU-only, all filters are implemented for both CPU and GPU.
The target can be set from the command line when launching the program using the `-t` or `--target` option (default target is the GPU device).

Moreover, the filters can be chained together with no memory transfers by specifying them in-order from the command line, e.g.:
```sh
# Flip the image, apply iterative blur 50 times, saturate green and saturate blue:
target/cuda-image-transform --horizontal-flip --blur50 --saturategreen --saturateblue
# Or with the shorthand options:
target/cuda-image-transform -f -b50 -sg -sb
```
> *Note:* some filters may override each other.
