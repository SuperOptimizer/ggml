#include "ggml-opt.h"
#include "mnist-common.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

int main(int argc, char ** argv) {

    mnist_model model = mnist_model_init_random("mnist-fc",  "", MNIST_NBATCH_LOGICAL, MNIST_NBATCH_PHYSICAL);

    mnist_model_build(model);

    mnist_model_save(model, "mnist-fc-f32.gguf");
}
