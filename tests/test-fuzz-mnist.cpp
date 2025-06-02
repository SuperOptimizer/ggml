#include "ggml.h"
#include "ggml-opt.h"


#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <thread>
#include <vector>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-opt.h"


#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <random>
#include <string>
#include <utility>

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-cpu.h"
#include "ggml-opt.h"

#define MNIST_NTRAIN 60000
#define MNIST_NTEST  10000

// Gradient accumulation can be achieved by setting the logical batch size to a multiple of the physical one.
// The logical batch size determines how many datapoints are used for a gradient update.
// The physical batch size determines how many datapoints are processed in parallel, larger values utilize compute better but need more memory.
#define MNIST_NBATCH_LOGICAL  1000
#define MNIST_NBATCH_PHYSICAL  500

static_assert(MNIST_NBATCH_LOGICAL % MNIST_NBATCH_PHYSICAL == 0, "MNIST_NBATCH_LOGICAL % MNIST_NBATCH_PHYSICAL != 0");
static_assert(MNIST_NTRAIN % MNIST_NBATCH_LOGICAL == 0, "MNIST_NTRAIN % MNIST_NBATCH_LOGICAL != 0");
static_assert(MNIST_NTEST  % MNIST_NBATCH_LOGICAL == 0, "MNIST_NTRAIN % MNIST_NBATCH_LOGICAL != 0");

#define MNIST_HW       28
#define MNIST_NINPUT   (MNIST_HW*MNIST_HW)
#define MNIST_NCLASSES 10

#define MNIST_NHIDDEN  500

// NCB = number of channels base
#define MNIST_CNN_NCB 8

struct gguf_context * gguf_init_from_file_impl(FILE * file, struct gguf_init_params params);

struct mnist_model {
    std::string arch;
    ggml_backend_sched_t backend_sched;
    std::vector<ggml_backend_t> backends;
    const int nbatch_logical;
    const int nbatch_physical;

    struct ggml_tensor * images     = nullptr;
    struct ggml_tensor * logits     = nullptr;

    struct ggml_tensor * fc1_weight = nullptr;
    struct ggml_tensor * fc1_bias   = nullptr;
    struct ggml_tensor * fc2_weight = nullptr;
    struct ggml_tensor * fc2_bias   = nullptr;

    struct ggml_tensor * conv1_kernel = nullptr;
    struct ggml_tensor * conv1_bias   = nullptr;
    struct ggml_tensor * conv2_kernel = nullptr;
    struct ggml_tensor * conv2_bias   = nullptr;
    struct ggml_tensor * dense_weight = nullptr;
    struct ggml_tensor * dense_bias   = nullptr;

    struct ggml_context * ctx_gguf    = nullptr;
    struct ggml_context * ctx_static  = nullptr;
    struct ggml_context * ctx_compute = nullptr;
    ggml_backend_buffer_t buf_gguf    = nullptr;
    ggml_backend_buffer_t buf_static  = nullptr;

    mnist_model(const std::string & backend_name, const int nbatch_logical, const int nbatch_physical)
            : nbatch_logical(nbatch_logical), nbatch_physical(nbatch_physical) {
        std::vector<ggml_backend_dev_t> devices;
        const int ncores_logical = std::thread::hardware_concurrency();
        const int nthreads = std::min(ncores_logical, (ncores_logical + 4) / 2);

        // Add primary backend:
        if (!backend_name.empty()) {
            ggml_backend_dev_t dev = ggml_backend_dev_by_name(backend_name.c_str());
            if (dev == nullptr) {
                fprintf(stderr, "%s: ERROR: backend %s not found, available:\n", __func__, backend_name.c_str());
                for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                    ggml_backend_dev_t dev_i = ggml_backend_dev_get(i);
                    fprintf(stderr, "  - %s (%s)\n", ggml_backend_dev_name(dev_i), ggml_backend_dev_description(dev_i));
                }
                exit(1);
            }

            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            GGML_ASSERT(backend);

            if (ggml_backend_is_cpu(backend)) {
                ggml_backend_cpu_set_n_threads(backend, nthreads);
            }

            backends.push_back(backend);
            devices.push_back(dev);
        }

        // Add all available backends as fallback.
        // A "backend" is a stream on a physical device so there is no problem with adding multiple backends for the same device.
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);

            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            GGML_ASSERT(backend);

            if (ggml_backend_is_cpu(backend)) {
                ggml_backend_cpu_set_n_threads(backend, nthreads);
            }

            backends.push_back(backend);
            devices.push_back(dev);
        }

        // The order of the backends passed to ggml_backend_sched_new determines which backend is given priority.
        backend_sched = ggml_backend_sched_new(backends.data(), nullptr, backends.size(), GGML_DEFAULT_GRAPH_SIZE, false, true);
        fprintf(stderr, "%s: using %s (%s) as primary backend\n",
                __func__, ggml_backend_name(backends[0]), ggml_backend_dev_description(devices[0]));
        if (backends.size() >= 2) {
            fprintf(stderr, "%s: unsupported operations will be executed on the following fallback backends (in order of priority):\n", __func__);
            for (size_t i = 1; i < backends.size(); ++i) {
                fprintf(stderr, "%s:  - %s (%s)\n", __func__, ggml_backend_name(backends[i]), ggml_backend_dev_description(devices[i]));
            }
        }

        {
            const size_t size_meta = 1024*ggml_tensor_overhead();
            struct ggml_init_params params = {
                /*.mem_size   =*/ size_meta,
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ctx_static = ggml_init(params);
        }

        {
            // The compute context needs a total of 3 compute graphs: forward pass + backwards pass (with/without optimizer step).
            const size_t size_meta = GGML_DEFAULT_GRAPH_SIZE*ggml_tensor_overhead() + 3*ggml_graph_overhead();
            struct ggml_init_params params = {
                /*.mem_size   =*/ size_meta,
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ctx_compute = ggml_init(params);
        }
    }

    ~mnist_model() {
        ggml_free(ctx_gguf);
        ggml_free(ctx_static);
        ggml_free(ctx_compute);

        ggml_backend_buffer_free(buf_gguf);
        ggml_backend_buffer_free(buf_static);
        ggml_backend_sched_free(backend_sched);
        for (ggml_backend_t backend : backends) {
            ggml_backend_free(backend);
        }
    }
};

bool mnist_image_load(const std::string & fname, ggml_opt_dataset_t dataset);
void mnist_image_print(FILE * f, ggml_opt_dataset_t dataset, const int iex);
bool mnist_label_load(const std::string & fname, ggml_opt_dataset_t dataset);

mnist_model       mnist_model_init_from_file(uint8_t* data, uint64_t len, const std::string & backend, const int nbatch_logical, const int nbatch_physical);
mnist_model       mnist_model_init_random(const std::string & arch, const std::string & backend, const int nbatch_logical, const int nbatch_physical);
void              mnist_model_build(mnist_model & model);
ggml_opt_result_t mnist_model_eval(mnist_model & model, ggml_opt_dataset_t dataset);
void              mnist_model_train(mnist_model & model, ggml_opt_dataset_t dataset, const int nepoch, const float val_split);
void              mnist_model_save(mnist_model & model, const std::string & fname);


bool mnist_image_load(const std::string & fname, ggml_opt_dataset_t dataset) {
    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "failed to open images file %s\n", fname.c_str());
        return false;
    }
    fin.seekg(16);

    uint8_t image[MNIST_NINPUT];
    struct ggml_tensor * images = ggml_opt_dataset_data(dataset);
    float * buf = ggml_get_data_f32(images);

    GGML_ASSERT(images->ne[0] == MNIST_NINPUT);
    for (int64_t iex = 0; iex < images->ne[1]; ++iex) {
        fin.read((char *) image, sizeof(image));

        for (int64_t i = 0; i < MNIST_NINPUT; ++i) {
            buf[iex*MNIST_NINPUT + i] = image[i] / 255.0f; // Normalize to [0, 1]
        }
    }

    return true;
}

void mnist_image_print(FILE * stream, ggml_opt_dataset_t dataset, const int iex) {
    struct ggml_tensor * images = ggml_opt_dataset_data(dataset);
    GGML_ASSERT(images->ne[0] == MNIST_NINPUT);
    GGML_ASSERT(iex < images->ne[1]);
    const float * image = ggml_get_data_f32(images) + iex*MNIST_NINPUT;

    for (int64_t row = 0; row < MNIST_HW; row++) {
        for (int64_t col = 0; col < MNIST_HW; col++) {
            const int rgb = roundf(255.0f * image[row*MNIST_HW + col]);
#ifdef _WIN32
            fprintf(stream, "%s", rgb >= 220 ? "##" : "__");                // Represented via text.
#else
            fprintf(stream, "\033[48;2;%d;%d;%dm  \033[0m", rgb, rgb, rgb); // Represented via colored blocks.
#endif // _WIN32
        }
        fprintf(stream, "\n");
    }
}

bool mnist_label_load(const std::string & fname, ggml_opt_dataset_t dataset) {
    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "failed to open labels file %s\n", fname.c_str());
        return 0;
    }
    fin.seekg(8);

    uint8_t label;
    struct ggml_tensor * labels = ggml_opt_dataset_labels(dataset);
    float * buf = ggml_get_data_f32(labels);

    GGML_ASSERT(labels->ne[0] == MNIST_NCLASSES);
    for (int64_t iex = 0; iex < labels->ne[1]; ++iex) {
        fin.read((char *) &label, sizeof(label));

        for (int64_t i = 0; i < MNIST_NCLASSES; ++i) {
            buf[iex*MNIST_NCLASSES + i] = i == label ? 1.0f : 0.0f;
        }
    }

    return true;
}

// Temporary util function for loading data from GGUF to a backend != CPU until GGML itself provides this functionality:
bool load_from_gguf(void* data, uint64_t len, struct ggml_context * ctx_ggml, struct gguf_context * ctx_gguf) {
    FILE *f = fmemopen((void*)data, len, "rb");

    if (!f) {
        return false;
    }

    const size_t buf_size = 4*1024*1024;
    void * buf = malloc(buf_size);

    const int n_tensors = gguf_get_n_tensors(ctx_gguf);
    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(ctx_gguf, i);

        struct ggml_tensor * tensor = ggml_get_tensor(ctx_ggml, name);
        if (!tensor) {
            continue;
        }

        const size_t offs = gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i);

        if (fseek(f, offs, SEEK_SET) != 0) {
            fclose(f);
            free(buf);
            return false;
        }

        const size_t nbytes = ggml_nbytes(tensor);
        for (size_t pos = 0; pos < nbytes; pos += buf_size) {
            const size_t nbytes_cpy = buf_size < nbytes - pos ? buf_size : nbytes - pos;

            if (fread(buf, 1, nbytes_cpy, f) != nbytes_cpy) {
                fclose(f);
                free(buf);
                return false;
            }

            ggml_backend_tensor_set(tensor, buf, pos, nbytes_cpy);
        }
    }

    fclose(f);
    free(buf);
    return true;
}

mnist_model mnist_model_init_from_file(uint8_t* data, uint64_t len, const std::string & backend, const int nbatch_logical, const int nbatch_physical) {
    mnist_model model(backend, nbatch_logical, nbatch_physical);

    struct gguf_context * ctx;
    {
        struct gguf_init_params params = {
            /*.no_alloc   =*/ true,
            /*.ctx        =*/ &model.ctx_gguf,
        };
        FILE *file = fmemopen((void*)data, len, "rb");
        if (!file) {
            exit(1);
        }

        ctx = gguf_init_from_file_impl(file, params);
        if (!ctx) {
            fprintf(stderr, "%s: gguf_init_from_file_impl() failed\n", __func__);
            fclose(file);
            exit(1);
        }
        fclose(file);
    }
    model.arch = gguf_get_val_str(ctx, gguf_find_key(ctx, "general.architecture"));
    fprintf(stderr, "%s: model arch is %s\n", __func__, model.arch.c_str());

    if (model.arch == "mnist-fc") {
        model.fc1_weight = ggml_get_tensor(model.ctx_gguf, "fc1.weight");
        GGML_ASSERT(model.fc1_weight->ne[0] == MNIST_NINPUT);
        GGML_ASSERT(model.fc1_weight->ne[1] == MNIST_NHIDDEN);
        GGML_ASSERT(model.fc1_weight->ne[2] == 1);
        GGML_ASSERT(model.fc1_weight->ne[3] == 1);

        model.fc1_bias = ggml_get_tensor(model.ctx_gguf, "fc1.bias");
        GGML_ASSERT(model.fc1_bias->ne[0] == MNIST_NHIDDEN);
        GGML_ASSERT(model.fc1_bias->ne[1] == 1);
        GGML_ASSERT(model.fc1_bias->ne[2] == 1);
        GGML_ASSERT(model.fc1_bias->ne[3] == 1);

        model.fc2_weight = ggml_get_tensor(model.ctx_gguf, "fc2.weight");
        GGML_ASSERT(model.fc2_weight->ne[0] == MNIST_NHIDDEN);
        GGML_ASSERT(model.fc2_weight->ne[1] == MNIST_NCLASSES);
        GGML_ASSERT(model.fc2_weight->ne[2] == 1);
        GGML_ASSERT(model.fc2_weight->ne[3] == 1);

        model.fc2_bias = ggml_get_tensor(model.ctx_gguf, "fc2.bias");
        GGML_ASSERT(model.fc2_bias->ne[0] == MNIST_NCLASSES);
        GGML_ASSERT(model.fc2_bias->ne[1] == 1);
        GGML_ASSERT(model.fc2_bias->ne[2] == 1);
        GGML_ASSERT(model.fc2_bias->ne[3] == 1);
    } else if (model.arch == "mnist-cnn") {
        model.conv1_kernel = ggml_get_tensor(model.ctx_gguf, "conv1.kernel");
        GGML_ASSERT(model.conv1_kernel->type == GGML_TYPE_F32);
        GGML_ASSERT(model.conv1_kernel->ne[0] == 3);
        GGML_ASSERT(model.conv1_kernel->ne[1] == 3);
        GGML_ASSERT(model.conv1_kernel->ne[2] == 1);
        GGML_ASSERT(model.conv1_kernel->ne[3] == MNIST_CNN_NCB);

        model.conv1_bias = ggml_get_tensor(model.ctx_gguf, "conv1.bias");
        GGML_ASSERT(model.conv1_bias->type == GGML_TYPE_F32);
        GGML_ASSERT(model.conv1_bias->ne[0] == 1);
        GGML_ASSERT(model.conv1_bias->ne[1] == 1);
        GGML_ASSERT(model.conv1_bias->ne[2] == MNIST_CNN_NCB);
        GGML_ASSERT(model.conv1_bias->ne[3] == 1);

        model.conv2_kernel = ggml_get_tensor(model.ctx_gguf, "conv2.kernel");
        GGML_ASSERT(model.conv2_kernel->type == GGML_TYPE_F32);
        GGML_ASSERT(model.conv2_kernel->ne[0] == 3);
        GGML_ASSERT(model.conv2_kernel->ne[1] == 3);
        GGML_ASSERT(model.conv2_kernel->ne[2] == MNIST_CNN_NCB);
        GGML_ASSERT(model.conv2_kernel->ne[3] == MNIST_CNN_NCB*2);

        model.conv2_bias = ggml_get_tensor(model.ctx_gguf, "conv2.bias");
        GGML_ASSERT(model.conv2_bias->type == GGML_TYPE_F32);
        GGML_ASSERT(model.conv2_bias->ne[0] == 1);
        GGML_ASSERT(model.conv2_bias->ne[1] == 1);
        GGML_ASSERT(model.conv2_bias->ne[2] == MNIST_CNN_NCB*2);
        GGML_ASSERT(model.conv2_bias->ne[3] == 1);

        model.dense_weight = ggml_get_tensor(model.ctx_gguf, "dense.weight");
        GGML_ASSERT(model.dense_weight->type == GGML_TYPE_F32);
        GGML_ASSERT(model.dense_weight->ne[0] == (MNIST_HW/4)*(MNIST_HW/4)*(MNIST_CNN_NCB*2));
        GGML_ASSERT(model.dense_weight->ne[1] == MNIST_NCLASSES);
        GGML_ASSERT(model.dense_weight->ne[2] == 1);
        GGML_ASSERT(model.dense_weight->ne[3] == 1);

        model.dense_bias = ggml_get_tensor(model.ctx_gguf, "dense.bias");
        GGML_ASSERT(model.dense_bias->type == GGML_TYPE_F32);
        GGML_ASSERT(model.dense_bias->ne[0] == MNIST_NCLASSES);
        GGML_ASSERT(model.dense_bias->ne[1] == 1);
        GGML_ASSERT(model.dense_bias->ne[2] == 1);
        GGML_ASSERT(model.dense_bias->ne[3] == 1);
    } else {
        fprintf(stderr, "%s: unknown model arch: %s\n", __func__, model.arch.c_str());
    }

    model.buf_gguf = ggml_backend_alloc_ctx_tensors(model.ctx_gguf, model.backends[0]);

    if(!load_from_gguf(data,len, model.ctx_gguf, ctx)) {
        exit(1);
    }

    model.images = ggml_new_tensor_2d(model.ctx_static, GGML_TYPE_F32, MNIST_NINPUT, MNIST_NBATCH_PHYSICAL);
    ggml_set_name(model.images, "images");
    ggml_set_input(model.images);

    model.buf_static = ggml_backend_alloc_ctx_tensors(model.ctx_static, model.backends[0]);

    return model;
}

mnist_model mnist_model_init_random(const std::string & arch, const std::string & backend, const int nbatch_logical, const int nbatch_physical) {
    mnist_model model(backend, nbatch_logical, nbatch_physical);
    model.arch = arch;

    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<float> nd{0.0f, 1e-2f};
    std::vector<ggml_tensor *> init_tensors;

    if (model.arch == "mnist-fc") {
        fprintf(stderr, "%s: initializing random weights for a fully connected model\n", __func__);

        model.fc1_weight = ggml_new_tensor_2d(model.ctx_static, GGML_TYPE_F32, MNIST_NINPUT,  MNIST_NHIDDEN);
        model.fc1_bias   = ggml_new_tensor_1d(model.ctx_static, GGML_TYPE_F32,                MNIST_NHIDDEN);
        model.fc2_weight = ggml_new_tensor_2d(model.ctx_static, GGML_TYPE_F32, MNIST_NHIDDEN, MNIST_NCLASSES);
        model.fc2_bias   = ggml_new_tensor_1d(model.ctx_static, GGML_TYPE_F32,                MNIST_NCLASSES);

        ggml_set_name(model.fc1_weight, "fc1.weight");
        ggml_set_name(model.fc1_bias,   "fc1.bias");
        ggml_set_name(model.fc2_weight, "fc2.weight");
        ggml_set_name(model.fc2_bias,   "fc2.bias");

        init_tensors.push_back(model.fc1_weight);
        init_tensors.push_back(model.fc1_bias);
        init_tensors.push_back(model.fc2_weight);
        init_tensors.push_back(model.fc2_bias);
    } else if (model.arch == "mnist-cnn") {
        model.conv1_kernel = ggml_new_tensor_4d(model.ctx_static, GGML_TYPE_F32, 3, 3, 1, MNIST_CNN_NCB);
        model.conv1_bias   = ggml_new_tensor_3d(model.ctx_static, GGML_TYPE_F32, 1, 1,    MNIST_CNN_NCB);
        model.conv2_kernel = ggml_new_tensor_4d(model.ctx_static, GGML_TYPE_F32, 3, 3, MNIST_CNN_NCB, MNIST_CNN_NCB*2);
        model.conv2_bias   = ggml_new_tensor_3d(model.ctx_static, GGML_TYPE_F32, 1, 1,                MNIST_CNN_NCB*2);
        model.dense_weight = ggml_new_tensor_2d(model.ctx_static, GGML_TYPE_F32, (MNIST_HW/4)*(MNIST_HW/4)*(MNIST_CNN_NCB*2), MNIST_NCLASSES);
        model.dense_bias   = ggml_new_tensor_1d(model.ctx_static, GGML_TYPE_F32, MNIST_NCLASSES);

        ggml_set_name(model.conv1_kernel, "conv1.kernel");
        ggml_set_name(model.conv1_bias,   "conv1.bias");
        ggml_set_name(model.conv2_kernel, "conv2.kernel");
        ggml_set_name(model.conv2_bias,   "conv2.bias");
        ggml_set_name(model.dense_weight, "dense.weight");
        ggml_set_name(model.dense_bias,   "dense.bias");

        init_tensors.push_back(model.conv1_kernel);
        init_tensors.push_back(model.conv1_bias);
        init_tensors.push_back(model.conv2_kernel);
        init_tensors.push_back(model.conv2_bias);
        init_tensors.push_back(model.dense_weight);
        init_tensors.push_back(model.dense_bias);
    } else {
        fprintf(stderr, "%s: unknown model arch: %s\n", __func__, model.arch.c_str());
    }

    model.images = ggml_new_tensor_2d(model.ctx_static, GGML_TYPE_F32, MNIST_NINPUT, MNIST_NBATCH_PHYSICAL);
    ggml_set_name(model.images, "images");
    ggml_set_input(model.images);

    model.buf_static = ggml_backend_alloc_ctx_tensors(model.ctx_static, model.backends[0]);

    for (ggml_tensor * t : init_tensors) {
        GGML_ASSERT(t->type == GGML_TYPE_F32);
        const int64_t ne = ggml_nelements(t);
        std::vector<float> tmp(ne);

        for (int64_t i = 0; i < ne; ++i) {
            tmp[i] = nd(gen);
        }
        ggml_backend_tensor_set(t, tmp.data(), 0, ggml_nbytes(t));
    }

    return model;
}

void mnist_model_build(mnist_model & model) {
    if (model.arch == "mnist-fc") {
        ggml_set_param(model.fc1_weight);
        ggml_set_param(model.fc1_bias);
        ggml_set_param(model.fc2_weight);
        ggml_set_param(model.fc2_bias);

        ggml_tensor * fc1 = ggml_relu(model.ctx_compute, ggml_add(model.ctx_compute,
            ggml_mul_mat(model.ctx_compute, model.fc1_weight, model.images),
            model.fc1_bias));
        model.logits = ggml_add(model.ctx_compute,
            ggml_mul_mat(model.ctx_compute, model.fc2_weight, fc1),
            model.fc2_bias);
    } else if (model.arch == "mnist-cnn") {
        ggml_set_param(model.conv1_kernel);
        ggml_set_param(model.conv1_bias);
        ggml_set_param(model.conv2_kernel);
        ggml_set_param(model.conv2_bias);
        ggml_set_param(model.dense_weight);
        ggml_set_param(model.dense_bias);

        struct ggml_tensor * images_2D = ggml_reshape_4d(model.ctx_compute, model.images, MNIST_HW, MNIST_HW, 1, model.images->ne[1]);

        struct ggml_tensor * conv1_out = ggml_relu(model.ctx_compute, ggml_add(model.ctx_compute,
            ggml_conv_2d(model.ctx_compute, model.conv1_kernel, images_2D, 1, 1, 1, 1, 1, 1),
            model.conv1_bias));
        GGML_ASSERT(conv1_out->ne[0] == MNIST_HW);
        GGML_ASSERT(conv1_out->ne[1] == MNIST_HW);
        GGML_ASSERT(conv1_out->ne[2] == MNIST_CNN_NCB);
        GGML_ASSERT(conv1_out->ne[3] == model.nbatch_physical);

        struct ggml_tensor * conv2_in = ggml_pool_2d(model.ctx_compute, conv1_out, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
        GGML_ASSERT(conv2_in->ne[0] == MNIST_HW/2);
        GGML_ASSERT(conv2_in->ne[1] == MNIST_HW/2);
        GGML_ASSERT(conv2_in->ne[2] == MNIST_CNN_NCB);
        GGML_ASSERT(conv2_in->ne[3] == model.nbatch_physical);

        struct ggml_tensor * conv2_out = ggml_relu(model.ctx_compute, ggml_add(model.ctx_compute,
            ggml_conv_2d(model.ctx_compute, model.conv2_kernel, conv2_in, 1, 1, 1, 1, 1, 1),
            model.conv2_bias));
        GGML_ASSERT(conv2_out->ne[0] == MNIST_HW/2);
        GGML_ASSERT(conv2_out->ne[1] == MNIST_HW/2);
        GGML_ASSERT(conv2_out->ne[2] == MNIST_CNN_NCB*2);
        GGML_ASSERT(conv2_out->ne[3] == model.nbatch_physical);

        struct ggml_tensor * dense_in = ggml_pool_2d(model.ctx_compute, conv2_out, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
        GGML_ASSERT(dense_in->ne[0] == MNIST_HW/4);
        GGML_ASSERT(dense_in->ne[1] == MNIST_HW/4);
        GGML_ASSERT(dense_in->ne[2] == MNIST_CNN_NCB*2);
        GGML_ASSERT(dense_in->ne[3] == model.nbatch_physical);

        dense_in = ggml_reshape_2d(model.ctx_compute,
            ggml_cont(model.ctx_compute, ggml_permute(model.ctx_compute, dense_in, 1, 2, 0, 3)),
            (MNIST_HW/4)*(MNIST_HW/4)*(MNIST_CNN_NCB*2), model.nbatch_physical);
        GGML_ASSERT(dense_in->ne[0] == (MNIST_HW/4)*(MNIST_HW/4)*(MNIST_CNN_NCB*2));
        GGML_ASSERT(dense_in->ne[1] == model.nbatch_physical);
        GGML_ASSERT(dense_in->ne[2] == 1);
        GGML_ASSERT(dense_in->ne[3] == 1);

        model.logits = ggml_add(model.ctx_compute, ggml_mul_mat(model.ctx_compute, model.dense_weight, dense_in), model.dense_bias);
    } else {
        GGML_ASSERT(false);
    }

    ggml_set_name(model.logits, "logits");
    ggml_set_output(model.logits);
    GGML_ASSERT(model.logits->type == GGML_TYPE_F32);
    GGML_ASSERT(model.logits->ne[0] == MNIST_NCLASSES);
    GGML_ASSERT(model.logits->ne[1] == model.nbatch_physical);
    GGML_ASSERT(model.logits->ne[2] == 1);
    GGML_ASSERT(model.logits->ne[3] == 1);
}

ggml_opt_result_t mnist_model_eval(mnist_model & model, ggml_opt_dataset_t dataset) {
    ggml_opt_result_t result = ggml_opt_result_init();

    ggml_opt_params params = ggml_opt_default_params(model.backend_sched, GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
    params.ctx_compute = model.ctx_compute;
    params.inputs      = model.images;
    params.outputs     = model.logits;
    params.build_type  = GGML_OPT_BUILD_TYPE_FORWARD;
    ggml_opt_context_t opt_ctx = ggml_opt_init(params);

    {
        const int64_t t_start_us = ggml_time_us();

        ggml_opt_epoch(opt_ctx, dataset, nullptr, result, /*idata_split =*/ 0, nullptr, nullptr);

        const int64_t t_total_us = ggml_time_us() - t_start_us;
        const double t_total_ms = 1e-3*t_total_us;
        const int nex = ggml_opt_dataset_data(dataset)->ne[1];
        fprintf(stderr, "%s: model evaluation on %d images took %.2lf ms, %.2lf us/image\n",
                __func__, nex, t_total_ms, (double) t_total_us/nex);
    }

    ggml_opt_free(opt_ctx);

    return result;
}

void mnist_model_train(mnist_model & model, ggml_opt_dataset_t dataset, const int nepoch, const float val_split) {
    ggml_opt_fit(model.backend_sched, model.ctx_compute, model.images, model.logits, dataset,
        GGML_OPT_LOSS_TYPE_CROSS_ENTROPY, ggml_opt_get_default_optimizer_params, nepoch, model.nbatch_logical, val_split, false);
}

void mnist_model_save(mnist_model & model, const std::string & fname) {
    printf("%s: saving model to '%s'\n", __func__, fname.c_str());

    struct ggml_context * ggml_ctx;
    {
        struct ggml_init_params params = {
            /*.mem_size   =*/ 100 * 1024*1024,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false,
        };
        ggml_ctx = ggml_init(params);
    }

    gguf_context * gguf_ctx = gguf_init_empty();
    gguf_set_val_str(gguf_ctx, "general.architecture", model.arch.c_str());

    std::vector<struct ggml_tensor *> weights;
    if (model.arch == "mnist-fc") {
        weights = {model.fc1_weight, model.fc1_bias, model.fc2_weight, model.fc2_bias};
    } else if (model.arch == "mnist-cnn") {
        weights = {model.conv1_kernel, model.conv1_bias, model.conv2_kernel, model.conv2_bias, model.dense_weight, model.dense_bias};
    } else {
        GGML_ASSERT(false);
    }
    for (struct ggml_tensor * t : weights) {
        struct ggml_tensor * copy = ggml_dup_tensor(ggml_ctx, t);
        ggml_set_name(copy, t->name);
        ggml_backend_tensor_get(t, copy->data, 0, ggml_nbytes(t));
        gguf_add_tensor(gguf_ctx, copy);
    }
    gguf_write_to_file(gguf_ctx, fname.c_str(), false);

    ggml_free(ggml_ctx);
    gguf_free(gguf_ctx);
}


// Main fuzzer entry point
extern "C" int LLVMFuzzerTestOneInput(uint8_t *data, size_t size) {
    // Skip empty inputs
    if (size == 0) {
        return 0;
    }

    srand(time(NULL));
    ggml_time_init();

    ggml_opt_dataset_t dataset = ggml_opt_dataset_init(GGML_TYPE_F32, GGML_TYPE_F32, MNIST_NINPUT, MNIST_NCLASSES, 1000, 100);

    mnist_model model = mnist_model_init_from_file(data, size, "", MNIST_NBATCH_LOGICAL, MNIST_NBATCH_PHYSICAL);
    mnist_model_build(model);

    ggml_opt_result_t result_eval = mnist_model_eval(model, dataset);

    std::vector<int32_t> pred(MNIST_NTEST);
    ggml_opt_result_pred(result_eval, pred.data());

    double loss;
    double loss_unc;
    ggml_opt_result_loss(result_eval, &loss, &loss_unc);

    double accuracy;
    double accuracy_unc;
    ggml_opt_result_accuracy(result_eval, &accuracy, &accuracy_unc);

    ggml_opt_result_free(result_eval);

    // Close the file

    return 0;
}

// Optional: Initialize the fuzzer (if needed)
extern "C" int LLVMFuzzerInitialize(int *argc, char ***argv) {
    // Any one-time initialization can go here
    return 0;
}
