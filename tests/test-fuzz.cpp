#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <string>
#include <ctime>

// Include required GGUF headers
#include "ggml.h"
#include "ggml-backend.h"
#include "../src/ggml-impl.h"
#include "gguf.h"


// Helper function to test unary operations
// Forward declarations of helpers
void test_unary_ops(struct gguf_context* ctx);
void test_binary_ops(struct gguf_context* ctx);
void test_conv_ops(struct gguf_context* ctx);
void test_quantization_ops(struct gguf_context* ctx);
void test_rope_ops(struct gguf_context* ctx);
void test_flash_attn_ops(struct gguf_context* ctx);
void test_pool_ops(struct gguf_context* ctx);
void test_tensor_views(struct gguf_context* ctx);
void test_tensor_permute(struct gguf_context* ctx);
void test_tensor_reshape(struct gguf_context* ctx);
void test_tensor_split_merge(struct gguf_context* ctx);


void test_tensor_reshape(struct gguf_context* ctx) {
    if (!ctx) return;

    struct ggml_tensor* src = ggml_new_tensor_2d(NULL, GGML_TYPE_F32, 8, 8);
    if (!src) return;

    struct ggml_tensor* res = NULL;

    // Test different reshape operations
    res = ggml_reshape_1d(NULL, src, 64);
    res = ggml_reshape_2d(NULL, src, 16, 4);
    res = ggml_reshape_3d(NULL, src, 4, 4, 4);

    GGML_UNUSED(res);
}

void test_tensor_split_merge(struct gguf_context* ctx) {
    if (!ctx) return;

    struct ggml_tensor* src = ggml_new_tensor_2d(NULL, GGML_TYPE_F32, 8, 8);
    if (!src) return;

    // Split into views
    struct ggml_tensor* view1 = ggml_view_2d(NULL, src, 4, 8, src->nb[1], 0);
    struct ggml_tensor* view2 = ggml_view_2d(NULL, src, 4, 8, src->nb[1], 4 * src->nb[0]);

    // Merge back using concat
    struct ggml_tensor* merged = ggml_concat(NULL, view1, view2, 0);

    GGML_UNUSED(merged);
}


// Tensor manipulation test implementations
void test_tensor_permute(struct gguf_context* ctx) {
    if (!ctx) return;

    struct ggml_tensor* src = ggml_new_tensor_3d(NULL, GGML_TYPE_F32, 4, 4, 4);
    if (!src) return;

    struct ggml_tensor* res = NULL;

    // Test different permutations
    res = ggml_permute(NULL, src, 0, 2, 1, 3); // Swap dims 1 and 2
    res = ggml_permute(NULL, src, 1, 0, 2, 3); // Swap dims 0 and 1

    GGML_UNUSED(res);
}

// Helper function to test key-value operations
void test_kv_operations(struct gguf_context* ctx) {
    if (!ctx) return;

    // Test key finding and access operations
    const int64_t key_id = gguf_find_key(ctx, "test_key");
    if (key_id >= 0) {
        // Exercise different value getters based on type
        enum gguf_type type = gguf_get_kv_type(ctx, key_id);
        switch (type) {
            case GGUF_TYPE_UINT8:
                gguf_get_val_u8(ctx, key_id);
                break;
            case GGUF_TYPE_INT8:
                gguf_get_val_i8(ctx, key_id);
                break;
            case GGUF_TYPE_UINT16:
                gguf_get_val_u16(ctx, key_id);
                break;
            case GGUF_TYPE_INT16:
                gguf_get_val_i16(ctx, key_id);
                break;
            case GGUF_TYPE_UINT32:
                gguf_get_val_u32(ctx, key_id);
                break;
            case GGUF_TYPE_INT32:
                gguf_get_val_i32(ctx, key_id);
                break;
            case GGUF_TYPE_FLOAT32:
                gguf_get_val_f32(ctx, key_id);
                break;
            case GGUF_TYPE_BOOL:
                gguf_get_val_bool(ctx, key_id);
                break;
            case GGUF_TYPE_STRING:
                gguf_get_val_str(ctx, key_id);
                break;
            default:
                break;
        }

        // Test array handling if it's an array type
        if (type == GGUF_TYPE_ARRAY) {
            enum gguf_type arr_type = gguf_get_arr_type(ctx, key_id);
            size_t arr_n = gguf_get_arr_n(ctx, key_id);
            if (arr_n > 0) {
                const void* arr_data = gguf_get_arr_data(ctx, key_id);
                if (arr_type == GGUF_TYPE_STRING) {
                    for (size_t i = 0; i < arr_n; i++) {
                        gguf_get_arr_str(ctx, key_id, i);
                    }
                }
            }
        }
    }
}

// Helper function to test tensor operations
void test_tensor_operations(struct gguf_context* ctx) {
    if (!ctx) return;

    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    for (int64_t i = 0; i < n_tensors; i++) {
        // Get tensor information
        const char* name = gguf_get_tensor_name(ctx, i);
        if (name) {
            // Test tensor finding
            int64_t found_id = gguf_find_tensor(ctx, name);
            if (found_id >= 0) {
                // Exercise tensor property getters
                size_t offset = gguf_get_tensor_offset(ctx, found_id);
                enum ggml_type type = gguf_get_tensor_type(ctx, found_id);
                size_t size = gguf_get_tensor_size(ctx, found_id);

                // Attempt to access tensor data based on type and size
                if (size > 0 && offset > 0) {
                    // You might want to add specific tensor data handling here
                    // based on the tensor type
                }
            }
        }
    }
}

// Helper to test metadata operations
void test_meta_operations(struct gguf_context* ctx) {
    if (!ctx) return;

    // Get metadata size
    size_t meta_size = gguf_get_meta_size(ctx);
    if (meta_size > 0) {
        // Allocate buffer and get metadata
        std::vector<uint8_t> meta_buffer(meta_size);
        gguf_get_meta_data(ctx, meta_buffer.data());
    }

    // Test version and alignment getters
    uint32_t version = gguf_get_version(ctx);
    size_t alignment = gguf_get_alignment(ctx);
    size_t data_offset = gguf_get_data_offset(ctx);
}






void test_backend_ops(ggml_backend_t backend, const uint8_t* data, size_t size) {
    if (!backend || size < 100) return;

    struct ggml_context* ctx = ggml_init({sizeof(float) * 1000, NULL});
    if (!ctx) return;

    struct ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 16);
    struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 16);
    if (!a || !b) {
        ggml_free(ctx);
        return;
    }

    memcpy(a->data, data, std::min(size, (size_t)ggml_nbytes(a)));
    memcpy(b->data, data + ggml_nbytes(a), std::min(size - ggml_nbytes(a), (size_t)ggml_nbytes(b)));

    struct ggml_tensor* c = ggml_mul_mat(ctx, a, b);
    struct ggml_tensor* d = ggml_add(ctx, c, b);

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, d);

    ggml_backend_graph_compute(backend, graph);

    ggml_backend_tensor_set_async(backend, a, data, 0, std::min(size, (size_t)ggml_nbytes(a)));
    ggml_backend_tensor_get_async(backend, b, (void*)(data + ggml_nbytes(a)), 0,
                                 std::min(size - ggml_nbytes(a), (size_t)ggml_nbytes(b)));

    ggml_backend_synchronize(backend);
    ggml_free(ctx);
}

void test_backend_scheduler(const uint8_t* data, size_t size) {
    if (size < 200) return;

    ggml_backend_t cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    if (!cpu_backend) return;

    ggml_backend_t backends[] = {cpu_backend};
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, NULL, 1, 1000, false);
    if (!sched) {
        ggml_backend_free(cpu_backend);
        return;
    }

    struct ggml_context* ctx = ggml_init({sizeof(float) * 2000, NULL});
    if (!ctx) {
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu_backend);
        return;
    }

    struct ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 32);
    struct ggml_tensor* y = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 32);
    if (!x || !y) {
        ggml_free(ctx);
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu_backend);
        return;
    }

    memcpy(x->data, data, std::min(size, (size_t)ggml_nbytes(x)));
    memcpy(y->data, data + ggml_nbytes(x), std::min(size - ggml_nbytes(x), (size_t)ggml_nbytes(y)));

    struct ggml_tensor* z = ggml_mul_mat(ctx, x, y);
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, z);

    ggml_backend_sched_alloc_graph(sched, graph);
    ggml_backend_sched_graph_compute(sched, graph);
    ggml_backend_sched_synchronize(sched);

    ggml_free(ctx);
    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu_backend);
}









// Helper function to test unary operations
void test_unary_ops(struct gguf_context* ctx) {
    if (!ctx) return;

    struct ggml_tensor* src = NULL;
    const int64_t n_tensors = gguf_get_n_tensors(ctx);

    for (int64_t i = 0; i < n_tensors; i++) {
        src = ggml_new_tensor_1d(NULL, GGML_TYPE_F32, gguf_get_tensor_size(ctx, i));
        if (!src) continue;

        // Test various unary ops
        struct ggml_tensor* res = NULL;

        res = ggml_abs(NULL, src);
        res = ggml_sgn(NULL, src);
        res = ggml_neg(NULL, src);
        res = ggml_step(NULL, src);
        res = ggml_tanh(NULL, src);
        res = ggml_elu(NULL, src);
        res = ggml_relu(NULL, src);
        res = ggml_gelu(NULL, src);

        GGML_UNUSED(res);
    }
}

// Helper function to test binary operations
void test_binary_ops(struct gguf_context* ctx) {
    if (!ctx) return;

    struct ggml_tensor *a = NULL, *b = NULL;
    const int64_t n_tensors = gguf_get_n_tensors(ctx);

    for (int64_t i = 0; i < n_tensors - 1; i++) {
        a = ggml_new_tensor_1d(NULL, GGML_TYPE_F32, gguf_get_tensor_size(ctx, i));
        b = ggml_new_tensor_1d(NULL, GGML_TYPE_F32, gguf_get_tensor_size(ctx, i+1));
        if (!a || !b) continue;

        struct ggml_tensor* res = NULL;

        res = ggml_add(NULL, a, b);
        res = ggml_sub(NULL, a, b);
        res = ggml_mul(NULL, a, b);
        res = ggml_div(NULL, a, b);
        res = ggml_scale(NULL, a, 0.5f);

        GGML_UNUSED(res);
    }
}

// Helper function to test convolution operations
void test_conv_ops(struct gguf_context* ctx) {
    if (!ctx) return;

    struct ggml_tensor *a = NULL, *b = NULL;
    const int64_t n_tensors = gguf_get_n_tensors(ctx);

    for (int64_t i = 0; i < n_tensors - 1; i++) {
        // Create 2D tensors for conv tests
        a = ggml_new_tensor_2d(NULL, GGML_TYPE_F32, 8, 8);
        b = ggml_new_tensor_2d(NULL, GGML_TYPE_F32, 3, 3);
        if (!a || !b) continue;

        struct ggml_tensor* res = NULL;

        // Test 1D and 2D convolutions
        res = ggml_conv_1d(NULL, a, b, 1, 1, 1);
        res = ggml_conv_2d(NULL, a, b, 1, 1, 1, 1, 1, 1);
        res = ggml_conv_1d(NULL, a, b, 1, 1, 1);

        GGML_UNUSED(res);
    }
}

// Helper function to test tensor manipulation
void test_tensor_views(struct gguf_context* ctx) {
    if (!ctx) return;

    struct ggml_tensor* src = NULL;
    const int64_t n_tensors = gguf_get_n_tensors(ctx);

    for (int64_t i = 0; i < n_tensors; i++) {
        src = ggml_new_tensor_2d(NULL, GGML_TYPE_F32, 8, 8);
        if (!src) continue;

        struct ggml_tensor* res = NULL;

        // Test various view operations
        res = ggml_view_1d(NULL, src, 4, 0);
        res = ggml_view_2d(NULL, src, 4, 4, src->nb[1], 0);
        res = ggml_reshape(NULL, src, ggml_new_tensor_1d(NULL, src->type, ggml_nelements(src)));

        GGML_UNUSED(res);
    }
}

// Helper function to test quantization
void test_quantization_ops(struct gguf_context* ctx) {
    if (!ctx) return;

    struct ggml_tensor* src = NULL;
    const int64_t n_tensors = gguf_get_n_tensors(ctx);

    for (int64_t i = 0; i < n_tensors; i++) {
        src = ggml_new_tensor_1d(NULL, GGML_TYPE_F32, 32);
        if (!src) continue;

        // Test quantization to different formats
        void* quantized_data = malloc(ggml_type_size(GGML_TYPE_Q4_0) * ggml_nelements(src));
        if (!quantized_data) continue;

        ggml_quantize_chunk(GGML_TYPE_Q4_0, (float*)src->data, quantized_data, 0, 1, src->ne[0], NULL);
        ggml_quantize_chunk(GGML_TYPE_Q4_1, (float*)src->data, quantized_data, 0, 1, src->ne[0], NULL);
        ggml_quantize_chunk(GGML_TYPE_Q5_0, (float*)src->data, quantized_data, 0, 1, src->ne[0], NULL);

        free(quantized_data);
    }
}

// Helper function to test RoPE operations
void test_rope_ops(struct gguf_context* ctx) {
    if (!ctx) return;

    struct ggml_tensor* src = NULL;
    const int64_t n_tensors = gguf_get_n_tensors(ctx);

    for (int64_t i = 0; i < n_tensors; i++) {
        src = ggml_new_tensor_2d(NULL, GGML_TYPE_F32, 32, 32);
        if (!src) continue;

        struct ggml_tensor* pos = ggml_new_tensor_1d(NULL, GGML_TYPE_I32, src->ne[1]);
        if (!pos) continue;

        struct ggml_tensor* res = NULL;

        // Test different RoPE variants
        res = ggml_rope(NULL, src, pos, 32, 0);
        res = ggml_rope_ext(NULL, src, pos, NULL, 32, 0, 2048, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        GGML_UNUSED(res);
    }
}

// Helper function to test flash attention
void test_flash_attn_ops(struct gguf_context* ctx) {
    if (!ctx) return;

    // Create Q, K, V tensors for attention
    struct ggml_tensor* q = ggml_new_tensor_3d(NULL, GGML_TYPE_F32, 32, 8, 1);  // [n_embd, n_batch, n_head]
    struct ggml_tensor* k = ggml_new_tensor_3d(NULL, GGML_TYPE_F32, 32, 8, 1);  // [n_embd, n_kv, n_head_kv]
    struct ggml_tensor* v = ggml_new_tensor_3d(NULL, GGML_TYPE_F32, 32, 8, 1);  // [n_embd, n_kv, n_head_kv]

    if (!q || !k || !v) return;

    struct ggml_tensor* res = NULL;

    // Test flash attention with different parameters
    res = ggml_flash_attn_ext(NULL, q, k, v, NULL, 1.0f, 0.0f, 0.0f);

    GGML_UNUSED(res);
}

// Helper function to test pooling operations
void test_pool_ops(struct gguf_context* ctx) {
    if (!ctx) return;

    struct ggml_tensor* src = ggml_new_tensor_2d(NULL, GGML_TYPE_F32, 8, 8);
    if (!src) return;

    struct ggml_tensor* res = NULL;

    // Test different pooling operations
    res = ggml_pool_1d(NULL, src, GGML_OP_POOL_MAX, 2, 2, 0);
    res = ggml_pool_2d(NULL, src, GGML_OP_POOL_AVG, 2, 2, 2, 2, 0, 0);

    GGML_UNUSED(res);
}



extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size == 0 || size > 1000000) {
        return 0;
    }
    FILE* mem_file = fmemopen((void*)data, size, "rb");

    struct gguf_init_params params = {false, NULL};
    struct gguf_context* ctx = gguf_init_from_file_impl(mem_file, params);
    int iters = 0;
#if 0
    if (ctx != NULL) {
        while(++iters < 5) {

            std::vector<int8_t> write_buffer;

            switch(rand() % 19){
                // Test original operations
                case 0: test_kv_operations(ctx); break;
                 case 1: test_tensor_operations(ctx); break;
                 case 2: test_meta_operations(ctx); break;
                 case 3: test_backend_ops(cpu_backend, data + 100, size - 100); break;
                 case 4: test_backend_scheduler(data + 200, size - 200); break;

                // Test new operations
                 case 5: test_unary_ops(ctx); break;
                 case 6:  test_binary_ops(ctx); break;
                 case 7:  test_conv_ops(ctx); break;
                 case 8:  test_quantization_ops(ctx); break;
                case 9:  test_rope_ops(ctx); break;
                case 10:  test_flash_attn_ops(ctx); break;
                case 11:  test_pool_ops(ctx); break;

                // Test tensor manipulation
                case 12: test_tensor_views(ctx); break;
                case 13: test_tensor_permute(ctx); break;
                 case 14: test_tensor_reshape(ctx); break;
                case 15: test_tensor_split_merge(ctx); break;

                 case 16: gguf_write_to_buf(ctx, write_buffer, false); break;
                case 17: gguf_write_to_buf(ctx, write_buffer, true); break;
                default:  break;
            }
        }
    }
#endif
    if(ctx != NULL) gguf_free(ctx);
    fclose(mem_file);
    return 0;
}
