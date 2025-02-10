#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <string>

// Include required GGUF headers
#include "ggml.h"
#include "ggml-backend.h"
#include "../src/ggml-impl.h"
#include "gguf.h"

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

// Main fuzzing function that AFL will repeatedly call
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    // Ignore empty or oversized inputs
    if (size == 0 || size > 100000) {
        return 0;
    }

    // Create a FILE* from the input buffer
    FILE* mem_file = fmemopen((void*)data, size, "rb");
    if (!mem_file) {
        return 0;
    }

    // Try different initialization parameters
    struct gguf_init_params params;
    params.no_alloc = false;
    params.ctx = NULL;

    // Initialize GGUF context from the file
    struct gguf_context* ctx = gguf_init_from_file_impl(mem_file, params);

    if (ctx != NULL) {
        // Test various operations on the context
        test_kv_operations(ctx);
        test_tensor_operations(ctx);
        test_meta_operations(ctx);

        // Test writing operations
        std::vector<int8_t> write_buffer;
        gguf_write_to_buf(ctx, write_buffer, false);  // Try full write
        gguf_write_to_buf(ctx, write_buffer, true);   // Try metadata-only write

        // Free the context
        gguf_free(ctx);
    }

    // Try with no_alloc = true
    fseek(mem_file, 0, SEEK_SET);
    params.no_alloc = true;
    ctx = gguf_init_from_file_impl(mem_file, params);
    if (ctx != NULL) {
        gguf_free(ctx);
    }

    fclose(mem_file);
    return 0;
}
