#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <time.h>

#include "gguf.h"

#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
#include <sanitizer/msan_interface.h>
#endif
#endif



struct gguf_context;
struct ggml_context;

// These functions are defined in the GGUF implementation
extern struct gguf_context * gguf_init_from_file_impl(FILE * file, struct gguf_init_params params);
extern void gguf_free(struct gguf_context * ctx);
extern void ggml_free(struct ggml_context * ctx);

// Main fuzzer entry point
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    // Skip empty inputs
    if (size == 0) {
        return 0;
    }
    #if defined(__has_feature)
    #if __has_feature(memory_sanitizer)
    __msan_unpoison(data, size);
    #endif
    #endif
    // Create a FILE* from the input buffer using fmemopen
    FILE *file = fmemopen((void*)data, size, "rb");
    if (!file) {
        return 0;
    }
    srand(time(0));
    int num = rand()%3;
    // Test 1: Try to read with no_alloc = true
    if(num==0)
    {
        struct ggml_context *ctx = nullptr;
        struct gguf_init_params params = {
            .no_alloc = true,
            .ctx = &ctx
        };

        struct gguf_context *gguf_ctx = gguf_init_from_file_impl(file, params);
                gguf_free(gguf_ctx);

        return 0;
        // Reset file position for next test
        rewind(file);
    }

    // Test 2: Try to read with no_alloc = false (will allocate memory for tensor data)
    if(num==1)
    {
        struct ggml_context *ctx = nullptr;
        struct gguf_init_params params = {
            .no_alloc = false,
            .ctx = &ctx
        };

        struct gguf_context *gguf_ctx = gguf_init_from_file_impl(file, params);
        gguf_free(gguf_ctx);
        
return 0;
        // Reset file position for next test
        rewind(file);
    }

    // Test 3: Try to read with nullptr ctx (should still parse metadata)
if(num==2)    
{
        struct gguf_init_params params = {
            .no_alloc = false,
            .ctx = nullptr
        };

        struct gguf_context *gguf_ctx = gguf_init_from_file_impl(file, params);
        
        if (gguf_ctx) {
            // If successful, we can test some of the getter functions
            // Note: These are extern declarations - you'd need to declare them
            // based on what's available in the actual implementation
            
            // Clean up
        }
    }

    // Close the file
    fclose(file);

    return 0;
}

// Optional: Initialize the fuzzer (if needed)
extern "C" int LLVMFuzzerInitialize(int *argc, char ***argv) {
    // Any one-time initialization can go here
    return 0;
}

