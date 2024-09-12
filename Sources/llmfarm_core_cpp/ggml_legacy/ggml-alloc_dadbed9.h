#pragma once

#include "ggml_dadbed9.h"

#ifdef  __cplusplus
extern "C" {
#endif


GGML_dadbed9_API struct ggml_dadbed9_allocr * ggml_dadbed9_allocr_new(void * data, size_t size, size_t alignment);
GGML_dadbed9_API struct ggml_dadbed9_allocr * ggml_dadbed9_allocr_new_measure(size_t alignment);

// tell the allocator to parse nodes following the order described in the list
// you should call this if your graph are optimized to execute out-of-order
GGML_dadbed9_API void   ggml_dadbed9_allocr_set_parse_seq(struct ggml_dadbed9_allocr * alloc, int * list, int n);

GGML_dadbed9_API void   ggml_dadbed9_allocr_free(struct ggml_dadbed9_allocr * alloc);
GGML_dadbed9_API bool   ggml_dadbed9_allocr_is_measure(struct ggml_dadbed9_allocr * alloc);
GGML_dadbed9_API void   ggml_dadbed9_allocr_reset(struct ggml_dadbed9_allocr * alloc);
GGML_dadbed9_API void   ggml_dadbed9_allocr_alloc(struct ggml_dadbed9_allocr * alloc, struct ggml_dadbed9_tensor * tensor);
GGML_dadbed9_API size_t ggml_dadbed9_allocr_alloc_graph(struct ggml_dadbed9_allocr * alloc, struct ggml_dadbed9_cgraph * graph);


#ifdef  __cplusplus
}
#endif
