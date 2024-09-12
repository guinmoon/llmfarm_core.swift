#pragma once

#include "ggml_d925ed.h"

#ifdef  __cplusplus
extern "C" {
#endif


GGML_d925ed_API struct ggml_d925ed_allocr * ggml_d925ed_allocr_new(void * data, size_t size, size_t alignment);
GGML_d925ed_API struct ggml_d925ed_allocr * ggml_d925ed_allocr_new_measure(size_t alignment);

// tell the allocator to parse nodes following the order described in the list
// you should call this if your graph are optimized to execute out-of-order
GGML_d925ed_API void   ggml_d925ed_allocr_set_parse_seq(struct ggml_d925ed_allocr * alloc, const int * list, int n);

GGML_d925ed_API void   ggml_d925ed_allocr_free(struct ggml_d925ed_allocr * alloc);
GGML_d925ed_API bool   ggml_d925ed_allocr_is_measure(struct ggml_d925ed_allocr * alloc);
GGML_d925ed_API void   ggml_d925ed_allocr_reset(struct ggml_d925ed_allocr * alloc);
GGML_d925ed_API void   ggml_d925ed_allocr_alloc(struct ggml_d925ed_allocr * alloc, struct ggml_d925ed_tensor * tensor);
GGML_d925ed_API size_t ggml_d925ed_allocr_alloc_graph(struct ggml_d925ed_allocr * alloc, struct ggml_d925ed_cgraph * graph);


#ifdef  __cplusplus
}
#endif
