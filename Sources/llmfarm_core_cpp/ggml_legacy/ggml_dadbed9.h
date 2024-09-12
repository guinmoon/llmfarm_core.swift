#pragma once

//
// GGML Tensor Library
//
// This documentation is still a work in progress.
// If you wish some specific topics to be covered, feel free to drop a comment:
//
//   https://github.com/ggerganov/whisper.cpp/issues/40
//
// ## Overview
//
// This library implements:
//
//  - a set of tensor operations
//  - automatic differentiation
//  - basic optimization algorithms
//
// The aim of this library is to provide a minimalistic approach for various machine learning tasks. This includes,
// but is not limited to, the following:
//
//  - linear regression
//  - support vector machines
//  - neural networks
//
// The library allows the user to define a certain function using the available tensor operations. This function
// definition is represented internally via a computation graph. Each tensor operation in the function definition
// corresponds to a node in the graph. Having the computation graph defined, the user can choose to compute the
// function's value and/or its gradient with respect to the input variables. Optionally, the function can be optimized
// using one of the available optimization algorithms.
//
// For example, here we define the function: f(x) = a*x^2 + b
//
//   {
//       struct ggml_dadbed9_init_params params = {
//           .mem_size   = 16*1024*1024,
//           .mem_buffer = NULL,
//       };
//
//       // memory allocation happens here
//       struct ggml_dadbed9_context * ctx = ggml_dadbed9_init(params);
//
//       struct ggml_dadbed9_tensor * x = ggml_dadbed9_new_tensor_1d(ctx, GGML_dadbed9_TYPE_F32, 1);
//
//       ggml_dadbed9_set_param(ctx, x); // x is an input variable
//
//       struct ggml_dadbed9_tensor * a  = ggml_dadbed9_new_tensor_1d(ctx, GGML_dadbed9_TYPE_F32, 1);
//       struct ggml_dadbed9_tensor * b  = ggml_dadbed9_new_tensor_1d(ctx, GGML_dadbed9_TYPE_F32, 1);
//       struct ggml_dadbed9_tensor * x2 = ggml_dadbed9_mul(ctx, x, x);
//       struct ggml_dadbed9_tensor * f  = ggml_dadbed9_add(ctx, ggml_dadbed9_mul(ctx, a, x2), b);
//
//       ...
//   }
//
// Notice that the function definition above does not involve any actual computation. The computation is performed only
// when the user explicitly requests it. For example, to compute the function's value at x = 2.0:
//
//   {
//       ...
//
//       struct ggml_dadbed9_cgraph gf = ggml_dadbed9_build_forward(f);
//
//       // set the input variable and parameter values
//       ggml_dadbed9_set_f32(x, 2.0f);
//       ggml_dadbed9_set_f32(a, 3.0f);
//       ggml_dadbed9_set_f32(b, 4.0f);
//
//       ggml_dadbed9_graph_compute_with_ctx(ctx, &gf, n_threads);
//
//       printf("f = %f\n", ggml_dadbed9_get_f32_1d(f, 0));
//
//       ...
//   }
//
// The actual computation is performed in the ggml_dadbed9_graph_compute() function.
//
// The ggml_dadbed9_new_tensor_...() functions create new tensors. They are allocated in the memory buffer provided to the
// ggml_dadbed9_init() function. You have to be careful not to exceed the memory buffer size. Therefore, you have to know
// in advance how much memory you need for your computation. Alternatively, you can allocate a large enough memory
// and after defining the computation graph, call the ggml_dadbed9_used_mem() function to find out how much memory was
// actually needed.
//
// The ggml_dadbed9_set_param() function marks a tensor as an input variable. This is used by the automatic
// differentiation and optimization algorithms.
//
// The described approach allows to define the function graph once and then compute its forward or backward graphs
// multiple times. All computations will use the same memory buffer allocated in the ggml_dadbed9_init() function. This way
// the user can avoid the memory allocation overhead at runtime.
//
// The library supports multi-dimensional tensors - up to 4 dimensions. The FP16 and FP32 data types are first class
// citizens, but in theory the library can be extended to support FP8 and integer data types.
//
// Each tensor operation produces a new tensor. Initially the library was envisioned to support only the use of unary
// and binary operations. Most of the available operations fall into one of these two categories. With time, it became
// clear that the library needs to support more complex operations. The way to support these operations is not clear
// yet, but a few examples are demonstrated in the following operations:
//
//   - ggml_dadbed9_permute()
//   - ggml_dadbed9_conv_1d_1s()
//   - ggml_dadbed9_conv_1d_2s()
//
// For each tensor operator, the library implements a forward and backward computation function. The forward function
// computes the output tensor value given the input tensor values. The backward function computes the adjoint of the
// input tensors given the adjoint of the output tensor. For a detailed explanation of what this means, take a
// calculus class, or watch the following video:
//
//   What is Automatic Differentiation?
//   https://www.youtube.com/watch?v=wG_nF1awSSY
//
//
// ## Tensor data (struct ggml_dadbed9_tensor)
//
// The tensors are stored in memory via the ggml_dadbed9_tensor struct. The structure provides information about the size of
// the tensor, the data type, and the memory buffer where the tensor data is stored. Additionally, it contains
// pointers to the "source" tensors - i.e. the tensors that were used to compute the current tensor. For example:
//
//   {
//       struct ggml_dadbed9_tensor * c = ggml_dadbed9_add(ctx, a, b);
//
//       assert(c->src[0] == a);
//       assert(c->src[1] == b);
//   }
//
// The multi-dimensional tensors are stored in row-major order. The ggml_dadbed9_tensor struct contains fields for the
// number of elements in each dimension ("ne") as well as the number of bytes ("nb", a.k.a. stride). This allows
// to store tensors that are not contiguous in memory, which is useful for operations such as transposition and
// permutation. All tensor operations have to take the stride into account and not assume that the tensor is
// contiguous in memory.
//
// The data of the tensor is accessed via the "data" pointer. For example:
//
//   {
//       struct ggml_dadbed9_tensor * a = ggml_dadbed9_new_tensor_2d(ctx, GGML_dadbed9_TYPE_F32, 2, 3);
//
//       // a[2, 1] = 1.0f;
//       *(float *) ((char *) a->data + 2*a->nb[1] + 1*a->nb[0]) = 1.0f;
//
//       // a[0, 2] = 2.0f;
//       *(float *) ((char *) a->data + 0*a->nb[1] + 2*a->nb[0]) = 2.0f;
//
//       ...
//   }
//
// Alternatively, there are helper functions, such as ggml_dadbed9_get_f32_1d() and ggml_dadbed9_set_f32_1d() that can be used.
//
// ## The matrix multiplication operator (ggml_dadbed9_mul_mat)
//
// TODO
//
//
// ## Multi-threading
//
// TODO
//
//
// ## Overview of ggml.c
//
// TODO
//
//
// ## SIMD optimizations
//
// TODO
//
//
// ## Debugging ggml
//
// TODO
//
//

#ifdef GGML_dadbed9_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef GGML_dadbed9_BUILD
#            define GGML_dadbed9_API __declspec(dllexport)
#        else
#            define GGML_dadbed9_API __declspec(dllimport)
#        endif
#    else
#        define GGML_dadbed9_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define GGML_dadbed9_API
#endif

// TODO: support for clang
#ifdef __GNUC__
#    define GGML_dadbed9_DEPRECATED(func, hint) func __attribute__((deprecated(hint)))
#elif defined(_MSC_VER)
#    define GGML_dadbed9_DEPRECATED(func, hint) __declspec(deprecated(hint)) func
#else
#    define GGML_dadbed9_DEPRECATED(func, hint) func
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define GGML_dadbed9_FILE_MAGIC   0x67676d6c // "ggml"
#define GGML_dadbed9_FILE_VERSION 1

#define GGML_dadbed9_QNT_VERSION        2    // bump this on quantization format changes
#define GGML_dadbed9_QNT_VERSION_FACTOR 1000 // do not change this

#define GGML_dadbed9_MAX_DIMS          4
#define GGML_dadbed9_MAX_NODES         4096
#define GGML_dadbed9_MAX_PARAMS        256
#define GGML_dadbed9_MAX_CONTEXTS      64
#define GGML_dadbed9_MAX_SRC           6
#define GGML_dadbed9_MAX_NAME          48
#define GGML_dadbed9_MAX_OP_PARAMS     32
#define GGML_dadbed9_DEFAULT_N_THREADS 4


#define GGML_dadbed9_EXIT_SUCCESS 0
#define GGML_dadbed9_EXIT_ABORTED 1

#define GGML_dadbed9_UNUSED(x) (void)(x)

#define GGML_dadbed9_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))

//#define GGML_dadbed9_ASSERT(x) \
//    do { \
//        if (!(x)) { \
//            fprintf(stderr, "GGML_dadbed9_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
//            abort(); \
//        } \
//    } while (0)

#include "exception_helper.h"

#define GGML_dadbed9_ASSERT(x) \
    do { \
        if (!(x)) { \
            fprintf(stderr, "GGML_dadbed9_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
            char descr[500]; \
            sprintf(descr, "GGML_dadbed9_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x);\
            throw_exception(descr); \
        } \
    } while (0)

// used to copy the number of elements and stride in bytes of tensors into local variables.
// main purpose is to reduce code duplication and improve readability.
//
// example:
//
//    GGML_dadbed9_TENSOR_LOCALS(int64_t, ne1, src1, ne);
//    GGML_dadbed9_TENSOR_LOCALS(size_t,  nb1, src1, nb);
//
#define GGML_dadbed9_TENSOR_LOCALS_1(type, prefix, pointer, array) \
    const type prefix##0 = (pointer)->array[0]; \
    GGML_dadbed9_UNUSED(prefix##0);
#define GGML_dadbed9_TENSOR_LOCALS_2(type, prefix, pointer, array) \
    GGML_dadbed9_TENSOR_LOCALS_1    (type, prefix, pointer, array) \
    const type prefix##1 = (pointer)->array[1]; \
    GGML_dadbed9_UNUSED(prefix##1);
#define GGML_dadbed9_TENSOR_LOCALS_3(type, prefix, pointer, array) \
    GGML_dadbed9_TENSOR_LOCALS_2    (type, prefix, pointer, array) \
    const type prefix##2 = (pointer)->array[2]; \
    GGML_dadbed9_UNUSED(prefix##2);
#define GGML_dadbed9_TENSOR_LOCALS(type, prefix, pointer, array) \
    GGML_dadbed9_TENSOR_LOCALS_3  (type, prefix, pointer, array) \
    const type prefix##3 = (pointer)->array[3]; \
    GGML_dadbed9_UNUSED(prefix##3);

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef __ARM_NEON
    // we use the built-in 16-bit float type
    typedef __fp16 ggml_dadbed9_fp16_t;
#else
    typedef uint16_t ggml_dadbed9_fp16_t;
#endif

    // convert FP16 <-> FP32
    GGML_dadbed9_API float       ggml_dadbed9_fp16_to_fp32(ggml_dadbed9_fp16_t x);
    GGML_dadbed9_API ggml_dadbed9_fp16_t ggml_dadbed9_fp32_to_fp16(float x);

    GGML_dadbed9_API void ggml_dadbed9_fp16_to_fp32_row(const ggml_dadbed9_fp16_t * x, float * y, int n);
    GGML_dadbed9_API void ggml_dadbed9_fp32_to_fp16_row(const float * x, ggml_dadbed9_fp16_t * y, int n);

    struct ggml_dadbed9_object;
    struct ggml_dadbed9_context;

    enum ggml_dadbed9_type {
        GGML_dadbed9_TYPE_F32  = 0,
        GGML_dadbed9_TYPE_F16  = 1,
        GGML_dadbed9_TYPE_Q4_0 = 2,
        GGML_dadbed9_TYPE_Q4_1 = 3,
        // GGML_dadbed9_TYPE_Q4_2 = 4, support has been removed
        // GGML_dadbed9_TYPE_Q4_3 (5) support has been removed
        GGML_dadbed9_TYPE_Q5_0 = 6,
        GGML_dadbed9_TYPE_Q5_1 = 7,
        GGML_dadbed9_TYPE_Q8_0 = 8,
        GGML_dadbed9_TYPE_Q8_1 = 9,
        // k-quantizations
        GGML_dadbed9_TYPE_Q2_K = 10,
        GGML_dadbed9_TYPE_Q3_K = 11,
        GGML_dadbed9_TYPE_Q4_K = 12,
        GGML_dadbed9_TYPE_Q5_K = 13,
        GGML_dadbed9_TYPE_Q6_K = 14,
        GGML_dadbed9_TYPE_Q8_K = 15,
        GGML_dadbed9_TYPE_I8,
        GGML_dadbed9_TYPE_I16,
        GGML_dadbed9_TYPE_I32,
        GGML_dadbed9_TYPE_COUNT,
    };

    enum ggml_dadbed9_backend {
        GGML_dadbed9_BACKEND_CPU = 0,
        GGML_dadbed9_BACKEND_GPU = 10,
        GGML_dadbed9_BACKEND_GPU_SPLIT = 20,
    };

    // model file types
    enum ggml_dadbed9_ftype {
        GGML_dadbed9_FTYPE_UNKNOWN     = -1,
        GGML_dadbed9_FTYPE_ALL_F32     = 0,
        GGML_dadbed9_FTYPE_MOSTLY_F16  = 1,  // except 1d tensors
        GGML_dadbed9_FTYPE_MOSTLY_Q4_0 = 2,  // except 1d tensors
        GGML_dadbed9_FTYPE_MOSTLY_Q4_1 = 3,  // except 1d tensors
        GGML_dadbed9_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4, // tok_embeddings.weight and output.weight are F16
        GGML_dadbed9_FTYPE_MOSTLY_Q8_0 = 7,  // except 1d tensors
        GGML_dadbed9_FTYPE_MOSTLY_Q5_0 = 8,  // except 1d tensors
        GGML_dadbed9_FTYPE_MOSTLY_Q5_1 = 9,  // except 1d tensors
        GGML_dadbed9_FTYPE_MOSTLY_Q2_K = 10, // except 1d tensors
        GGML_dadbed9_FTYPE_MOSTLY_Q3_K = 11, // except 1d tensors
        GGML_dadbed9_FTYPE_MOSTLY_Q4_K = 12, // except 1d tensors
        GGML_dadbed9_FTYPE_MOSTLY_Q5_K = 13, // except 1d tensors
        GGML_dadbed9_FTYPE_MOSTLY_Q6_K = 14, // except 1d tensors
    };

    // available tensor operations:
    enum ggml_dadbed9_op {
        GGML_dadbed9_OP_NONE = 0,

        GGML_dadbed9_OP_DUP,
        GGML_dadbed9_OP_ADD,
        GGML_dadbed9_OP_ADD1,
        GGML_dadbed9_OP_ACC,
        GGML_dadbed9_OP_SUB,
        GGML_dadbed9_OP_MUL,
        GGML_dadbed9_OP_DIV,
        GGML_dadbed9_OP_SQR,
        GGML_dadbed9_OP_SQRT,
        GGML_dadbed9_OP_LOG,
        GGML_dadbed9_OP_SUM,
        GGML_dadbed9_OP_SUM_ROWS,
        GGML_dadbed9_OP_MEAN,
        GGML_dadbed9_OP_ARGMAX,
        GGML_dadbed9_OP_REPEAT,
        GGML_dadbed9_OP_REPEAT_BACK,
        GGML_dadbed9_OP_SILU_BACK,
        GGML_dadbed9_OP_NORM, // normalize
        GGML_dadbed9_OP_RMS_NORM,
        GGML_dadbed9_OP_RMS_NORM_BACK,

        GGML_dadbed9_OP_MUL_MAT,
        GGML_dadbed9_OP_OUT_PROD,

        GGML_dadbed9_OP_SCALE,
        GGML_dadbed9_OP_SET,
        GGML_dadbed9_OP_CPY,
        GGML_dadbed9_OP_CONT,
        GGML_dadbed9_OP_RESHAPE,
        GGML_dadbed9_OP_VIEW,
        GGML_dadbed9_OP_PERMUTE,
        GGML_dadbed9_OP_TRANSPOSE,
        GGML_dadbed9_OP_GET_ROWS,
        GGML_dadbed9_OP_GET_ROWS_BACK,
        GGML_dadbed9_OP_DIAG,
        GGML_dadbed9_OP_DIAG_MASK_INF,
        GGML_dadbed9_OP_DIAG_MASK_ZERO,
        GGML_dadbed9_OP_SOFT_MAX,
        GGML_dadbed9_OP_SOFT_MAX_BACK,
        GGML_dadbed9_OP_ROPE,
        GGML_dadbed9_OP_ROPE_BACK,
        GGML_dadbed9_OP_ALIBI,
        GGML_dadbed9_OP_CLAMP,
        GGML_dadbed9_OP_CONV_1D,
        GGML_dadbed9_OP_CONV_2D,
        GGML_dadbed9_OP_POOL_1D,
        GGML_dadbed9_OP_POOL_2D,

        GGML_dadbed9_OP_FLASH_ATTN,
        GGML_dadbed9_OP_FLASH_FF,
        GGML_dadbed9_OP_FLASH_ATTN_BACK,
        GGML_dadbed9_OP_WIN_PART,
        GGML_dadbed9_OP_WIN_UNPART,

        GGML_dadbed9_OP_UNARY,

        GGML_dadbed9_OP_MAP_UNARY,
        GGML_dadbed9_OP_MAP_BINARY,

        GGML_dadbed9_OP_MAP_CUSTOM1_F32,
        GGML_dadbed9_OP_MAP_CUSTOM2_F32,
        GGML_dadbed9_OP_MAP_CUSTOM3_F32,

        GGML_dadbed9_OP_MAP_CUSTOM1,
        GGML_dadbed9_OP_MAP_CUSTOM2,
        GGML_dadbed9_OP_MAP_CUSTOM3,

        GGML_dadbed9_OP_CROSS_ENTROPY_LOSS,
        GGML_dadbed9_OP_CROSS_ENTROPY_LOSS_BACK,

        GGML_dadbed9_OP_COUNT,
    };

    enum ggml_dadbed9_unary_op {
        GGML_dadbed9_UNARY_OP_ABS,
        GGML_dadbed9_UNARY_OP_SGN,
        GGML_dadbed9_UNARY_OP_NEG,
        GGML_dadbed9_UNARY_OP_STEP,
        GGML_dadbed9_UNARY_OP_TANH,
        GGML_dadbed9_UNARY_OP_ELU,
        GGML_dadbed9_UNARY_OP_RELU,
        GGML_dadbed9_UNARY_OP_GELU,
        GGML_dadbed9_UNARY_OP_GELU_QUICK,
        GGML_dadbed9_UNARY_OP_SILU,
    };

    enum ggml_dadbed9_object_type {
        GGML_dadbed9_OBJECT_TENSOR,
        GGML_dadbed9_OBJECT_GRAPH,
        GGML_dadbed9_OBJECT_WORK_BUFFER
    };

    // ggml object
    struct ggml_dadbed9_object {
        size_t offs;
        size_t size;

        struct ggml_dadbed9_object * next;

        enum ggml_dadbed9_object_type type;

        char padding[4];
    };

    static const size_t GGML_dadbed9_OBJECT_SIZE = sizeof(struct ggml_dadbed9_object);

    // n-dimensional tensor
    struct ggml_dadbed9_tensor {
        enum ggml_dadbed9_type    type;
        enum ggml_dadbed9_backend backend;

        int     n_dims;
        int64_t ne[GGML_dadbed9_MAX_DIMS]; // number of elements
        size_t  nb[GGML_dadbed9_MAX_DIMS]; // stride in bytes:
                                   // nb[0] = sizeof(type)
                                   // nb[1] = nb[0]   * ne[0] + padding
                                   // nb[i] = nb[i-1] * ne[i-1]

        // compute data
        enum ggml_dadbed9_op op;

        // op params - allocated as int32_t for alignment
        int32_t op_params[GGML_dadbed9_MAX_OP_PARAMS / sizeof(int32_t)];

        bool is_param;

        struct ggml_dadbed9_tensor * grad;
        struct ggml_dadbed9_tensor * src[GGML_dadbed9_MAX_SRC];

        // performance
        int     perf_runs;
        int64_t perf_cycles;
        int64_t perf_time_us;

        void * data;

        char name[GGML_dadbed9_MAX_NAME];

        void * extra; // extra things e.g. for ggml-cuda.cu

        char padding[4];
    };

    static const size_t GGML_dadbed9_TENSOR_SIZE = sizeof(struct ggml_dadbed9_tensor);

    // the compute plan that needs to be prepared for ggml_dadbed9_graph_compute()
    // since https://github.com/ggerganov/ggml/issues/287
    struct ggml_dadbed9_cplan {
        size_t    work_size; // size of work buffer, calculated by `ggml_dadbed9_graph_plan()`
        uint8_t * work_data; // work buffer, to be allocated by caller before calling to `ggml_dadbed9_graph_compute()`

        int n_threads;

        // the `n_tasks` of nodes, 1:1 mapping to cgraph nodes
        int n_tasks[GGML_dadbed9_MAX_NODES];

        // abort ggml_dadbed9_graph_compute when true
        bool (*abort_callback)(void * data);
        void * abort_callback_data;
    };

    // next prime after GGML_dadbed9_MAX_NODES
    // #define GGML_dadbed9_GRAPH_HASHTABLE_SIZE 4099
    // next prime after GGML_dadbed9_MAX_NODES * 2 (nodes + leafs)
    #define GGML_dadbed9_GRAPH_HASHTABLE_SIZE 8273

    // computation graph
    struct ggml_dadbed9_cgraph {
        int n_nodes;
        int n_leafs;

        struct ggml_dadbed9_tensor * nodes[GGML_dadbed9_MAX_NODES];
        struct ggml_dadbed9_tensor * grads[GGML_dadbed9_MAX_NODES];
        struct ggml_dadbed9_tensor * leafs[GGML_dadbed9_MAX_NODES];

        void * visited_hash_table[GGML_dadbed9_GRAPH_HASHTABLE_SIZE];

        // performance
        int     perf_runs;
        int64_t perf_cycles;
        int64_t perf_time_us;
    };

    static const size_t GGML_dadbed9_GRAPH_SIZE = sizeof(struct ggml_dadbed9_cgraph);

    // scratch buffer
    struct ggml_dadbed9_scratch {
        size_t offs;
        size_t size;
        void * data;
    };

    struct ggml_dadbed9_init_params {
        // memory pool
        size_t mem_size;   // bytes
        void * mem_buffer; // if NULL, memory will be allocated internally
        bool   no_alloc;   // don't allocate memory for the tensor data
    };


    // compute types

    // NOTE: the INIT or FINALIZE pass is not scheduled unless explicitly enabled.
    // This behavior was changed since https://github.com/ggerganov/llama.cpp/pull/1995.
    enum ggml_dadbed9_task_type {
        GGML_dadbed9_TASK_INIT = 0,
        GGML_dadbed9_TASK_COMPUTE,
        GGML_dadbed9_TASK_FINALIZE,
    };

    struct ggml_dadbed9_compute_params {
        enum ggml_dadbed9_task_type type;

        // ith = thread index, nth = number of threads
        int ith, nth;

        // work buffer for all threads
        size_t wsize;
        void * wdata;
    };

    // misc

    GGML_dadbed9_API void    ggml_dadbed9_time_init(void); // call this once at the beginning of the program
    GGML_dadbed9_API int64_t ggml_dadbed9_time_ms(void);
    GGML_dadbed9_API int64_t ggml_dadbed9_time_us(void);
    GGML_dadbed9_API int64_t ggml_dadbed9_cycles(void);
    GGML_dadbed9_API int64_t ggml_dadbed9_cycles_per_ms(void);

    GGML_dadbed9_API void    ggml_dadbed9_numa_init(void); // call once for better performance on NUMA systems
    GGML_dadbed9_API bool    ggml_dadbed9_is_numa(void); // true if init detected that system has >1 NUMA node

    GGML_dadbed9_API void    ggml_dadbed9_print_object (const struct ggml_dadbed9_object * obj);
    GGML_dadbed9_API void    ggml_dadbed9_print_objects(const struct ggml_dadbed9_context * ctx);

    GGML_dadbed9_API int64_t ggml_dadbed9_nelements   (const struct ggml_dadbed9_tensor * tensor);
    GGML_dadbed9_API int64_t ggml_dadbed9_nrows       (const struct ggml_dadbed9_tensor * tensor);
    GGML_dadbed9_API size_t  ggml_dadbed9_nbytes      (const struct ggml_dadbed9_tensor * tensor);
    GGML_dadbed9_API size_t  ggml_dadbed9_nbytes_split(const struct ggml_dadbed9_tensor * tensor, int nrows_split);

    GGML_dadbed9_API int     ggml_dadbed9_blck_size (enum ggml_dadbed9_type type);
    GGML_dadbed9_API size_t  ggml_dadbed9_type_size (enum ggml_dadbed9_type type); // size in bytes for all elements in a block
    GGML_dadbed9_API float   ggml_dadbed9_type_sizef(enum ggml_dadbed9_type type); // ggml_dadbed9_type_size()/ggml_dadbed9_blck_size() as float

    GGML_dadbed9_API const char * ggml_dadbed9_type_name(enum ggml_dadbed9_type type);
    GGML_dadbed9_API const char * ggml_dadbed9_op_name  (enum ggml_dadbed9_op   op);
    GGML_dadbed9_API const char * ggml_dadbed9_op_symbol(enum ggml_dadbed9_op   op);

    GGML_dadbed9_API size_t  ggml_dadbed9_element_size(const struct ggml_dadbed9_tensor * tensor);

    GGML_dadbed9_API bool    ggml_dadbed9_is_quantized(enum ggml_dadbed9_type type);

    // TODO: temporary until model loading of ggml examples is refactored
    GGML_dadbed9_API enum ggml_dadbed9_type ggml_dadbed9_ftype_to_ggml_dadbed9_type(enum ggml_dadbed9_ftype ftype);

    GGML_dadbed9_API bool ggml_dadbed9_is_transposed(const struct ggml_dadbed9_tensor * tensor);
    GGML_dadbed9_API bool ggml_dadbed9_is_contiguous(const struct ggml_dadbed9_tensor * tensor);
    GGML_dadbed9_API bool ggml_dadbed9_is_permuted  (const struct ggml_dadbed9_tensor * tensor);

    GGML_dadbed9_API bool ggml_dadbed9_are_same_shape(const struct ggml_dadbed9_tensor * t0, const struct ggml_dadbed9_tensor * t1);

    // use this to compute the memory overhead of a tensor
    GGML_dadbed9_API size_t ggml_dadbed9_tensor_overhead(void);

    // main

    GGML_dadbed9_API struct ggml_dadbed9_context * ggml_dadbed9_init(struct ggml_dadbed9_init_params params);
    GGML_dadbed9_API void                  ggml_dadbed9_free(struct ggml_dadbed9_context * ctx);

    GGML_dadbed9_API size_t  ggml_dadbed9_used_mem(const struct ggml_dadbed9_context * ctx);

    GGML_dadbed9_API size_t  ggml_dadbed9_set_scratch (struct ggml_dadbed9_context * ctx, struct ggml_dadbed9_scratch scratch);
    GGML_dadbed9_API bool    ggml_dadbed9_get_no_alloc(struct ggml_dadbed9_context * ctx);
    GGML_dadbed9_API void    ggml_dadbed9_set_no_alloc(struct ggml_dadbed9_context * ctx, bool no_alloc);

    GGML_dadbed9_API void *  ggml_dadbed9_get_mem_buffer     (const struct ggml_dadbed9_context * ctx);
    GGML_dadbed9_API size_t  ggml_dadbed9_get_mem_size       (const struct ggml_dadbed9_context * ctx);
    GGML_dadbed9_API size_t  ggml_dadbed9_get_max_tensor_size(const struct ggml_dadbed9_context * ctx);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_new_tensor(
            struct ggml_dadbed9_context * ctx,
            enum   ggml_dadbed9_type type,
            int    n_dims,
            const int64_t *ne);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_new_tensor_1d(
            struct ggml_dadbed9_context * ctx,
            enum   ggml_dadbed9_type type,
            int64_t ne0);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_new_tensor_2d(
            struct ggml_dadbed9_context * ctx,
            enum   ggml_dadbed9_type type,
            int64_t ne0,
            int64_t ne1);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_new_tensor_3d(
            struct ggml_dadbed9_context * ctx,
            enum   ggml_dadbed9_type type,
            int64_t ne0,
            int64_t ne1,
            int64_t ne2);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_new_tensor_4d(
            struct ggml_dadbed9_context * ctx,
            enum   ggml_dadbed9_type type,
            int64_t ne0,
            int64_t ne1,
            int64_t ne2,
            int64_t ne3);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_new_i32(struct ggml_dadbed9_context * ctx, int32_t value);
    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_new_f32(struct ggml_dadbed9_context * ctx, float value);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_dup_tensor (struct ggml_dadbed9_context * ctx, const struct ggml_dadbed9_tensor * src);
    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_view_tensor(struct ggml_dadbed9_context * ctx, const struct ggml_dadbed9_tensor * src);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_get_tensor(struct ggml_dadbed9_context * ctx, const char * name);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_set_zero(struct ggml_dadbed9_tensor * tensor);
    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_set_i32 (struct ggml_dadbed9_tensor * tensor, int32_t value);
    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_set_f32 (struct ggml_dadbed9_tensor * tensor, float value);

    GGML_dadbed9_API int32_t ggml_dadbed9_get_i32_1d(const struct ggml_dadbed9_tensor * tensor, int i);
    GGML_dadbed9_API void    ggml_dadbed9_set_i32_1d(const struct ggml_dadbed9_tensor * tensor, int i, int32_t value);

    GGML_dadbed9_API float   ggml_dadbed9_get_f32_1d(const struct ggml_dadbed9_tensor * tensor, int i);
    GGML_dadbed9_API void    ggml_dadbed9_set_f32_1d(const struct ggml_dadbed9_tensor * tensor, int i, float value);

    GGML_dadbed9_API void *  ggml_dadbed9_get_data    (const struct ggml_dadbed9_tensor * tensor);
    GGML_dadbed9_API float * ggml_dadbed9_get_data_f32(const struct ggml_dadbed9_tensor * tensor);

    GGML_dadbed9_API enum ggml_dadbed9_unary_op ggml_dadbed9_get_unary_op(const struct ggml_dadbed9_tensor * tensor);

    GGML_dadbed9_API const char *         ggml_dadbed9_get_name   (const struct ggml_dadbed9_tensor * tensor);
    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_set_name   (      struct ggml_dadbed9_tensor * tensor, const char * name);
    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_format_name(      struct ggml_dadbed9_tensor * tensor, const char * fmt, ...);

    //
    // operations on tensors with backpropagation
    //

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_dup(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a);

    // in-place, returns view(a)
    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_dup_inplace(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_add(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            struct ggml_dadbed9_tensor  * b);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_add_inplace(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            struct ggml_dadbed9_tensor  * b);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_add1(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            struct ggml_dadbed9_tensor  * b);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_add1_inplace(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            struct ggml_dadbed9_tensor  * b);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_acc(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            struct ggml_dadbed9_tensor  * b,
            size_t                nb1,
            size_t                nb2,
            size_t                nb3,
            size_t                offset);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_acc_inplace(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            struct ggml_dadbed9_tensor  * b,
            size_t                nb1,
            size_t                nb2,
            size_t                nb3,
            size_t                offset);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_sub(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            struct ggml_dadbed9_tensor  * b);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_sub_inplace(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            struct ggml_dadbed9_tensor  * b);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_mul(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            struct ggml_dadbed9_tensor  * b);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_mul_inplace(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            struct ggml_dadbed9_tensor  * b);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_div(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            struct ggml_dadbed9_tensor  * b);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_div_inplace(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            struct ggml_dadbed9_tensor  * b);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_sqr(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_sqr_inplace(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_sqrt(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_sqrt_inplace(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_log(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_log_inplace(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a);

    // return scalar
    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_sum(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a);

    // sums along rows, with input shape [a,b,c,d] return shape [1,b,c,d]
    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_sum_rows(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a);

    // mean along rows
    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_mean(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a);

    // argmax along rows
    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_argmax(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a);

    // if a is the same shape as b, and a is not parameter, return a
    // otherwise, return a new tensor: repeat(a) to fit in b
    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_repeat(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            struct ggml_dadbed9_tensor  * b);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_repeat_back(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            struct ggml_dadbed9_tensor  * b);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_abs(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_abs_inplace(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_sgn(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_sgn_inplace(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_neg(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_neg_inplace(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_step(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_step_inplace(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_tanh(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_tanh_inplace(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_elu(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_elu_inplace(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_relu(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_relu_inplace(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a);

    // TODO: double-check this computation is correct
    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_gelu(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_gelu_inplace(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_gelu_quick(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_gelu_quick_inplace(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_silu(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_silu_inplace(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a);

    // a - x
    // b - dy
    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_silu_back(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            struct ggml_dadbed9_tensor  * b);

    // normalize along rows
    // TODO: eps is hardcoded to 1e-5 for now
    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_norm(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_norm_inplace(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_rms_norm(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            float                 eps);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_rms_norm_inplace(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            float                 eps);

    // a - x
    // b - dy
    // TODO: update with configurable eps
    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_rms_norm_back(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            struct ggml_dadbed9_tensor  * b);

    // A: n columns, m rows
    // B: n columns, p rows  (i.e. we transpose it internally)
    // result is m columns, p rows
    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_mul_mat(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            struct ggml_dadbed9_tensor  * b);

    // A: m columns, n rows,
    // B: p columns, n rows,
    // result is m columns, p rows
    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_out_prod(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            struct ggml_dadbed9_tensor  * b);

    //
    // operations on tensors without backpropagation
    //

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_scale(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            struct ggml_dadbed9_tensor  * b);

    // in-place, returns view(a)
    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_scale_inplace(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            struct ggml_dadbed9_tensor  * b);

    // b -> view(a,offset,nb1,nb2,3), return modified a
    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_set(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            struct ggml_dadbed9_tensor  * b,
            size_t                nb1,
            size_t                nb2,
            size_t                nb3,
            size_t                offset);

    // b -> view(a,offset,nb1,nb2,3), return view(a)
    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_set_inplace(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            struct ggml_dadbed9_tensor  * b,
            size_t                nb1,
            size_t                nb2,
            size_t                nb3,
            size_t                offset);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_set_1d(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            struct ggml_dadbed9_tensor  * b,
            size_t                offset);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_set_1d_inplace(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            struct ggml_dadbed9_tensor  * b,
            size_t                offset);

    // b -> view(a,offset,nb1,nb2,3), return modified a
    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_set_2d(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            struct ggml_dadbed9_tensor  * b,
            size_t                nb1,
            size_t                offset);

    // b -> view(a,offset,nb1,nb2,3), return view(a)
    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_set_2d_inplace(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            struct ggml_dadbed9_tensor  * b,
            size_t                nb1,
            size_t                offset);


    // a -> b, return view(b)
    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_cpy(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            struct ggml_dadbed9_tensor  * b);

    // a -> b, in-place, return view(b)
    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_cpy_inplace(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            struct ggml_dadbed9_tensor  * b);

    // make contiguous
    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_cont(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a);

    // make contiguous, in-place
    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_cont_inplace(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a);

    // return view(a), b specifies the new shape
    // TODO: when we start computing gradient, make a copy instead of view
    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_reshape(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            struct ggml_dadbed9_tensor  * b);

    // return view(a)
    // TODO: when we start computing gradient, make a copy instead of view
    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_reshape_1d(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            int64_t               ne0);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_reshape_2d(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            int64_t               ne0,
            int64_t               ne1);

    // return view(a)
    // TODO: when we start computing gradient, make a copy instead of view
    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_reshape_3d(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_reshape_4d(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2,
            int64_t               ne3);

    // offset in bytes
    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_view_1d(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            int64_t               ne0,
            size_t                offset);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_view_2d(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            size_t                nb1, // row stride in bytes
            size_t                offset);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_view_3d(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2,
            size_t                nb1, // row   stride in bytes
            size_t                nb2, // slice stride in bytes
            size_t                offset);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_view_4d(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2,
            int64_t               ne3,
            size_t                nb1, // row   stride in bytes
            size_t                nb2, // slice stride in bytes
            size_t                nb3,
            size_t                offset);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_permute(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            int                   axis0,
            int                   axis1,
            int                   axis2,
            int                   axis3);

    // alias for ggml_dadbed9_permute(ctx, a, 1, 0, 2, 3)
    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_transpose(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_get_rows(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            struct ggml_dadbed9_tensor  * b);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_get_rows_back(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            struct ggml_dadbed9_tensor  * b,
            struct ggml_dadbed9_tensor  * c);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_diag(
        struct ggml_dadbed9_context     * ctx,
        struct ggml_dadbed9_tensor      * a);

    // set elements above the diagonal to -INF
    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_diag_mask_inf(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            int                   n_past);

    // in-place, returns view(a)
    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_diag_mask_inf_inplace(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            int                   n_past);

    // set elements above the diagonal to 0
    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_diag_mask_zero(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            int                   n_past);

    // in-place, returns view(a)
    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_diag_mask_zero_inplace(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            int                   n_past);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_soft_max(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a);

    // in-place, returns view(a)
    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_soft_max_inplace(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_soft_max_back(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            struct ggml_dadbed9_tensor  * b);

    // in-place, returns view(a)
    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_soft_max_back_inplace(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            struct ggml_dadbed9_tensor  * b);

    // rotary position embedding
    // if mode & 1 == 1, skip n_past elements
    // if mode & 2 == 1, GPT-NeoX style
    // if mode & 4 == 1, ChatGLM style
    // TODO: avoid creating a new tensor every time
    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_rope(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            int                   n_past,
            int                   n_dims,
            int                   mode,
            int                   n_ctx);

    // in-place, returns view(a)
    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_rope_inplace(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            int                   n_past,
            int                   n_dims,
            int                   mode,
            int                   n_ctx);

    // custom RoPE
    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_rope_custom(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            int                   n_past,
            int                   n_dims,
            int                   mode,
            int                   n_ctx,
            float                 freq_base,
            float                 freq_scale);

    // in-place, returns view(a)
    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_rope_custom_inplace(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            int                   n_past,
            int                   n_dims,
            int                   mode,
            int                   n_ctx,
            float                 freq_base,
            float                 freq_scale);

    // rotary position embedding backward, i.e compute dx from dy
    // a - dy
    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_rope_back(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            int                   n_past,
            int                   n_dims,
            int                   mode,
            int                   n_ctx);

    // alibi position embedding
    // in-place, returns view(a)
    struct ggml_dadbed9_tensor * ggml_dadbed9_alibi(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            int                   n_past,
            int                   n_head,
            float                 bias_max);

    // clamp
    // in-place, returns view(a)
    struct ggml_dadbed9_tensor * ggml_dadbed9_clamp(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            float                 min,
            float                 max);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_conv_1d(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            struct ggml_dadbed9_tensor  * b,
            int                   s0,  // stride
            int                   p0,  // padding
            int                   d0); // dilation

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_conv_2d(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            struct ggml_dadbed9_tensor  * b,
            int                   s0,
            int                   s1,
            int                   p0,
            int                   p1,
            int                   d0,
            int                   d1);

    // conv_1d with padding = half
    // alias for ggml_dadbed9_conv_1d(a, b, s, a->ne[0]/2, d)
    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_conv_1d_ph(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            struct ggml_dadbed9_tensor  * b,
            int                   s,
            int                   d);

    enum ggml_dadbed9_op_pool {
        GGML_dadbed9_OP_POOL_MAX,
        GGML_dadbed9_OP_POOL_AVG,
        GGML_dadbed9_OP_POOL_COUNT,
    };

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_pool_1d(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            enum ggml_dadbed9_op_pool     op,
            int                   k0, // kernel size
            int                   s0, // stride
            int                   p0); // padding

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_pool_2d(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            enum ggml_dadbed9_op_pool     op,
            int                   k0,
            int                   k1,
            int                   s0,
            int                   s1,
            int                   p0,
            int                   p1);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_flash_attn(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * q,
            struct ggml_dadbed9_tensor  * k,
            struct ggml_dadbed9_tensor  * v,
            bool                  masked);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_flash_attn_back(
           struct ggml_dadbed9_context * ctx,
           struct ggml_dadbed9_tensor  * q,
           struct ggml_dadbed9_tensor  * k,
           struct ggml_dadbed9_tensor  * v,
           struct ggml_dadbed9_tensor  * d,
           bool                  masked);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_flash_ff(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            struct ggml_dadbed9_tensor  * b0,
            struct ggml_dadbed9_tensor  * b1,
            struct ggml_dadbed9_tensor  * c0,
            struct ggml_dadbed9_tensor  * c1);

    // partition into non-overlapping windows with padding if needed
    // example:
    // a:   768   64   64    1
    // w:    14
    // res: 768   14   14    25
    // used in sam
    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_win_part(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            int                   w);

    // reverse of ggml_dadbed9_win_part
    // used in sam
    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_win_unpart(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * a,
            int                   w0,
            int                   h0,
            int                   w);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_unary(
            struct ggml_dadbed9_context * ctx,
             struct ggml_dadbed9_tensor * a,
             enum ggml_dadbed9_unary_op op);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_unary_inplace(
        struct ggml_dadbed9_context * ctx,
        struct ggml_dadbed9_tensor  * a,
        enum ggml_dadbed9_unary_op op);

    // custom operators

    typedef void (*ggml_dadbed9_unary_op_f32_t) (const int, float *, const float *);
    typedef void (*ggml_dadbed9_binary_op_f32_t)(const int, float *, const float *, const float *);

    typedef void (*ggml_dadbed9_custom1_op_f32_t)(struct ggml_dadbed9_tensor *, const struct ggml_dadbed9_tensor *);
    typedef void (*ggml_dadbed9_custom2_op_f32_t)(struct ggml_dadbed9_tensor *, const struct ggml_dadbed9_tensor *, const struct ggml_dadbed9_tensor *);
    typedef void (*ggml_dadbed9_custom3_op_f32_t)(struct ggml_dadbed9_tensor *, const struct ggml_dadbed9_tensor *, const struct ggml_dadbed9_tensor *, const struct ggml_dadbed9_tensor *);

    GGML_dadbed9_DEPRECATED(GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_map_unary_f32(
            struct ggml_dadbed9_context        * ctx,
            struct ggml_dadbed9_tensor         * a,
                   ggml_dadbed9_unary_op_f32_t   fun),
        "use ggml_dadbed9_map_custom1 instead");

    GGML_dadbed9_DEPRECATED(GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_map_unary_inplace_f32(
            struct ggml_dadbed9_context        * ctx,
            struct ggml_dadbed9_tensor         * a,
                   ggml_dadbed9_unary_op_f32_t   fun),
        "use ggml_dadbed9_map_custom1_inplace instead");

    GGML_dadbed9_DEPRECATED(GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_map_binary_f32(
            struct ggml_dadbed9_context         * ctx,
            struct ggml_dadbed9_tensor          * a,
            struct ggml_dadbed9_tensor          * b,
                   ggml_dadbed9_binary_op_f32_t   fun),
        "use ggml_dadbed9_map_custom2 instead");

    GGML_dadbed9_DEPRECATED(GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_map_binary_inplace_f32(
            struct ggml_dadbed9_context         * ctx,
            struct ggml_dadbed9_tensor          * a,
            struct ggml_dadbed9_tensor          * b,
                   ggml_dadbed9_binary_op_f32_t   fun),
        "use ggml_dadbed9_map_custom2_inplace instead");

    GGML_dadbed9_DEPRECATED(GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_map_custom1_f32(
            struct ggml_dadbed9_context          * ctx,
            struct ggml_dadbed9_tensor           * a,
                   ggml_dadbed9_custom1_op_f32_t   fun),
        "use ggml_dadbed9_map_custom1 instead");

    GGML_dadbed9_DEPRECATED(GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_map_custom1_inplace_f32(
            struct ggml_dadbed9_context          * ctx,
            struct ggml_dadbed9_tensor           * a,
                   ggml_dadbed9_custom1_op_f32_t   fun),
        "use ggml_dadbed9_map_custom1_inplace instead");

    GGML_dadbed9_DEPRECATED(GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_map_custom2_f32(
            struct ggml_dadbed9_context          * ctx,
            struct ggml_dadbed9_tensor           * a,
            struct ggml_dadbed9_tensor           * b,
                   ggml_dadbed9_custom2_op_f32_t   fun),
        "use ggml_dadbed9_map_custom2 instead");

    GGML_dadbed9_DEPRECATED(GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_map_custom2_inplace_f32(
            struct ggml_dadbed9_context          * ctx,
            struct ggml_dadbed9_tensor           * a,
            struct ggml_dadbed9_tensor           * b,
                   ggml_dadbed9_custom2_op_f32_t   fun),
        "use ggml_dadbed9_map_custom2_inplace instead");

    GGML_dadbed9_DEPRECATED(GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_map_custom3_f32(
            struct ggml_dadbed9_context          * ctx,
            struct ggml_dadbed9_tensor           * a,
            struct ggml_dadbed9_tensor           * b,
            struct ggml_dadbed9_tensor           * c,
                   ggml_dadbed9_custom3_op_f32_t   fun),
        "use ggml_dadbed9_map_custom3 instead");

    GGML_dadbed9_DEPRECATED(GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_map_custom3_inplace_f32(
            struct ggml_dadbed9_context          * ctx,
            struct ggml_dadbed9_tensor           * a,
            struct ggml_dadbed9_tensor           * b,
            struct ggml_dadbed9_tensor           * c,
                   ggml_dadbed9_custom3_op_f32_t   fun),
        "use ggml_dadbed9_map_custom3_inplace instead");

    // custom operators v2

    typedef void (*ggml_dadbed9_custom1_op_t)(struct ggml_dadbed9_tensor * dst , const struct ggml_dadbed9_tensor * a, int ith, int nth, void * userdata);
    typedef void (*ggml_dadbed9_custom2_op_t)(struct ggml_dadbed9_tensor * dst , const struct ggml_dadbed9_tensor * a, const struct ggml_dadbed9_tensor * b, int ith, int nth, void * userdata);
    typedef void (*ggml_dadbed9_custom3_op_t)(struct ggml_dadbed9_tensor * dst , const struct ggml_dadbed9_tensor * a, const struct ggml_dadbed9_tensor * b, const struct ggml_dadbed9_tensor * c, int ith, int nth, void * userdata);

    #define GGML_dadbed9_N_TASKS_MAX -1

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_map_custom1(
            struct ggml_dadbed9_context   * ctx,
            struct ggml_dadbed9_tensor    * a,
            ggml_dadbed9_custom1_op_t       fun,
            int                     n_tasks,
            void                  * userdata);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_map_custom1_inplace(
            struct ggml_dadbed9_context   * ctx,
            struct ggml_dadbed9_tensor    * a,
            ggml_dadbed9_custom1_op_t       fun,
            int                     n_tasks,
            void                  * userdata);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_map_custom2(
            struct ggml_dadbed9_context   * ctx,
            struct ggml_dadbed9_tensor    * a,
            struct ggml_dadbed9_tensor    * b,
            ggml_dadbed9_custom2_op_t       fun,
            int                     n_tasks,
            void                  * userdata);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_map_custom2_inplace(
            struct ggml_dadbed9_context   * ctx,
            struct ggml_dadbed9_tensor    * a,
            struct ggml_dadbed9_tensor    * b,
            ggml_dadbed9_custom2_op_t       fun,
            int                     n_tasks,
            void                  * userdata);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_map_custom3(
            struct ggml_dadbed9_context   * ctx,
            struct ggml_dadbed9_tensor    * a,
            struct ggml_dadbed9_tensor    * b,
            struct ggml_dadbed9_tensor    * c,
            ggml_dadbed9_custom3_op_t       fun,
            int                     n_tasks,
            void                  * userdata);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_map_custom3_inplace(
            struct ggml_dadbed9_context   * ctx,
            struct ggml_dadbed9_tensor    * a,
            struct ggml_dadbed9_tensor    * b,
            struct ggml_dadbed9_tensor    * c,
            ggml_dadbed9_custom3_op_t       fun,
            int                     n_tasks,
            void                  * userdata);

    // loss function

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_cross_entropy_loss(
            struct ggml_dadbed9_context         * ctx,
            struct ggml_dadbed9_tensor          * a,
            struct ggml_dadbed9_tensor          * b);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_cross_entropy_loss_back(
            struct ggml_dadbed9_context         * ctx,
            struct ggml_dadbed9_tensor          * a,
            struct ggml_dadbed9_tensor          * b,
            struct ggml_dadbed9_tensor          * c);

    //
    // automatic differentiation
    //

    GGML_dadbed9_API void ggml_dadbed9_set_param(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_tensor  * tensor);


    GGML_dadbed9_API void ggml_dadbed9_build_forward_expand(struct ggml_dadbed9_cgraph * cgraph, struct ggml_dadbed9_tensor * tensor);

    GGML_dadbed9_API struct ggml_dadbed9_cgraph ggml_dadbed9_build_forward (struct ggml_dadbed9_tensor * tensor);
    GGML_dadbed9_API struct ggml_dadbed9_cgraph ggml_dadbed9_build_backward(struct ggml_dadbed9_context * ctx, struct ggml_dadbed9_cgraph * gf, bool keep);

    // graph allocation in a context
    GGML_dadbed9_API struct ggml_dadbed9_cgraph * ggml_dadbed9_new_graph        (struct ggml_dadbed9_context * ctx);
    GGML_dadbed9_API struct ggml_dadbed9_cgraph * ggml_dadbed9_build_forward_ctx(struct ggml_dadbed9_context * ctx, struct ggml_dadbed9_tensor * tensor);
    GGML_dadbed9_API size_t ggml_dadbed9_graph_overhead(void);

    // ggml_dadbed9_graph_plan() has to be called before ggml_dadbed9_graph_compute()
    // when plan.work_size > 0, caller must allocate memory for plan.work_data
    GGML_dadbed9_API struct ggml_dadbed9_cplan ggml_dadbed9_graph_plan   (struct ggml_dadbed9_cgraph * cgraph, int n_threads /*= GGML_dadbed9_DEFAULT_N_THREADS*/);
    GGML_dadbed9_API               int ggml_dadbed9_graph_compute(struct ggml_dadbed9_cgraph * cgraph, struct ggml_dadbed9_cplan * cplan);
    GGML_dadbed9_API              void ggml_dadbed9_graph_reset  (struct ggml_dadbed9_cgraph * cgraph);

    // same as ggml_dadbed9_graph_compute() but the work data is allocated as a part of the context
    // note: the drawback of this API is that you must have ensured that the context has enough memory for the work data
    GGML_dadbed9_API void ggml_dadbed9_graph_compute_with_ctx(struct ggml_dadbed9_context * ctx, struct ggml_dadbed9_cgraph * cgraph, int n_threads);

    GGML_dadbed9_API struct ggml_dadbed9_tensor * ggml_dadbed9_graph_get_tensor(struct ggml_dadbed9_cgraph * cgraph, const char * name);

    GGML_dadbed9_API void               ggml_dadbed9_graph_export(const struct ggml_dadbed9_cgraph * cgraph, const char * fname);
    GGML_dadbed9_API struct ggml_dadbed9_cgraph ggml_dadbed9_graph_import(const char * fname, struct ggml_dadbed9_context ** ctx_data, struct ggml_dadbed9_context ** ctx_eval);

    // print info and performance information for the graph
    GGML_dadbed9_API void ggml_dadbed9_graph_print(const struct ggml_dadbed9_cgraph * cgraph);

    // dump the graph into a file using the dot format
    GGML_dadbed9_API void ggml_dadbed9_graph_dump_dot(const struct ggml_dadbed9_cgraph * gb, const struct ggml_dadbed9_cgraph * gf, const char * filename);

    //
    // optimization
    //

    // optimization methods
    enum ggml_dadbed9_opt_type {
        GGML_dadbed9_OPT_ADAM,
        GGML_dadbed9_OPT_LBFGS,
    };

    // linesearch methods
    enum ggml_dadbed9_linesearch {
        GGML_dadbed9_LINESEARCH_DEFAULT = 1,

        GGML_dadbed9_LINESEARCH_BACKTRACKING_ARMIJO       = 0,
        GGML_dadbed9_LINESEARCH_BACKTRACKING_WOLFE        = 1,
        GGML_dadbed9_LINESEARCH_BACKTRACKING_STRONG_WOLFE = 2,
    };

    // optimization return values
    enum ggml_dadbed9_opt_result {
        GGML_dadbed9_OPT_OK = 0,
        GGML_dadbed9_OPT_DID_NOT_CONVERGE,
        GGML_dadbed9_OPT_NO_CONTEXT,
        GGML_dadbed9_OPT_INVALID_WOLFE,
        GGML_dadbed9_OPT_FAIL,

        GGML_dadbed9_LINESEARCH_FAIL = -128,
        GGML_dadbed9_LINESEARCH_MINIMUM_STEP,
        GGML_dadbed9_LINESEARCH_MAXIMUM_STEP,
        GGML_dadbed9_LINESEARCH_MAXIMUM_ITERATIONS,
        GGML_dadbed9_LINESEARCH_INVALID_PARAMETERS,
    };

    // optimization parameters
    //
    //   see ggml.c (ggml_dadbed9_opt_default_params) for default values
    //
    struct ggml_dadbed9_opt_params {
        enum ggml_dadbed9_opt_type type;

        int n_threads;

        // delta-based convergence test
        //
        //   if past == 0 - disabled
        //   if past > 0:
        //     stop if |f(x) - f(x_past)| < delta * max(1, |f(x)|)
        //
        int past;
        float delta;

        // maximum number of iterations without improvement
        //
        //   if 0 - disabled
        //   if > 0:
        //     assume convergence if no cost improvement in this number of iterations
        //
        int max_no_improvement;

        bool print_forward_graph;
        bool print_backward_graph;

        // ADAM parameters
        struct {
            int n_iter;

            float sched; // schedule multiplier (fixed, decay or warmup)
            float decay; // weight decay for AdamW, use 0.0f to disable
            float alpha; // learning rate
            float beta1;
            float beta2;
            float eps;   // epsilon for numerical stability
            float eps_f; // epsilon for convergence test
            float eps_g; // epsilon for convergence test
        } adam;

        // LBFGS parameters
        struct {
            int m; // number of corrections to approximate the inv. Hessian
            int n_iter;
            int max_linesearch;

            float eps;      // convergence tolerance
            float ftol;     // line search tolerance
            float wolfe;
            float min_step;
            float max_step;

            enum ggml_dadbed9_linesearch linesearch;
        } lbfgs;
    };

    struct ggml_dadbed9_opt_context {
        struct ggml_dadbed9_context * ctx;
        struct ggml_dadbed9_opt_params params;

        int iter;
        int64_t nx; // number of parameter elements

        bool just_initialized;

        struct {
            struct ggml_dadbed9_tensor * x;  // view of the parameters
            struct ggml_dadbed9_tensor * g1; // gradient
            struct ggml_dadbed9_tensor * g2; // gradient squared
            struct ggml_dadbed9_tensor * m;  // first moment
            struct ggml_dadbed9_tensor * v;  // second moment
            struct ggml_dadbed9_tensor * mh; // first moment hat
            struct ggml_dadbed9_tensor * vh; // second moment hat
            struct ggml_dadbed9_tensor * pf; // past function values
            float fx_best;
            float fx_prev;
            int n_no_improvement;
        } adam;

        struct {
            struct ggml_dadbed9_tensor * x;    // current parameters
            struct ggml_dadbed9_tensor * xp;   // previous parameters
            struct ggml_dadbed9_tensor * g;    // current gradient
            struct ggml_dadbed9_tensor * gp;   // previous gradient
            struct ggml_dadbed9_tensor * d;    // search direction
            struct ggml_dadbed9_tensor * pf;   // past function values
            struct ggml_dadbed9_tensor * lmal; // the L-BFGS memory alpha
            struct ggml_dadbed9_tensor * lmys; // the L-BFGS memory ys
            struct ggml_dadbed9_tensor * lms;  // the L-BFGS memory s
            struct ggml_dadbed9_tensor * lmy;  // the L-BFGS memory y
            float fx_best;
            float step;
            int j;
            int k;
            int end;
            int n_no_improvement;
        } lbfgs;
    };

    GGML_dadbed9_API struct ggml_dadbed9_opt_params ggml_dadbed9_opt_default_params(enum ggml_dadbed9_opt_type type);

    // optimize the function defined by the tensor f
    GGML_dadbed9_API enum ggml_dadbed9_opt_result ggml_dadbed9_opt(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_opt_params params,
            struct ggml_dadbed9_tensor * f);

    // initialize optimizer context
    GGML_dadbed9_API void ggml_dadbed9_opt_init(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_opt_context * opt,
            struct ggml_dadbed9_opt_params params,
            int64_t nx);

    // continue optimizing the function defined by the tensor f
    GGML_dadbed9_API enum ggml_dadbed9_opt_result ggml_dadbed9_opt_resume(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_opt_context * opt,
            struct ggml_dadbed9_tensor * f);

    // continue optimizing the function defined by the tensor f
    GGML_dadbed9_API enum ggml_dadbed9_opt_result ggml_dadbed9_opt_resume_g(
            struct ggml_dadbed9_context * ctx,
            struct ggml_dadbed9_opt_context * opt,
            struct ggml_dadbed9_tensor * f,
            struct ggml_dadbed9_cgraph * gf,
            struct ggml_dadbed9_cgraph * gb);

    //
    // quantization
    //

    GGML_dadbed9_API size_t ggml_dadbed9_quantize_q4_0(const float * src, void * dst, int n, int k, int64_t * hist);
    GGML_dadbed9_API size_t ggml_dadbed9_quantize_q4_1(const float * src, void * dst, int n, int k, int64_t * hist);
    GGML_dadbed9_API size_t ggml_dadbed9_quantize_q5_0(const float * src, void * dst, int n, int k, int64_t * hist);
    GGML_dadbed9_API size_t ggml_dadbed9_quantize_q5_1(const float * src, void * dst, int n, int k, int64_t * hist);
    GGML_dadbed9_API size_t ggml_dadbed9_quantize_q8_0(const float * src, void * dst, int n, int k, int64_t * hist);

    GGML_dadbed9_API size_t ggml_dadbed9_quantize_chunk(enum ggml_dadbed9_type type, const float * src, void * dst, int start, int n, int64_t * hist);

    //
    // system info
    //

    GGML_dadbed9_API int ggml_dadbed9_cpu_has_avx        (void);
    GGML_dadbed9_API int ggml_dadbed9_cpu_has_avx2       (void);
    GGML_dadbed9_API int ggml_dadbed9_cpu_has_avx512     (void);
    GGML_dadbed9_API int ggml_dadbed9_cpu_has_avx512_vbmi(void);
    GGML_dadbed9_API int ggml_dadbed9_cpu_has_avx512_vnni(void);
    GGML_dadbed9_API int ggml_dadbed9_cpu_has_fma        (void);
    GGML_dadbed9_API int ggml_dadbed9_cpu_has_neon       (void);
    GGML_dadbed9_API int ggml_dadbed9_cpu_has_arm_fma    (void);
    GGML_dadbed9_API int ggml_dadbed9_cpu_has_f16c       (void);
    GGML_dadbed9_API int ggml_dadbed9_cpu_has_fp16_va    (void);
    GGML_dadbed9_API int ggml_dadbed9_cpu_has_wasm_simd  (void);
    GGML_dadbed9_API int ggml_dadbed9_cpu_has_blas       (void);
    GGML_dadbed9_API int ggml_dadbed9_cpu_has_cublas     (void);
    GGML_dadbed9_API int ggml_dadbed9_cpu_has_clblast    (void);
    GGML_dadbed9_API int ggml_dadbed9_cpu_has_gpublas    (void);
    GGML_dadbed9_API int ggml_dadbed9_cpu_has_sse3       (void);
    GGML_dadbed9_API int ggml_dadbed9_cpu_has_vsx        (void);

    //
    // Internal types and functions exposed for tests and benchmarks
    //

#ifdef  __cplusplus
// restrict not standard in C++
#define GGML_dadbed9_RESTRICT
#else
#define GGML_dadbed9_RESTRICT restrict
#endif
    typedef void (*ggml_dadbed9_to_float_t)  (const void  * GGML_dadbed9_RESTRICT x, float * GGML_dadbed9_RESTRICT y, int k);
    typedef void (*ggml_dadbed9_from_float_t)(const float * GGML_dadbed9_RESTRICT x, void  * GGML_dadbed9_RESTRICT y, int k);
    typedef void (*ggml_dadbed9_vec_dot_t)   (const int n, float * GGML_dadbed9_RESTRICT s, const void * GGML_dadbed9_RESTRICT x, const void * GGML_dadbed9_RESTRICT y);

    typedef struct {
        const char      * type_name;
        int               blck_size;
        size_t            type_size;
        bool              is_quantized;
        ggml_dadbed9_to_float_t   to_float;
        ggml_dadbed9_from_float_t from_float;
        ggml_dadbed9_from_float_t from_float_reference;
        ggml_dadbed9_vec_dot_t    vec_dot;
        enum ggml_dadbed9_type    vec_dot_type;
    } ggml_dadbed9_type_traits_t;

    ggml_dadbed9_type_traits_t ggml_dadbed9_internal_get_type_traits(enum ggml_dadbed9_type type);

#ifdef  __cplusplus
}
#endif
