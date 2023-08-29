#pragma once

//
// GGML_a1d0ea7 Tensor Library
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
//       struct ggml_a1d0ea7_init_params params = {
//           .mem_size   = 16*1024*1024,
//           .mem_buffer = NULL,
//       };
//
//       // memory allocation happens here
//       struct ggml_a1d0ea7_context * ctx = ggml_a1d0ea7_init(params);
//
//       struct ggml_a1d0ea7_tensor * x = ggml_a1d0ea7_new_tensor_1d(ctx, GGML_a1d0ea7_TYPE_F32, 1);
//
//       ggml_a1d0ea7_set_param(ctx, x); // x is an input variable
//
//       struct ggml_a1d0ea7_tensor * a  = ggml_a1d0ea7_new_tensor_1d(ctx, GGML_a1d0ea7_TYPE_F32, 1);
//       struct ggml_a1d0ea7_tensor * b  = ggml_a1d0ea7_new_tensor_1d(ctx, GGML_a1d0ea7_TYPE_F32, 1);
//       struct ggml_a1d0ea7_tensor * x2 = ggml_a1d0ea7_mul(ctx, x, x);
//       struct ggml_a1d0ea7_tensor * f  = ggml_a1d0ea7_add(ctx, ggml_a1d0ea7_mul(ctx, a, x2), b);
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
//       struct ggml_a1d0ea7_cgraph gf = ggml_a1d0ea7_build_forward(f);
//
//       // set the input variable and parameter values
//       ggml_a1d0ea7_set_f32(x, 2.0f);
//       ggml_a1d0ea7_set_f32(a, 3.0f);
//       ggml_a1d0ea7_set_f32(b, 4.0f);
//
//       ggml_a1d0ea7_graph_compute(ctx0, &gf);
//
//       printf("f = %f\n", ggml_a1d0ea7_get_f32_1d(f, 0));
//
//       ...
//   }
//
// The actual computation is performed in the ggml_a1d0ea7_graph_compute() function.
//
// The ggml_a1d0ea7_new_tensor_...() functions create new tensors. They are allocated in the memory buffer provided to the
// ggml_a1d0ea7_init() function. You have to be careful not to exceed the memory buffer size. Therefore, you have to know
// in advance how much memory you need for your computation. Alternatively, you can allocate a large enough memory
// and after defining the computation graph, call the ggml_a1d0ea7_used_mem() function to find out how much memory was
// actually needed.
//
// The ggml_a1d0ea7_set_param() function marks a tensor as an input variable. This is used by the automatic
// differentiation and optimization algorithms.
//
// The described approach allows to define the function graph once and then compute its forward or backward graphs
// multiple times. All computations will use the same memory buffer allocated in the ggml_a1d0ea7_init() function. This way
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
//   - ggml_a1d0ea7_permute()
//   - ggml_a1d0ea7_conv_1d_1s()
//   - ggml_a1d0ea7_conv_1d_2s()
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
// ## Tensor data (struct ggml_a1d0ea7_tensor)
//
// The tensors are stored in memory via the ggml_a1d0ea7_tensor struct. The structure provides information about the size of
// the tensor, the data type, and the memory buffer where the tensor data is stored. Additionally, it contains
// pointers to the "source" tensors - i.e. the tensors that were used to compute the current tensor. For example:
//
//   {
//       struct ggml_a1d0ea7_tensor * c = ggml_a1d0ea7_add(ctx, a, b);
//
//       assert(c->src[0] == a);
//       assert(c->src[1] == b);
//   }
//
// The multi-dimensional tensors are stored in row-major order. The ggml_a1d0ea7_tensor struct contains fields for the
// number of elements in each dimension ("ne") as well as the number of bytes ("nb", a.k.a. stride). This allows
// to store tensors that are not contiguous in memory, which is useful for operations such as transposition and
// permutation. All tensor operations have to take the stride into account and not assume that the tensor is
// contiguous in memory.
//
// The data of the tensor is accessed via the "data" pointer. For example:
//
//   {
//       struct ggml_a1d0ea7_tensor * a = ggml_a1d0ea7_new_tensor_2d(ctx, GGML_a1d0ea7_TYPE_F32, 2, 3);
//
//       // a[1, 2] = 1.0f;
//       *(float *) ((char *) a->data + 2*a->nb[1] + 1*a->nb[0]) = 1.0f;
//
//       // a[2, 0] = 2.0f;
//       *(float *) ((char *) a->data + 0*a->nb[1] + 2*a->nb[0]) = 2.0f;
//
//       ...
//   }
//
// Alternatively, there are helper functions, such as ggml_a1d0ea7_get_f32_1d() and ggml_a1d0ea7_set_f32_1d() that can be used.
//
// ## The matrix multiplication operator (ggml_a1d0ea7_mul_mat)
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

#ifdef GGML_a1d0ea7_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef GGML_a1d0ea7_BUILD
#            define GGML_a1d0ea7_API __declspec(dllexport)
#        else
#            define GGML_a1d0ea7_API __declspec(dllimport)
#        endif
#    else
#        define GGML_a1d0ea7_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define GGML_a1d0ea7_API
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define GGML_a1d0ea7_FILE_MAGIC   0x67676d6c // "ggml"
#define GGML_a1d0ea7_FILE_VERSION 1

#define GGML_a1d0ea7_QNT_VERSION        2    // bump this on quantization format changes
#define GGML_a1d0ea7_QNT_VERSION_FACTOR 1000 // do not change this

#define GGML_a1d0ea7_MAX_DIMS          4
#define GGML_a1d0ea7_MAX_NODES         4096
#define GGML_a1d0ea7_MAX_PARAMS        256
#define GGML_a1d0ea7_MAX_CONTEXTS      64
#define GGML_a1d0ea7_MAX_OPT           4
#define GGML_a1d0ea7_MAX_NAME          32
#define GGML_a1d0ea7_DEFAULT_N_THREADS 4

#define GGML_a1d0ea7_ASSERT(x) \
    do { \
        if (!(x)) { \
            fprintf(stderr, "GGML_a1d0ea7_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
            abort(); \
        } \
    } while (0)

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef __ARM_NEON
    // we use the built-in 16-bit float type
    typedef __fp16 ggml_a1d0ea7_fp16_t;
#else
    typedef uint16_t ggml_a1d0ea7_fp16_t;
#endif

    // convert FP16 <-> FP32
    GGML_a1d0ea7_API float       ggml_a1d0ea7_fp16_to_fp32(ggml_a1d0ea7_fp16_t x);
    GGML_a1d0ea7_API ggml_a1d0ea7_fp16_t ggml_a1d0ea7_fp32_to_fp16(float x);

    GGML_a1d0ea7_API void ggml_a1d0ea7_fp16_to_fp32_row(const ggml_a1d0ea7_fp16_t * x, float * y, size_t n);
    GGML_a1d0ea7_API void ggml_a1d0ea7_fp32_to_fp16_row(const float * x, ggml_a1d0ea7_fp16_t * y, size_t n);

    struct ggml_a1d0ea7_object;
    struct ggml_a1d0ea7_context;

    enum ggml_a1d0ea7_type {
        GGML_a1d0ea7_TYPE_F32  = 0,
        GGML_a1d0ea7_TYPE_F16  = 1,
        GGML_a1d0ea7_TYPE_Q4_0 = 2,
        GGML_a1d0ea7_TYPE_Q4_1 = 3,
        // GGML_a1d0ea7_TYPE_Q4_2 = 4, support has been removed
        // GGML_a1d0ea7_TYPE_Q4_3 (5) support has been removed
        GGML_a1d0ea7_TYPE_Q5_0 = 6,
        GGML_a1d0ea7_TYPE_Q5_1 = 7,
        GGML_a1d0ea7_TYPE_Q8_0 = 8,
        GGML_a1d0ea7_TYPE_Q8_1 = 9,
        // k-quantizations
        GGML_a1d0ea7_TYPE_Q2_K = 10,
        GGML_a1d0ea7_TYPE_Q3_K = 11,
        GGML_a1d0ea7_TYPE_Q4_K = 12,
        GGML_a1d0ea7_TYPE_Q5_K = 13,
        GGML_a1d0ea7_TYPE_Q6_K = 14,
        GGML_a1d0ea7_TYPE_Q8_K = 15,
        GGML_a1d0ea7_TYPE_I8,
        GGML_a1d0ea7_TYPE_I16,
        GGML_a1d0ea7_TYPE_I32,
        GGML_a1d0ea7_TYPE_COUNT,
    };

    enum ggml_a1d0ea7_backend {
        GGML_a1d0ea7_BACKEND_CPU = 0,
        GGML_a1d0ea7_BACKEND_GPU = 10,
        GGML_a1d0ea7_BACKEND_GPU_SPLIT = 20,
    };

    // model file types
    enum ggml_a1d0ea7_ftype {
        GGML_a1d0ea7_FTYPE_UNKNOWN     = -1,
        GGML_a1d0ea7_FTYPE_ALL_F32     = 0,
        GGML_a1d0ea7_FTYPE_MOSTLY_F16  = 1,  // except 1d tensors
        GGML_a1d0ea7_FTYPE_MOSTLY_Q4_0 = 2,  // except 1d tensors
        GGML_a1d0ea7_FTYPE_MOSTLY_Q4_1 = 3,  // except 1d tensors
        GGML_a1d0ea7_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4, // tok_embeddings.weight and output.weight are F16
        GGML_a1d0ea7_FTYPE_MOSTLY_Q8_0 = 7,  // except 1d tensors
        GGML_a1d0ea7_FTYPE_MOSTLY_Q5_0 = 8,  // except 1d tensors
        GGML_a1d0ea7_FTYPE_MOSTLY_Q5_1 = 9,  // except 1d tensors
        GGML_a1d0ea7_FTYPE_MOSTLY_Q2_K = 10, // except 1d tensors
        GGML_a1d0ea7_FTYPE_MOSTLY_Q3_K = 11, // except 1d tensors
        GGML_a1d0ea7_FTYPE_MOSTLY_Q4_K = 12, // except 1d tensors
        GGML_a1d0ea7_FTYPE_MOSTLY_Q5_K = 13, // except 1d tensors
        GGML_a1d0ea7_FTYPE_MOSTLY_Q6_K = 14, // except 1d tensors
    };

    // available tensor operations:
    enum ggml_a1d0ea7_op {
        GGML_a1d0ea7_OP_NONE = 0,

        GGML_a1d0ea7_OP_DUP,
        GGML_a1d0ea7_OP_ADD,
        GGML_a1d0ea7_OP_ADD1,
        GGML_a1d0ea7_OP_ACC,
        GGML_a1d0ea7_OP_SUB,
        GGML_a1d0ea7_OP_MUL,
        GGML_a1d0ea7_OP_DIV,
        GGML_a1d0ea7_OP_SQR,
        GGML_a1d0ea7_OP_SQRT,
        GGML_a1d0ea7_OP_LOG,
        GGML_a1d0ea7_OP_SUM,
        GGML_a1d0ea7_OP_SUM_ROWS,
        GGML_a1d0ea7_OP_MEAN,
        GGML_a1d0ea7_OP_REPEAT,
        GGML_a1d0ea7_OP_REPEAT_BACK,
        GGML_a1d0ea7_OP_ABS,
        GGML_a1d0ea7_OP_SGN,
        GGML_a1d0ea7_OP_NEG,
        GGML_a1d0ea7_OP_STEP,
        GGML_a1d0ea7_OP_RELU,
        GGML_a1d0ea7_OP_GELU,
        GGML_a1d0ea7_OP_GELU_QUICK,
        GGML_a1d0ea7_OP_SILU,
        GGML_a1d0ea7_OP_SILU_BACK,
        GGML_a1d0ea7_OP_NORM, // normalize
        GGML_a1d0ea7_OP_RMS_NORM,
        GGML_a1d0ea7_OP_RMS_NORM_BACK,

        GGML_a1d0ea7_OP_MUL_MAT,
        GGML_a1d0ea7_OP_OUT_PROD,

        GGML_a1d0ea7_OP_SCALE,
        GGML_a1d0ea7_OP_SET,
        GGML_a1d0ea7_OP_CPY,
        GGML_a1d0ea7_OP_CONT,
        GGML_a1d0ea7_OP_RESHAPE,
        GGML_a1d0ea7_OP_VIEW,
        GGML_a1d0ea7_OP_PERMUTE,
        GGML_a1d0ea7_OP_TRANSPOSE,
        GGML_a1d0ea7_OP_GET_ROWS,
        GGML_a1d0ea7_OP_GET_ROWS_BACK,
        GGML_a1d0ea7_OP_DIAG,
        GGML_a1d0ea7_OP_DIAG_MASK_INF,
        GGML_a1d0ea7_OP_DIAG_MASK_ZERO,
        GGML_a1d0ea7_OP_SOFT_MAX,
        GGML_a1d0ea7_OP_SOFT_MAX_BACK,
        GGML_a1d0ea7_OP_ROPE,
        GGML_a1d0ea7_OP_ROPE_BACK,
        GGML_a1d0ea7_OP_ALIBI,
        GGML_a1d0ea7_OP_CLAMP,
        GGML_a1d0ea7_OP_CONV_1D_S1_PH,
        GGML_a1d0ea7_OP_CONV_1D_S2_PH,
        GGML_a1d0ea7_OP_CONV_2D_SK_P0,

        GGML_a1d0ea7_OP_FLASH_ATTN,
        GGML_a1d0ea7_OP_FLASH_FF,
        GGML_a1d0ea7_OP_FLASH_ATTN_BACK,
        GGML_a1d0ea7_OP_WIN_PART,
        GGML_a1d0ea7_OP_WIN_UNPART,

        GGML_a1d0ea7_OP_MAP_UNARY,
        GGML_a1d0ea7_OP_MAP_BINARY,

        GGML_a1d0ea7_OP_MAP_CUSTOM1,
        GGML_a1d0ea7_OP_MAP_CUSTOM2,
        GGML_a1d0ea7_OP_MAP_CUSTOM3,

        GGML_a1d0ea7_OP_CROSS_ENTROPY_LOSS,
        GGML_a1d0ea7_OP_CROSS_ENTROPY_LOSS_BACK,

        GGML_a1d0ea7_OP_COUNT,
    };


    // ggml object
    struct ggml_a1d0ea7_object {
        size_t offs;
        size_t size;

        struct ggml_a1d0ea7_object * next;

        char padding[8];
    };

    static const size_t GGML_a1d0ea7_OBJECT_SIZE = sizeof(struct ggml_a1d0ea7_object);

    // n-dimensional tensor
    struct ggml_a1d0ea7_tensor {
        enum ggml_a1d0ea7_type    type;
        enum ggml_a1d0ea7_backend backend;

        int     n_dims;
        int64_t ne[GGML_a1d0ea7_MAX_DIMS]; // number of elements
        size_t  nb[GGML_a1d0ea7_MAX_DIMS]; // stride in bytes:
                                   // nb[0] = sizeof(type)
                                   // nb[1] = nb[0]   * ne[0] + padding
                                   // nb[i] = nb[i-1] * ne[i-1]

        // compute data
        enum ggml_a1d0ea7_op op;

        bool is_param;

        struct ggml_a1d0ea7_tensor * grad;
        struct ggml_a1d0ea7_tensor * src0;
        struct ggml_a1d0ea7_tensor * src1;
        struct ggml_a1d0ea7_tensor * opt[GGML_a1d0ea7_MAX_OPT];

        // thread scheduling
        int n_tasks;

        // performance
        int     perf_runs;
        int64_t perf_cycles;
        int64_t perf_time_us;

        void * data;

        char name[GGML_a1d0ea7_MAX_NAME];

        void * extra; // extra things e.g. for ggml-cuda.cu

        char padding[4];
    };

    static const size_t GGML_a1d0ea7_TENSOR_SIZE = sizeof(struct ggml_a1d0ea7_tensor);

    // computation graph
    struct ggml_a1d0ea7_cgraph {
        int n_nodes;
        int n_leafs;
        int n_threads;

        size_t work_size;
        struct ggml_a1d0ea7_tensor * work;

        struct ggml_a1d0ea7_tensor * nodes[GGML_a1d0ea7_MAX_NODES];
        struct ggml_a1d0ea7_tensor * grads[GGML_a1d0ea7_MAX_NODES];
        struct ggml_a1d0ea7_tensor * leafs[GGML_a1d0ea7_MAX_NODES];

        // performance
        int     perf_runs;
        int64_t perf_cycles;
        int64_t perf_time_us;
    };

    // scratch buffer
    struct ggml_a1d0ea7_scratch {
        size_t offs;
        size_t size;
        void * data;
    };

    struct ggml_a1d0ea7_init_params {
        // memory pool
        size_t mem_size;   // bytes
        void * mem_buffer; // if NULL, memory will be allocated internally
        bool   no_alloc;   // don't allocate memory for the tensor data
    };


    // compute types
    enum ggml_a1d0ea7_task_type {
        GGML_a1d0ea7_TASK_INIT = 0,
        GGML_a1d0ea7_TASK_COMPUTE,
        GGML_a1d0ea7_TASK_FINALIZE,
    };

    struct ggml_a1d0ea7_compute_params {
        enum ggml_a1d0ea7_task_type type;

        // ith = thread index, nth = number of threads
        int ith, nth;

        // work buffer for all threads
        size_t wsize;
        void * wdata;
    };

    // misc

    GGML_a1d0ea7_API void    ggml_a1d0ea7_time_init(void); // call this once at the beginning of the program
    GGML_a1d0ea7_API int64_t ggml_a1d0ea7_time_ms(void);
    GGML_a1d0ea7_API int64_t ggml_a1d0ea7_time_us(void);
    GGML_a1d0ea7_API int64_t ggml_a1d0ea7_cycles(void);
    GGML_a1d0ea7_API int64_t ggml_a1d0ea7_cycles_per_ms(void);

    GGML_a1d0ea7_API void    ggml_a1d0ea7_print_object (const struct ggml_a1d0ea7_object * obj);
    GGML_a1d0ea7_API void    ggml_a1d0ea7_print_objects(const struct ggml_a1d0ea7_context * ctx);

    GGML_a1d0ea7_API int64_t ggml_a1d0ea7_nelements   (const struct ggml_a1d0ea7_tensor * tensor);
    GGML_a1d0ea7_API int64_t ggml_a1d0ea7_nrows       (const struct ggml_a1d0ea7_tensor * tensor);
    GGML_a1d0ea7_API size_t  ggml_a1d0ea7_nbytes      (const struct ggml_a1d0ea7_tensor * tensor);
    GGML_a1d0ea7_API size_t  ggml_a1d0ea7_nbytes_split(const struct ggml_a1d0ea7_tensor * tensor, int nrows_split);

    GGML_a1d0ea7_API int     ggml_a1d0ea7_blck_size (enum ggml_a1d0ea7_type type);
    GGML_a1d0ea7_API size_t  ggml_a1d0ea7_type_size (enum ggml_a1d0ea7_type type); // size in bytes for all elements in a block
    GGML_a1d0ea7_API float   ggml_a1d0ea7_type_sizef(enum ggml_a1d0ea7_type type); // ggml_a1d0ea7_type_size()/ggml_a1d0ea7_blck_size() as float

    GGML_a1d0ea7_API const char * ggml_a1d0ea7_type_name(enum ggml_a1d0ea7_type type);
    GGML_a1d0ea7_API const char * ggml_a1d0ea7_op_name  (enum ggml_a1d0ea7_op   op);

    GGML_a1d0ea7_API size_t  ggml_a1d0ea7_element_size(const struct ggml_a1d0ea7_tensor * tensor);

    GGML_a1d0ea7_API bool    ggml_a1d0ea7_is_quantized(enum ggml_a1d0ea7_type type);

    // TODO: temporary until model loading of ggml examples is refactored
    GGML_a1d0ea7_API enum ggml_a1d0ea7_type ggml_a1d0ea7_ftype_to_ggml_a1d0ea7_type(enum ggml_a1d0ea7_ftype ftype);

    GGML_a1d0ea7_API bool ggml_a1d0ea7_is_transposed(const struct ggml_a1d0ea7_tensor * tensor);
    GGML_a1d0ea7_API bool ggml_a1d0ea7_is_contiguous(const struct ggml_a1d0ea7_tensor * tensor);
    GGML_a1d0ea7_API bool ggml_a1d0ea7_is_permuted  (const struct ggml_a1d0ea7_tensor * tensor);

    // use this to compute the memory overhead of a tensor
    GGML_a1d0ea7_API size_t ggml_a1d0ea7_tensor_overhead(void);

    // main

    GGML_a1d0ea7_API struct ggml_a1d0ea7_context * ggml_a1d0ea7_init(struct ggml_a1d0ea7_init_params params);
    GGML_a1d0ea7_API void                  ggml_a1d0ea7_free(struct ggml_a1d0ea7_context * ctx);

    GGML_a1d0ea7_API size_t  ggml_a1d0ea7_used_mem(const struct ggml_a1d0ea7_context * ctx);

    GGML_a1d0ea7_API size_t  ggml_a1d0ea7_set_scratch (struct ggml_a1d0ea7_context * ctx, struct ggml_a1d0ea7_scratch scratch);
    GGML_a1d0ea7_API void    ggml_a1d0ea7_set_no_alloc(struct ggml_a1d0ea7_context * ctx, bool no_alloc);

    GGML_a1d0ea7_API void *  ggml_a1d0ea7_get_mem_buffer     (const struct ggml_a1d0ea7_context * ctx);
    GGML_a1d0ea7_API size_t  ggml_a1d0ea7_get_mem_size       (const struct ggml_a1d0ea7_context * ctx);
    GGML_a1d0ea7_API size_t  ggml_a1d0ea7_get_max_tensor_size(const struct ggml_a1d0ea7_context * ctx);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_new_tensor(
            struct ggml_a1d0ea7_context * ctx,
            enum   ggml_a1d0ea7_type type,
            int    n_dims,
            const int64_t *ne);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_new_tensor_1d(
            struct ggml_a1d0ea7_context * ctx,
            enum   ggml_a1d0ea7_type type,
            int64_t ne0);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_new_tensor_2d(
            struct ggml_a1d0ea7_context * ctx,
            enum   ggml_a1d0ea7_type type,
            int64_t ne0,
            int64_t ne1);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_new_tensor_3d(
            struct ggml_a1d0ea7_context * ctx,
            enum   ggml_a1d0ea7_type type,
            int64_t ne0,
            int64_t ne1,
            int64_t ne2);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_new_tensor_4d(
            struct ggml_a1d0ea7_context * ctx,
            enum   ggml_a1d0ea7_type type,
            int64_t ne0,
            int64_t ne1,
            int64_t ne2,
            int64_t ne3);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_new_i32(struct ggml_a1d0ea7_context * ctx, int32_t value);
    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_new_f32(struct ggml_a1d0ea7_context * ctx, float value);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_dup_tensor (struct ggml_a1d0ea7_context * ctx, const struct ggml_a1d0ea7_tensor * src);
    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_view_tensor(struct ggml_a1d0ea7_context * ctx, const struct ggml_a1d0ea7_tensor * src);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_get_tensor(struct ggml_a1d0ea7_context * ctx, const char * name);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_set_zero(struct ggml_a1d0ea7_tensor * tensor);
    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_set_i32 (struct ggml_a1d0ea7_tensor * tensor, int32_t value);
    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_set_f32 (struct ggml_a1d0ea7_tensor * tensor, float value);

    GGML_a1d0ea7_API int32_t ggml_a1d0ea7_get_i32_1d(const struct ggml_a1d0ea7_tensor * tensor, int i);
    GGML_a1d0ea7_API void    ggml_a1d0ea7_set_i32_1d(const struct ggml_a1d0ea7_tensor * tensor, int i, int32_t value);

    GGML_a1d0ea7_API float   ggml_a1d0ea7_get_f32_1d(const struct ggml_a1d0ea7_tensor * tensor, int i);
    GGML_a1d0ea7_API void    ggml_a1d0ea7_set_f32_1d(const struct ggml_a1d0ea7_tensor * tensor, int i, float value);

    GGML_a1d0ea7_API void *  ggml_a1d0ea7_get_data    (const struct ggml_a1d0ea7_tensor * tensor);
    GGML_a1d0ea7_API float * ggml_a1d0ea7_get_data_f32(const struct ggml_a1d0ea7_tensor * tensor);

    GGML_a1d0ea7_API const char *         ggml_a1d0ea7_get_name(const struct ggml_a1d0ea7_tensor * tensor);
    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_set_name(struct ggml_a1d0ea7_tensor * tensor, const char * name);
    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_format_name(struct ggml_a1d0ea7_tensor * tensor, const char * fmt, ...);

    //
    // operations on tensors with backpropagation
    //

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_dup(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_add(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            struct ggml_a1d0ea7_tensor  * b);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_add_inplace(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            struct ggml_a1d0ea7_tensor  * b);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_add1(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            struct ggml_a1d0ea7_tensor  * b);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_add1_inplace(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            struct ggml_a1d0ea7_tensor  * b);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_acc(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            struct ggml_a1d0ea7_tensor  * b,
            size_t                nb1,
            size_t                nb2,
            size_t                nb3,
            size_t                offset);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_acc_inplace(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            struct ggml_a1d0ea7_tensor  * b,
            size_t                nb1,
            size_t                nb2,
            size_t                nb3,
            size_t                offset);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_sub(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            struct ggml_a1d0ea7_tensor  * b);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_sub_inplace(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            struct ggml_a1d0ea7_tensor  * b);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_mul(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            struct ggml_a1d0ea7_tensor  * b);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_mul_inplace(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            struct ggml_a1d0ea7_tensor  * b);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_div(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            struct ggml_a1d0ea7_tensor  * b);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_div_inplace(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            struct ggml_a1d0ea7_tensor  * b);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_sqr(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_sqr_inplace(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_sqrt(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_sqrt_inplace(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_log(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_log_inplace(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a);

    // return scalar
    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_sum(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a);

    // sums along rows, with input shape [a,b,c,d] return shape [1,b,c,d]
    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_sum_rows(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a);

    // mean along rows
    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_mean(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a);

    // if a is the same shape as b, and a is not parameter, return a
    // otherwise, return a new tensor: repeat(a) to fit in b
    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_repeat(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            struct ggml_a1d0ea7_tensor  * b);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_repeat_back(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            struct ggml_a1d0ea7_tensor  * b);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_abs(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_abs_inplace(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_sgn(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_sgn_inplace(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_neg(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_neg_inplace(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_step(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_step_inplace(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_relu(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_relu_inplace(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a);

    // TODO: double-check this computation is correct
    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_gelu(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_gelu_inplace(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_gelu_quick(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_gelu_quick_inplace(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_silu(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_silu_inplace(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a);

    // a - x
    // b - dy
    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_silu_back(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            struct ggml_a1d0ea7_tensor  * b);

    // normalize along rows
    // TODO: eps is hardcoded to 1e-5 for now
    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_norm(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_norm_inplace(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_rms_norm(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_rms_norm_inplace(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a);

    // a - x
    // b - dy
    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_rms_norm_back(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            struct ggml_a1d0ea7_tensor  * b);

    // A: n columns, m rows
    // B: n columns, p rows  (i.e. we transpose it internally)
    // result is m columns, p rows
    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_mul_mat(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            struct ggml_a1d0ea7_tensor  * b);

    // A: m columns, n rows,
    // B: p columns, n rows,
    // result is m columns, p rows
    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_out_prod(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            struct ggml_a1d0ea7_tensor  * b);

    //
    // operations on tensors without backpropagation
    //

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_scale(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            struct ggml_a1d0ea7_tensor  * b);

    // in-place, returns view(a)
    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_scale_inplace(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            struct ggml_a1d0ea7_tensor  * b);

    // b -> view(a,offset,nb1,nb2,3), return modified a
    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_set(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            struct ggml_a1d0ea7_tensor  * b,
            size_t                nb1,
            size_t                nb2,
            size_t                nb3,
            size_t                offset);

    // b -> view(a,offset,nb1,nb2,3), return view(a)
    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_set_inplace(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            struct ggml_a1d0ea7_tensor  * b,
            size_t                nb1,
            size_t                nb2,
            size_t                nb3,
            size_t                offset);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_set_1d(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            struct ggml_a1d0ea7_tensor  * b,
            size_t                offset);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_set_1d_inplace(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            struct ggml_a1d0ea7_tensor  * b,
            size_t                offset);

    // b -> view(a,offset,nb1,nb2,3), return modified a
    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_set_2d(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            struct ggml_a1d0ea7_tensor  * b,
            size_t                nb1,
            size_t                offset);

    // b -> view(a,offset,nb1,nb2,3), return view(a)
    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_set_2d_inplace(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            struct ggml_a1d0ea7_tensor  * b,
            size_t                nb1,
            size_t                offset);


    // a -> b, return view(b)
    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_cpy(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            struct ggml_a1d0ea7_tensor  * b);

    // make contiguous
    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_cont(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a);

    // return view(a), b specifies the new shape
    // TODO: when we start computing gradient, make a copy instead of view
    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_reshape(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            struct ggml_a1d0ea7_tensor  * b);

    // return view(a)
    // TODO: when we start computing gradient, make a copy instead of view
    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_reshape_1d(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            int64_t               ne0);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_reshape_2d(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            int64_t               ne0,
            int64_t               ne1);

    // return view(a)
    // TODO: when we start computing gradient, make a copy instead of view
    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_reshape_3d(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_reshape_4d(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2,
            int64_t               ne3);

    // offset in bytes
    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_view_1d(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            int64_t               ne0,
            size_t                offset);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_view_2d(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            size_t                nb1, // row stride in bytes
            size_t                offset);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_view_3d(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2,
            size_t                nb1, // row   stride in bytes
            size_t                nb2, // slice stride in bytes
            size_t                offset);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_view_4d(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2,
            int64_t               ne3,
            size_t                nb1, // row   stride in bytes
            size_t                nb2, // slice stride in bytes
            size_t                nb3,
            size_t                offset);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_permute(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            int                   axis0,
            int                   axis1,
            int                   axis2,
            int                   axis3);

    // alias for ggml_a1d0ea7_permute(ctx, a, 1, 0, 2, 3)
    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_transpose(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_get_rows(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            struct ggml_a1d0ea7_tensor  * b);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_get_rows_back(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            struct ggml_a1d0ea7_tensor  * b,
            struct ggml_a1d0ea7_tensor  * c);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_diag(
        struct ggml_a1d0ea7_context     * ctx,
        struct ggml_a1d0ea7_tensor      * a);

    // set elements above the diagonal to -INF
    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_diag_mask_inf(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            int                   n_past);

    // in-place, returns view(a)
    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_diag_mask_inf_inplace(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            int                   n_past);

    // set elements above the diagonal to 0
    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_diag_mask_zero(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            int                   n_past);

    // in-place, returns view(a)
    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_diag_mask_zero_inplace(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            int                   n_past);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_soft_max(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a);

    // in-place, returns view(a)
    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_soft_max_inplace(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_soft_max_back(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            struct ggml_a1d0ea7_tensor  * b);

    // in-place, returns view(a)
    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_soft_max_back_inplace(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            struct ggml_a1d0ea7_tensor  * b);

    // rotary position embedding
    // if mode & 1 == 1, skip n_past elements
    // if mode & 2 == 1, GPT-NeoX style
    // TODO: avoid creating a new tensor every time
    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_rope(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            int                   n_past,
            int                   n_dims,
            int                   mode);

    // in-place, returns view(a)
    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_rope_inplace(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            int                   n_past,
            int                   n_dims,
            int                   mode);

    // rotary position embedding backward, i.e compute dx from dy
    // a - dy
    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_rope_back(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            int                   n_past,
            int                   n_dims,
            int                   mode);

    // alibi position embedding
    // in-place, returns view(a)
    struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_alibi(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            int                   n_past,
            int                   n_head,
            float                 bias_max);

    // clamp
    // in-place, returns view(a)
    struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_clamp(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            float                 min,
            float                 max);

    // TODO: implement general-purpose convolutions
    // GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_conv_1d(
    //        struct ggml_a1d0ea7_context * ctx,
    //        struct ggml_a1d0ea7_tensor  * a,
    //        struct ggml_a1d0ea7_tensor  * b,
    //        int                   s0
    //        int                   p0,
    //        int                   d0);
    //
    // GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_conv_2d(
    //        struct ggml_a1d0ea7_context * ctx,
    //        struct ggml_a1d0ea7_tensor  * a,
    //        struct ggml_a1d0ea7_tensor  * b,
    //        int                   s0,
    //        int                   s1,
    //        int                   p0,
    //        int                   p1,
    //        int                   d0,
    //        int                   d1);

    // padding = half
    // TODO: we don't support extra parameters for now
    //       that's why we are hard-coding the stride, padding, and dilation
    //       not great ..
    // example:
    // a:      3   80  768    1
    // b:   3000   80    1    1
    // res: 3000  768    1    1
    // used in whisper
    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_conv_1d_s1_ph(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            struct ggml_a1d0ea7_tensor  * b);

    // used in whisper
    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_conv_1d_s2_ph(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            struct ggml_a1d0ea7_tensor  * b);

    // kernel size is a->ne[0] x a->ne[1]
    // stride is equal to kernel size
    // padding is zero
    // example:
    // a:     16   16    3  768
    // b:   1024 1024    3    1
    // res:   64   64  768    1
    // used in sam
    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_conv_2d_sk_p0(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            struct ggml_a1d0ea7_tensor  * b);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_flash_attn(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * q,
            struct ggml_a1d0ea7_tensor  * k,
            struct ggml_a1d0ea7_tensor  * v,
            bool                  masked);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_flash_attn_back(
           struct ggml_a1d0ea7_context * ctx,
           struct ggml_a1d0ea7_tensor  * q,
           struct ggml_a1d0ea7_tensor  * k,
           struct ggml_a1d0ea7_tensor  * v,
           struct ggml_a1d0ea7_tensor  * d,
           bool                  masked);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_flash_ff(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            struct ggml_a1d0ea7_tensor  * b0,
            struct ggml_a1d0ea7_tensor  * b1,
            struct ggml_a1d0ea7_tensor  * c0,
            struct ggml_a1d0ea7_tensor  * c1);

    // partition into non-overlapping windows with padding if needed
    // example:
    // a:   768   64   64    1
    // w:    14
    // res: 768   14   14    25
    // used in sam
    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_win_part(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            int                   w);

    // reverse of ggml_a1d0ea7_win_part
    // used in sam
    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_win_unpart(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor  * a,
            int                   w0,
            int                   h0,
            int                   w);

    // custom operators

    typedef void (*ggml_a1d0ea7_unary_op_f32_t) (const int, float *, const float *);
    typedef void (*ggml_a1d0ea7_binary_op_f32_t)(const int, float *, const float *, const float *);

    typedef void (*ggml_a1d0ea7_custom1_op_f32_t)(struct ggml_a1d0ea7_tensor *, const struct ggml_a1d0ea7_tensor *);
    typedef void (*ggml_a1d0ea7_custom2_op_f32_t)(struct ggml_a1d0ea7_tensor *, const struct ggml_a1d0ea7_tensor *, const struct ggml_a1d0ea7_tensor *);
    typedef void (*ggml_a1d0ea7_custom3_op_f32_t)(struct ggml_a1d0ea7_tensor *, const struct ggml_a1d0ea7_tensor *, const struct ggml_a1d0ea7_tensor *, const struct ggml_a1d0ea7_tensor *);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_map_unary_f32(
            struct ggml_a1d0ea7_context        * ctx,
            struct ggml_a1d0ea7_tensor         * a,
                   ggml_a1d0ea7_unary_op_f32_t   fun);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_map_unary_inplace_f32(
            struct ggml_a1d0ea7_context        * ctx,
            struct ggml_a1d0ea7_tensor         * a,
                   ggml_a1d0ea7_unary_op_f32_t   fun);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_map_binary_f32(
            struct ggml_a1d0ea7_context         * ctx,
            struct ggml_a1d0ea7_tensor          * a,
            struct ggml_a1d0ea7_tensor          * b,
                   ggml_a1d0ea7_binary_op_f32_t   fun);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_map_binary_inplace_f32(
            struct ggml_a1d0ea7_context         * ctx,
            struct ggml_a1d0ea7_tensor          * a,
            struct ggml_a1d0ea7_tensor          * b,
                   ggml_a1d0ea7_binary_op_f32_t   fun);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_map_custom1_f32(
            struct ggml_a1d0ea7_context          * ctx,
            struct ggml_a1d0ea7_tensor           * a,
                   ggml_a1d0ea7_custom1_op_f32_t   fun);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_map_custom1_inplace_f32(
            struct ggml_a1d0ea7_context          * ctx,
            struct ggml_a1d0ea7_tensor           * a,
                   ggml_a1d0ea7_custom1_op_f32_t   fun);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_map_custom2_f32(
            struct ggml_a1d0ea7_context          * ctx,
            struct ggml_a1d0ea7_tensor           * a,
            struct ggml_a1d0ea7_tensor           * b,
                   ggml_a1d0ea7_custom2_op_f32_t   fun);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_map_custom2_inplace_f32(
            struct ggml_a1d0ea7_context          * ctx,
            struct ggml_a1d0ea7_tensor           * a,
            struct ggml_a1d0ea7_tensor           * b,
                   ggml_a1d0ea7_custom2_op_f32_t   fun);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_map_custom3_f32(
            struct ggml_a1d0ea7_context          * ctx,
            struct ggml_a1d0ea7_tensor           * a,
            struct ggml_a1d0ea7_tensor           * b,
            struct ggml_a1d0ea7_tensor           * c,
                   ggml_a1d0ea7_custom3_op_f32_t   fun);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_map_custom3_inplace_f32(
            struct ggml_a1d0ea7_context          * ctx,
            struct ggml_a1d0ea7_tensor           * a,
            struct ggml_a1d0ea7_tensor           * b,
            struct ggml_a1d0ea7_tensor           * c,
                   ggml_a1d0ea7_custom3_op_f32_t   fun);

    // loss function

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_cross_entropy_loss(
            struct ggml_a1d0ea7_context         * ctx,
            struct ggml_a1d0ea7_tensor          * a,
            struct ggml_a1d0ea7_tensor          * b);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_cross_entropy_loss_back(
            struct ggml_a1d0ea7_context         * ctx,
            struct ggml_a1d0ea7_tensor          * a,
            struct ggml_a1d0ea7_tensor          * b,
            struct ggml_a1d0ea7_tensor          * c);

    //
    // automatic differentiation
    //

    GGML_a1d0ea7_API void ggml_a1d0ea7_set_param(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_tensor * tensor);

    GGML_a1d0ea7_API void ggml_a1d0ea7_build_forward_expand(struct ggml_a1d0ea7_cgraph * cgraph, struct ggml_a1d0ea7_tensor * tensor);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_cgraph ggml_a1d0ea7_build_forward (struct ggml_a1d0ea7_tensor * tensor);
    GGML_a1d0ea7_API struct ggml_a1d0ea7_cgraph ggml_a1d0ea7_build_backward(struct ggml_a1d0ea7_context * ctx, struct ggml_a1d0ea7_cgraph * gf, bool keep);

    GGML_a1d0ea7_API void ggml_a1d0ea7_graph_compute(struct ggml_a1d0ea7_context * ctx, struct ggml_a1d0ea7_cgraph * cgraph);
    GGML_a1d0ea7_API void ggml_a1d0ea7_graph_reset  (struct ggml_a1d0ea7_cgraph * cgraph);

    GGML_a1d0ea7_API struct ggml_a1d0ea7_tensor * ggml_a1d0ea7_graph_get_tensor(struct ggml_a1d0ea7_cgraph * cgraph, const char * name);

    GGML_a1d0ea7_API void               ggml_a1d0ea7_graph_export(const struct ggml_a1d0ea7_cgraph * cgraph, const char * fname);
    GGML_a1d0ea7_API struct ggml_a1d0ea7_cgraph ggml_a1d0ea7_graph_import(const char * fname, struct ggml_a1d0ea7_context ** ctx_data, struct ggml_a1d0ea7_context ** ctx_eval);

    // print info and performance information for the graph
    GGML_a1d0ea7_API void ggml_a1d0ea7_graph_print(const struct ggml_a1d0ea7_cgraph * cgraph);

    // dump the graph into a file using the dot format
    GGML_a1d0ea7_API void ggml_a1d0ea7_graph_dump_dot(const struct ggml_a1d0ea7_cgraph * gb, const struct ggml_a1d0ea7_cgraph * gf, const char * filename);

    //
    // optimization
    //

    // optimization methods
    enum ggml_a1d0ea7_opt_type {
        GGML_a1d0ea7_OPT_ADAM,
        GGML_a1d0ea7_OPT_LBFGS,
    };

    // linesearch methods
    enum ggml_a1d0ea7_linesearch {
        GGML_a1d0ea7_LINESEARCH_DEFAULT = 1,

        GGML_a1d0ea7_LINESEARCH_BACKTRACKING_ARMIJO       = 0,
        GGML_a1d0ea7_LINESEARCH_BACKTRACKING_WOLFE        = 1,
        GGML_a1d0ea7_LINESEARCH_BACKTRACKING_STRONG_WOLFE = 2,
    };

    // optimization return values
    enum ggml_a1d0ea7_opt_result {
        GGML_a1d0ea7_OPT_OK = 0,
        GGML_a1d0ea7_OPT_DID_NOT_CONVERGE,
        GGML_a1d0ea7_OPT_NO_CONTEXT,
        GGML_a1d0ea7_OPT_INVALID_WOLFE,
        GGML_a1d0ea7_OPT_FAIL,

        GGML_a1d0ea7_LINESEARCH_FAIL = -128,
        GGML_a1d0ea7_LINESEARCH_MINIMUM_STEP,
        GGML_a1d0ea7_LINESEARCH_MAXIMUM_STEP,
        GGML_a1d0ea7_LINESEARCH_MAXIMUM_ITERATIONS,
        GGML_a1d0ea7_LINESEARCH_INVALID_PARAMETERS,
    };

    // optimization parameters
    //
    //   see ggml.c (ggml_a1d0ea7_opt_default_params) for default values
    //
    struct ggml_a1d0ea7_opt_params {
        enum ggml_a1d0ea7_opt_type type;

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

            enum ggml_a1d0ea7_linesearch linesearch;
        } lbfgs;
    };

    struct ggml_a1d0ea7_opt_context {
        struct ggml_a1d0ea7_context * ctx;
        struct ggml_a1d0ea7_opt_params params;

        int iter;
        int64_t nx; // number of parameter elements

        bool just_initialized;

        struct {
            struct ggml_a1d0ea7_tensor * x;  // view of the parameters
            struct ggml_a1d0ea7_tensor * g1; // gradient
            struct ggml_a1d0ea7_tensor * g2; // gradient squared
            struct ggml_a1d0ea7_tensor * m;  // first moment
            struct ggml_a1d0ea7_tensor * v;  // second moment
            struct ggml_a1d0ea7_tensor * mh; // first moment hat
            struct ggml_a1d0ea7_tensor * vh; // second moment hat
            struct ggml_a1d0ea7_tensor * pf; // past function values
            float fx_best;
            float fx_prev;
            int n_no_improvement;
        } adam;

        struct {
            struct ggml_a1d0ea7_tensor * x;    // current parameters
            struct ggml_a1d0ea7_tensor * xp;   // previous parameters
            struct ggml_a1d0ea7_tensor * g;    // current gradient
            struct ggml_a1d0ea7_tensor * gp;   // previous gradient
            struct ggml_a1d0ea7_tensor * d;    // search direction
            struct ggml_a1d0ea7_tensor * pf;   // past function values
            struct ggml_a1d0ea7_tensor * lmal; // the L-BFGS memory alpha
            struct ggml_a1d0ea7_tensor * lmys; // the L-BFGS memory ys
            struct ggml_a1d0ea7_tensor * lms;  // the L-BFGS memory s
            struct ggml_a1d0ea7_tensor * lmy;  // the L-BFGS memory y
            float fx_best;
            float step;
            int j;
            int k;
            int end;
            int n_no_improvement;
        } lbfgs;
    };

    GGML_a1d0ea7_API struct ggml_a1d0ea7_opt_params ggml_a1d0ea7_opt_default_params(enum ggml_a1d0ea7_opt_type type);

    // optimize the function defined by the tensor f
    GGML_a1d0ea7_API enum ggml_a1d0ea7_opt_result ggml_a1d0ea7_opt(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_opt_params params,
            struct ggml_a1d0ea7_tensor * f);

    // initialize optimizer context
    GGML_a1d0ea7_API void ggml_a1d0ea7_opt_init(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_opt_context * opt,
            struct ggml_a1d0ea7_opt_params params,
            int64_t nx);

    // continue optimizing the function defined by the tensor f
    GGML_a1d0ea7_API enum ggml_a1d0ea7_opt_result ggml_a1d0ea7_opt_resume(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_opt_context * opt,
            struct ggml_a1d0ea7_tensor * f);

    // continue optimizing the function defined by the tensor f
    GGML_a1d0ea7_API enum ggml_a1d0ea7_opt_result ggml_a1d0ea7_opt_resume_g(
            struct ggml_a1d0ea7_context * ctx,
            struct ggml_a1d0ea7_opt_context * opt,
            struct ggml_a1d0ea7_tensor * f,
            struct ggml_a1d0ea7_cgraph * gf,
            struct ggml_a1d0ea7_cgraph * gb);

    //
    // quantization
    //

    GGML_a1d0ea7_API size_t ggml_a1d0ea7_quantize_q4_0(const float * src, void * dst, int n, int k, int64_t * hist);
    GGML_a1d0ea7_API size_t ggml_a1d0ea7_quantize_q4_1(const float * src, void * dst, int n, int k, int64_t * hist);
    GGML_a1d0ea7_API size_t ggml_a1d0ea7_quantize_q5_0(const float * src, void * dst, int n, int k, int64_t * hist);
    GGML_a1d0ea7_API size_t ggml_a1d0ea7_quantize_q5_1(const float * src, void * dst, int n, int k, int64_t * hist);
    GGML_a1d0ea7_API size_t ggml_a1d0ea7_quantize_q8_0(const float * src, void * dst, int n, int k, int64_t * hist);

    GGML_a1d0ea7_API size_t ggml_a1d0ea7_quantize_chunk(enum ggml_a1d0ea7_type type, const float * src, void * dst, int start, int n, int64_t * hist);

    //
    // system info
    //

    GGML_a1d0ea7_API int ggml_a1d0ea7_cpu_has_avx        (void);
    GGML_a1d0ea7_API int ggml_a1d0ea7_cpu_has_avx2       (void);
    GGML_a1d0ea7_API int ggml_a1d0ea7_cpu_has_avx512     (void);
    GGML_a1d0ea7_API int ggml_a1d0ea7_cpu_has_avx512_vbmi(void);
    GGML_a1d0ea7_API int ggml_a1d0ea7_cpu_has_avx512_vnni(void);
    GGML_a1d0ea7_API int ggml_a1d0ea7_cpu_has_fma        (void);
    GGML_a1d0ea7_API int ggml_a1d0ea7_cpu_has_neon       (void);
    GGML_a1d0ea7_API int ggml_a1d0ea7_cpu_has_arm_fma    (void);
    GGML_a1d0ea7_API int ggml_a1d0ea7_cpu_has_f16c       (void);
    GGML_a1d0ea7_API int ggml_a1d0ea7_cpu_has_fp16_va    (void);
    GGML_a1d0ea7_API int ggml_a1d0ea7_cpu_has_wasm_simd  (void);
    GGML_a1d0ea7_API int ggml_a1d0ea7_cpu_has_blas       (void);
    GGML_a1d0ea7_API int ggml_a1d0ea7_cpu_has_cublas     (void);
    GGML_a1d0ea7_API int ggml_a1d0ea7_cpu_has_clblast    (void);
    GGML_a1d0ea7_API int ggml_a1d0ea7_cpu_has_gpublas    (void);
    GGML_a1d0ea7_API int ggml_a1d0ea7_cpu_has_sse3       (void);
    GGML_a1d0ea7_API int ggml_a1d0ea7_cpu_has_vsx        (void);

    //
    // Internal types and functions exposed for tests and benchmarks
    //

#ifdef  __cplusplus
    // restrict not standard in C++
#define GGML_a1d0ea7_RESTRICT
#else
#define GGML_a1d0ea7_RESTRICT restrict
#endif
    typedef void (*dequantize_row_q_t)(const void * GGML_a1d0ea7_RESTRICT x, float * GGML_a1d0ea7_RESTRICT y, int k);
    typedef void (*quantize_row_q_t)  (const float * GGML_a1d0ea7_RESTRICT x, void * GGML_a1d0ea7_RESTRICT y, int k);
    typedef void (*vec_dot_q_t)       (const int n, float * GGML_a1d0ea7_RESTRICT s, const void * GGML_a1d0ea7_RESTRICT x, const void * GGML_a1d0ea7_RESTRICT y);

    typedef struct {
        dequantize_row_q_t dequantize_row_q;
        quantize_row_q_t   quantize_row_q;
        quantize_row_q_t   quantize_row_q_reference;
        quantize_row_q_t   quantize_row_q_dot;
        vec_dot_q_t        vec_dot_q;
        enum ggml_a1d0ea7_type     vec_dot_type;
    } quantize_fns_t;

    quantize_fns_t ggml_a1d0ea7_internal_get_quantize_fn(size_t i);

#ifdef  __cplusplus
}
#endif
