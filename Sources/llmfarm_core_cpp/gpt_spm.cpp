#include "./spm-headers/gpt_spm.h"
#include "common.h"
#include "sampling.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cinttypes>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <random>



int LLAMA_BUILD_NUMBER = 0;
char const *LLAMA_COMMIT = "unknown";
char const *LLAMA_COMPILER = "Apple clang version 15.0.0 (clang-1500.3.9.4)";
char const *LLAMA_BUILD_TARGET = "x86_64-apple-darwin23.5.0";





const char * print_system_info(void) {
    static std::string s;

    s  = "";
    s += "AVX = "         + std::to_string(ggml_cpu_has_avx())         + " | ";
    s += "AVX2 = "        + std::to_string(ggml_cpu_has_avx2())        + " | ";
    s += "AVX512 = "      + std::to_string(ggml_cpu_has_avx512())      + " | ";
    s += "AVX512_VBMI = " + std::to_string(ggml_cpu_has_avx512_vbmi()) + " | ";
    s += "AVX512_VNNI = " + std::to_string(ggml_cpu_has_avx512_vnni()) + " | ";
    s += "FMA = "         + std::to_string(ggml_cpu_has_fma())         + " | ";
    s += "NEON = "        + std::to_string(ggml_cpu_has_neon())        + " | ";
    s += "ARM_FMA = "     + std::to_string(ggml_cpu_has_arm_fma())     + " | ";
    s += "F16C = "        + std::to_string(ggml_cpu_has_f16c())        + " | ";
    s += "FP16_VA = "     + std::to_string(ggml_cpu_has_fp16_va())     + " | ";
    s += "WASM_SIMD = "   + std::to_string(ggml_cpu_has_wasm_simd())   + " | ";
    s += "BLAS = "        + std::to_string(ggml_cpu_has_blas())        + " | ";
    s += "SSE3 = "        + std::to_string(ggml_cpu_has_sse3())        + " | ";
    s += "SSSE3 = "       + std::to_string(ggml_cpu_has_ssse3())       + " | ";
    s += "VSX = "         + std::to_string(ggml_cpu_has_vsx())         + " | ";

    return s.c_str();
}

// gpt_token gpt_base_token_bos(){
//    return 0;
// }

// gpt_token gpt_base_token_eos() {
//     return 0;
// }

// struct gpt_context_params gpt_context_default_params() {
//     struct gpt_context_params result = {
//         /*.n_ctx                       =*/ 512,
//         /*.n_parts                     =*/ -1,
//         /*.seed                        =*/ 0,
//         /*.n_batch                     =*/ 8,
//         /*.f16_kv                      =*/ false,
//         /*.logits_all                  =*/ false,
//         /*.vocab_only                  =*/ false,
//         /*.use_mmap                    =*/ true,
//         /*.use_mlock                   =*/ false,
//         /*.embedding                   =*/ false,
//         /*.progress_callback           =*/ nullptr,
//         /*.progress_callback_user_data =*/ nullptr,
//     };
//     return result;
// };


// int gpt_base_n_vocab(struct gpt_base_context * ctx) {
//     return ctx->vocab.id_to_token.size();
// }

// int gpt_base_n_ctx(struct gpt_base_context * ctx) {
//     return ctx->model.hparams.n_ctx;
// }

// int gpt_base_n_embd(struct gpt_base_context * ctx) {
//     return ctx->model.hparams.n_embd;
// }

// float * gpt_base_get_logits(struct gpt_base_context * ctx) {
//     return ctx->logits.data();
// }

// float * gpt_base_get_embeddings(struct gpt_base_context * ctx) {
//     return ctx->embedding.data();
// }

// gpt_token gpt_base_str_to_token(struct gpt_base_context * ctx, const char * str) {
//     return ctx->vocab.token_to_id[str];
// }

// const char * gpt_base_token_to_str(struct gpt_base_context * ctx, gpt_token token) {
//     if (token >= ctx->vocab.id_to_token.size()) {
//         return nullptr;
//     }
//     return ctx->vocab.id_to_token[token].c_str();
// }




// int gpt_base_tokenize(
//         struct gpt_base_context * ctx,
//                   const char * text,
//                  gpt_token * tokens,
//                          int   n_max_tokens,
//                         bool   add_bos) {
// //    auto res = gptneox_tokenize(ctx->vocab, text, add_bos);
//     auto res = gpt_tokenize(ctx->vocab, text);
    
//     if (n_max_tokens < (int) res.size()) {
//         fprintf(stderr, "%s: too many tokens\n", __func__);
//         return -((int) res.size());
//     }

//     for (size_t i = 0; i < res.size(); i++) {
//         tokens[i] = res[i];
//     }

//     return res.size();
// }

// void gpt_base_shift_kv_cache(struct gpt_base_context * ctx, int n) {
//     auto & model = ctx->model;
//     auto & kv_self = model.kv_self;
//     auto & hparams = model.hparams;
//     auto n_layer = hparams.n_layer;
//     auto n_embd = hparams.n_embd;
//     auto n_ctx = hparams.n_ctx;
//     for(int il = 0; il < n_layer; il++) {
//         // K: Embeddings are in regular order so moving them is easy as copying the memory
//         {
//             int elem_byte_size = ggml_dadbed9_element_size(kv_self.k);
//             uint8_t * dst_ptr = ((uint8_t *)kv_self.k->data) + (elem_byte_size * n_embd * (il * n_ctx));
//             uint8_t * src_ptr = ((uint8_t *)kv_self.k->data) + (elem_byte_size * n_embd * (il * n_ctx + n));
//             memcpy(dst_ptr, src_ptr, elem_byte_size * n_embd * (n_ctx - n));
//         }
        
//         // V: Embeddings are transposed so each embedding element must be copied separately
//         {
//             int elem_byte_size = ggml_dadbed9_element_size(kv_self.v);
//             for(int i = 0; i < n_embd; i++) {
//                 uint8_t * dst_ptr = ((uint8_t *)kv_self.v->data) + (elem_byte_size * (il * n_ctx * i));
//                 uint8_t * src_ptr = ((uint8_t *)kv_self.v->data) + (elem_byte_size * (il * n_ctx * i + n));
//                 memcpy(dst_ptr, src_ptr, elem_byte_size * (n_ctx - n));
//             }
//         }
//     }
// }



// int32_t gpt_base_sample(struct gpt_base_context * ctx, int top_k, float top_p, float temp) {
//     const int64_t t_start_sample_us = ggml_dadbed9_time_us();
//     int n_logits = ctx->vocab.id_to_token.size();    

// //    gpt_vocab::id smpl = gpt_sample_top_k_top_p(n_logits, ctx->logits.data() + (ctx->logits.size() - ctx->vocab.id_to_token.size()), top_k, top_p, temp, ctx->rng);
//     gpt_vocab::id smpl = gpt_sample_top_k_top_p(ctx->vocab, ctx->logits.data() + (ctx->logits.size() - ctx->vocab.id_to_token.size()), top_k, top_p, temp, ctx->rng);
//     if (ctx) {
//         ctx->t_sample_us += ggml_dadbed9_time_us() - t_start_sample_us;
//     }
//     return  smpl;
// }


// int32_t gpt_base_sample_repeat(struct gpt_base_context * ctx,
//                                const int32_t * last_n_tokens_data,
//                                size_t last_n_tokens_data_size,
//                                int top_k, float top_p, float temp,
//                                int repeat_last_n,
//                                float repeat_penalty) {
//     const int64_t t_start_sample_us = ggml_dadbed9_time_us();
//     int n_logits = ctx->vocab.id_to_token.size();
// //    gpt_vocab::id smpl = gpt_sample_top_k_top_p_repeat(n_logits, ctx->logits.data() + (ctx->logits.size() - ctx->vocab.id_to_token.size()),
// //                                                       last_n_tokens_data,last_n_tokens_data_size,
// //                                                       top_k, top_p, temp,
// //                                                       repeat_last_n,repeat_penalty,
// //                                                       ctx->rng);
//     gpt_vocab::id smpl = gpt_sample_top_k_top_p_repeat(ctx->vocab, ctx->logits.data() + (ctx->logits.size() - ctx->vocab.id_to_token.size()),
//                                                        last_n_tokens_data,last_n_tokens_data_size,
//                                                        top_k, top_p, temp,
//                                                        repeat_last_n,repeat_penalty,
//                                                        ctx->rng);
//     if (ctx) {
//         ctx->t_sample_us += ggml_dadbed9_time_us() - t_start_sample_us;
//     }
//     return  smpl;
// }


// void rwkv_tokenize(){
    
// }

// void rwkv_init_logits(struct rwkv_context * model) {

//     const size_t n_vocab = rwkv_get_logits_len(model);


//     float * state = (float * )malloc(sizeof(float) * rwkv_get_state_len(model));
//     float * logits = (float * )malloc(sizeof(float) * n_vocab);

//     uint32_t prompt_seq[] = { 10002, 209, 312, 209, 74 };

//     const size_t prompt_length = 4;

//     rwkv_init_state(model, state);
//     rwkv_eval_sequence(model, prompt_seq, prompt_length, state, state, logits);

// }

//int32_t rwkv_sample(int n_logits, float * logits, int top_k, float top_p, float temp) {
//    std::mt19937 rng = std::mt19937(time(NULL));
////    gpt_vocab::id smpl = gpt_sample_top_k_top_p(n_logits, logits, top_k, top_p, temp, rng);
//    gpt_vocab::id smpl = gpt_sample_top_k_top_p(n_logits, logits, top_k, top_p, temp, rng);
//    return  smpl;
//}


//int32_t rwkv_sample_repeat(int n_logits, float * logits,
//                               const int32_t * last_n_tokens_data,
//                               size_t last_n_tokens_data_size,
//                               int top_k, float top_p, float temp,
//                               int repeat_last_n,
//                               float repeat_penalty) {
//    std::mt19937 rng = std::mt19937(time(NULL));
//    gpt_vocab::id smpl = gpt_sample_top_k_top_p_repeat(n_logits, logits,
//                                                       last_n_tokens_data,last_n_tokens_data_size,
//                                                       top_k, top_p, temp,
//                                                       repeat_last_n,repeat_penalty,
//                                                       rng);
//    return  smpl;
//}

// bool llama_save_state(struct llama_context * ctx, const char * fname){
//     const size_t state_size = llama_get_state_size(ctx);
//     uint8_t * state_mem = new uint8_t[state_size];
//     FILE *fp_write = fopen(fname, "wb");
//     llama_copy_state_data(ctx, state_mem); // could also copy directly to memory mapped file
//     fwrite(&state_size, 1, sizeof(state_size), fp_write);
//     fwrite(state_mem, 1, state_size, fp_write);
//     fclose(fp_write);
//     delete[] state_mem;
//     return  true;
// }

// bool llama_load_state(struct llama_context * ctx, const char * fname){
//     FILE *fp_read = fopen(fname, "rb");
//     size_t state_size = 0;
//     fread(&state_size, 1, sizeof(state_size), fp_read);
//     uint8_t * state_mem = new uint8_t[state_size];
//     const size_t ret = fread(state_mem, 1, state_size, fp_read);
//     if (ret != state_size) {
//         fprintf(stderr, "\n%s : failed to read state\n", __func__);
//         GGML_ASSERT(false);
//     }
//     llama_set_state_data(ctx, state_mem);
//     delete[] state_mem;
//     return  true;
// }

char* llama_token_to_str_res = new char[3];


struct callback_data {
    std::vector<uint8_t> data;
};


static std::string ggml_ne_string(const ggml_tensor * t) {
    std::string str;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        str += std::to_string(t->ne[i]);
        if (i + 1 < GGML_MAX_DIMS) {
            str += ", ";
        }
    }
    return str;
}

char tensor_name_str[128] = {0};

char * get_tensor_name(struct ggml_tensor * t){
    if (t){
        //        sprintf(tensor_name_str, "%s", t->name);
        const struct ggml_tensor * src0 = t->src[0];
        const struct ggml_tensor * src1 = t->src[1];
        
        if (src1) {
            sprintf(tensor_name_str, "%s{%s}", src1->name, ggml_ne_string(src1).c_str());
        }
    }
    return  tensor_name_str;
}

int check_tensor_name(struct ggml_tensor * t){
    if (t){
        printf("%s\n",t->name);
        return strcmp(t->name,"result_output");
    }
    return -1;
}


/*struct llama_sampling_context*/ 
 struct common_sampler*  init_sampling(llama_model* model,
                                                int32_t     n_prev                = 64,                 // number of previous tokens to remember
                                                int32_t     top_k                 = 40,                 // <= 0 to use vocab size
                                                float       top_p                 = 0.95f,              // 1.0 = disabled
                                                float       min_p                 = 0.05f,              // 0.0 = disabled
                                                float       tfs_z                 = 1.00f,              // 1.0 = disabled
                                                float       typical_p             = 1.00f,              // 1.0 = disabled
                                                float       temp                  = 0.80f,              // <= 0.0 to sample greedily, 0.0 to not output probabilities
                                                float       dynatemp_range        = 0.00f,              // 0.0 = disabled
                                                float       dynatemp_exponent     = 1.00f,              // controls how entropy maps to temperature in dynamic temperature sampler
                                                int32_t     penalty_last_n        = 64,                 // last n tokens to penalize (0 = disable penalty, -1 = context size)
                                                float       penalty_repeat        = 1.00f,              // 1.0 = disabled
                                                float       penalty_freq          = 0.00f,              // 0.0 = disabled
                                                float       penalty_present       = 0.00f,              // 0.0 = disabled
                                                int32_t     mirostat              = 0,                 // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
                                                float       mirostat_tau          = 5.00f,              // target entropy
                                                float       mirostat_eta          = 0.10f,              // learning rate
                                                bool        penalize_nl           = false,              // consider newlines as a repeatable token
                                                uint32_t    seed                  = LLAMA_DEFAULT_SEED,
                                                const char * grammar_path = ""){
    // sparams
    struct common_sampler_params  sparams;
    sparams.n_prev = n_prev;
    sparams.top_k = top_k;
    sparams.top_p = top_p;              // 1.0 = disabled
    sparams.min_p = min_p;             // 0.0 = disabled
    sparams.tfs_z = tfs_z;              // 1.0 = disabled
    sparams.typ_p = typical_p;             // 1.0 = disabled
    sparams.temp = temp;             // <= 0.0 to sample greedily, 0.0 to not output probabilities
    sparams.dynatemp_range  = dynatemp_range;             // 0.0 = disabled
    sparams.dynatemp_exponent = dynatemp_exponent;            // controls how entropy maps to temperature in dynamic temperature sampler
    sparams.penalty_last_n = penalty_last_n;                // last n tokens to penalize (0 = disable penalty, -1 = context size)
    sparams.penalty_repeat = penalty_repeat;            // 1.0 = disabled
    sparams.penalty_freq   = penalty_freq;             // 0.0 = disabled
    sparams.penalty_present = penalty_present;            // 0.0 = disabled
    sparams.mirostat    = mirostat;                 // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    sparams.mirostat_tau   = mirostat_tau;           // target entropy
    sparams.mirostat_eta   = mirostat_eta;
    sparams.seed = seed;
    if (sparams.seed == 0)
        sparams.seed = LLAMA_DEFAULT_SEED;        

    if (grammar_path != nullptr &&  grammar_path != ""){
        printf("Grammar: %s",grammar_path);
        std::ifstream file(grammar_path);
        std::copy(
                std::istreambuf_iterator<char>(file),
                std::istreambuf_iterator<char>(),
                std::back_inserter(sparams.grammar)
            );
    }

    struct common_sampler * ctx_sampling = common_sampler_init(model, sparams);
    return ctx_sampling;
}

llama_token spm_llama_sampling_sample(
        /*llama_sampling_context*/common_sampler * ctx_sampling,
        struct llama_context * ctx_main,
        // struct llama_context * ctx_cfg,
        int idx = -1,
        bool grammar_first = false)
{

    //    llama_sampling_sample(ctx_sampling,ctx_main,ctx_cfg,idx);
    common_sampler_sample(ctx_sampling, ctx_main, idx, grammar_first);
}

void spm_llama_sampling_accept(
        struct /*llama_sampling_context*/common_sampler * ctx_sampling,
        struct llama_context * ctx_main,
        llama_token id,
        bool apply_grammar)
{
    // llama_sampling_accept(ctx_sampling,ctx_main,id,apply_grammar);
    common_sampler_accept(ctx_sampling, id, apply_grammar);
}

