#pragma once


#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "llama_dadbed9.h"
#include "llama.h"

#include "../ggml/ggml_dadbed9.h"



#ifdef __cplusplus
extern "C" {
#endif

typedef int gpt_token;
typedef int llama_token;



const char * print_system_info(void);

typedef struct gpt_neox_token_data {
    int id;  // token id
    float logit; // log-odds of the token
    float p;     // probability of the token
} gpt_neox_token_data;

typedef struct gpt_neox_token_data_array {
    gpt_neox_token_data * data;
    size_t size;
    bool sorted;
} gpt_neox_token_data_array;

typedef struct gpt_token_data {
    gpt_token id; // token id
    float logit;    // log-odds of the token
    float p;        // probability of the token
} gpt_token_data;

typedef struct gpt_token_data_array {
    gpt_token_data * data;
    size_t size;
    bool sorted;
} gpt_token_data_array;


typedef void (*gpt_progress_callback)(float progress, void *ctx);

//struct gpt_params {
//    int32_t seed      = -1;  // RNG seed
//    int32_t n_threads = std::min(4, (int32_t) std::thread::hardware_concurrency());
//    int32_t n_predict = 200; // new tokens to predict
//    int32_t n_batch   = 8;   // batch size for prompt processing
//
//    // sampling parameters
//    int32_t top_k          = 40;
//    float   top_p          = 0.9f;
//    float   temp           = 0.9f;
//    int32_t repeat_last_n  = 64;
//    float   repeat_penalty = 1.00f;
//
//    std::string model      = "models/gpt-2-117M/ggml-model.bin"; // model path
//    std::string prompt     = "";
//    std::string token_test = "";
//
//    bool    interactive      = false;
//    int32_t interactive_port = -1;
//
//    int32_t n_gpu_layers     = 0;
//};

struct gpt_context_params {
    int n_ctx;   // text context
    int n_parts; // -1 for default
    uint32_t seed;    // RNG seed, 0 for random
    int32_t n_batch;

    bool f16_kv;     // use fp16 for KV cache
    bool logits_all; // the gptneox_eval() call computes all logits, not just the last one
    bool vocab_only; // only load the vocabulary, no weights
    bool use_mmap;   // use mmap if possible
    bool use_mlock;  // force system to keep model in RAM
    bool embedding;  // embedding mode only

    // called with a progress value between 0 and 1, pass NULL to disable
    gpt_progress_callback progress_callback;
    // context pointer passed to the progress callback
    void * progress_callback_user_data;
    int32_t n_thread;
};

struct gpt_context_params gpt_context_default_params();



gpt_token gpt_base_token_bos();
gpt_token gpt_base_token_eos();




int gpt_base_n_vocab(struct gpt_base_context * ctx);

int gpt_base_n_ctx(struct gpt_base_context * ctx);

int gpt_base_n_embd(struct gpt_base_context * ctx);

float * gpt_base_get_logits(struct gpt_base_context * ctx);

float * gpt_base_get_embeddings(struct gpt_base_context * ctx);

gpt_token gpt_base_str_to_token(struct gpt_base_context * ctx, const char * str);

const char * gpt_base_token_to_str(struct gpt_base_context * ctx, gpt_token token);


int gpt_base_tokenize(
        struct gpt_base_context * ctx,
                  const char * text,
                 gpt_token * tokens,
                         int   n_max_tokens,
                        bool   add_bos);

void gpt_base_shift_kv_cache(struct gpt_base_context * ctx, int n);


int32_t gpt_base_sample(struct gpt_base_context * ctx, int top_k, float top_p, float temp);
int32_t gpt_base_sample_repeat(struct gpt_base_context * ctx,
                                           const int32_t * last_n_tokens_data,
                                           size_t last_n_tokens_data_size,
                                           int top_k, float top_p, float temp,
                                           int repeat_last_n,
                                           float repeat_penalty);

void rwkv_init_logits(struct rwkv_context * model);
int32_t rwkv_sample(int n_logits, float * logits, int top_k, float top_p, float temp);
int32_t rwkv_sample_repeat(int n_logits, float * logits,
                               const int32_t * last_n_tokens_data,
                               size_t last_n_tokens_data_size,
                               int top_k, float top_p, float temp,
                               int repeat_last_n,
                           float repeat_penalty);

//const char * llama_token_to_str(const struct llama_context * ctx, llama_token token);

bool llama_save_state(struct llama_context * ctx, const char * fname);
bool llama_load_state(struct llama_context * ctx, const char * fname);

struct llama_grammar* llama_load_grammar(const char* grammar_path);

//struct llama_dadbed9_token_data_array;

void llama_sample_grammar_for_dadbed9(struct llama_context * ctx, llama_dadbed9_token_data_array * candidates, const struct llama_grammar * grammar );
llama_token llama_sample_token_for_dadbed9(struct llama_context * ctx, llama_dadbed9_token_data_array * candidates );
llama_token llama_sample_token_mirostat_for_dadbed9(struct llama_context * ctx, llama_dadbed9_token_data_array * candidates,float tau,float   eta,int   m,float * mu );
llama_token llama_sample_token_mirostat_v2_for_dadbed9(struct llama_context * ctx, llama_dadbed9_token_data_array * candidates,float tau,float   eta, float * mu ) ;


typedef struct llama_sampling_params_spm {
    int32_t     n_prev                ;       // number of previous tokens to remember
    int32_t     n_probs               ;        // if greater than 0, output the probabilities of top n_probs tokens.
    int32_t     min_keep              ;        // 0 = disabled, otherwise samplers should return at least min_keep tokens
    int32_t     top_k                 ;       // <= 0 to use vocab size
    float       top_p                 ;    // 1.0 = disabled
    float       min_p                 ;    // 0.0 = disabled
    float       tfs_z                 ;    // 1.0 = disabled
    float       typical_p             ;    // 1.0 = disabled
    float       temp                  ;    // <= 0.0 to sample greedily, 0.0 to not output probabilities
    float       dynatemp_range        ;    // 0.0 = disabled
    float       dynatemp_exponent     ;    // controls how entropy maps to temperature in dynamic temperature sampler
    int32_t     penalty_last_n        ;       // last n tokens to penalize (0 = disable penalty, -1 = context size)
    float       penalty_repeat        ;    // 1.0 = disabled
    float       penalty_freq          ;    // 0.0 = disabled
    float       penalty_present       ;    // 0.0 = disabled
    int32_t     mirostat              ;        // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    float       mirostat_tau          ;    // target entropy
    float       mirostat_eta          ;    // learning rate
    bool        penalize_nl           ;     // consider newlines as a repeatable token
} llama_sampling_params_spm;

char * get_tensor_name(struct ggml_tensor * t);
int check_tensor_name(struct ggml_tensor * t);

struct llama_sampling_context * init_sampling(  int32_t     n_prev,                 // number of previous tokens to remember
                                                int32_t     top_k,                 // <= 0 to use vocab size
                                                float       top_p,              // 1.0 = disabled
                                                float       min_p,              // 0.0 = disabled
                                                float       tfs_z,              // 1.0 = disabled
                                                float       typical_p,              // 1.0 = disabled
                                                float       temp,              // <= 0.0 to sample greedily, 0.0 to not output probabilities
                                                float       dynatemp_range,              // 0.0 = disabled
                                                float       dynatemp_exponent,              // controls how entropy maps to temperature in dynamic temperature sampler
                                                int32_t     penalty_last_n,                 // last n tokens to penalize (0 = disable penalty, -1 = context size)
                                                float       penalty_repeat,              // 1.0 = disabled
                                                float       penalty_freq,              // 0.0 = disabled
                                                float       penalty_present,              // 0.0 = disabled
                                                int32_t     mirostat,                 // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
                                                float       mirostat_tau,              // target entropy
                                                float       mirostat_eta,              // learning rate
                                                bool        penalize_nl,              // consider newlines as a repeatable token
                                                uint32_t    seed,
                                                const char * grammar_path);


llama_token spm_llama_sampling_sample(
        struct llama_sampling_context * ctx_sampling,
        struct llama_context * ctx_main,
        struct llama_context * ctx_cfg,
        int idx);

void spm_llama_sampling_accept(
        struct llama_sampling_context * ctx_sampling,
        struct llama_context * ctx_main,
        llama_token id,
        bool apply_grammar);

#ifdef __cplusplus
}
#endif
