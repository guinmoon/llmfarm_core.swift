//
//  LLaMa.swift
//  Mia
//
//  Created by Byron Everson on 4/15/23.
//

import Foundation
import llmfarm_core_cpp

public class LLaMa: LLMBase {
    
    public var model: OpaquePointer?
    public var hardware_arch: String=""
    
    public override func llm_load_model(path: String = "", contextParams: ModelContextParams = .default, params:gpt_context_params ) throws -> Bool{
        var params = llama_context_default_params()
        params.n_ctx = contextParams.context
        //        params.n_parts = contextParams.parts
        params.seed = UInt32(contextParams.seed)
        params.f16_kv = contextParams.f16Kv
        params.logits_all = contextParams.logitsAll
        params.vocab_only = contextParams.vocabOnly
        params.use_mlock = contextParams.useMlock
        params.embedding = contextParams.embedding
        if contextParams.use_metal{
            params.n_gpu_layers = 1
        }
        if !contextParams.useMMap{
            params.use_mmap = false
        }
        if contextParams.useMlock{
            params.use_mlock = true
        }
        self.hardware_arch = Get_Machine_Hardware_Name()// Disable Metal on intel Mac
        if self.hardware_arch=="x86_64"{
            params.n_gpu_layers = 0
        }
        params.use_mmap = false
        
        var exception = tryBlock {
            self.model = llama_load_model_from_file(path, params)
        }
        if self.model == nil{
            return false
        }
        exception = tryBlock {
            self.context = llama_new_context_with_model(self.model, params)
        }
        if self.context == nil {
            return false
        }
//        var tokens_tmp: [llama_token] = [Int32](repeating: 0, count: 100000)
//        var tokens_count:Int = 0
//        llama_load_session_file(self.context,"/Users/guinmoon/Library/Containers/com.guinmoon.LLMFarm/Data/Documents/models/dump_state.bin",tokens_tmp.mutPtr, 100000,&tokens_count)
//        self.session_tokens.append(contentsOf: tokens_tmp[0..<tokens_count])
//        try? llm_eval(inputBatch:self.session_tokens)
//        llama_load_state(self.context,"/Users/guinmoon/Library/Containers/com.guinmoon.LLMFarm/Data/Documents/models/dump_state_.bin")
//        llama_model_apply_lora_from_file(model,"/Users/guinmoon/dev/alpaca_llama_etc/lora-open-llama-3b-v2-q8_0-my_finetune-LATEST.bin",nil,6);
        return true
    }
    
    deinit {
//        llama_save_state(self.context,"/Users/guinmoon/Library/Containers/com.guinmoon.LLMFarm/Data/Documents/models/dump_state_.bin")
//        llama_save_session_file(self.context,"/Users/guinmoon/Library/Containers/com.guinmoon.LLMFarm/Data/Documents/models/dump_state.bin",self.session_tokens, self.session_tokens.count)
        llama_free(context)
        llama_free_model(model)
    }
    
    override func llm_get_n_ctx(ctx: OpaquePointer!) -> Int32{
        return llama_n_ctx(ctx)
    }
    
    override func llm_n_vocab(_ ctx: OpaquePointer!) -> Int32{
        return llama_n_vocab(ctx)
    }
    
    override func llm_get_logits(_ ctx: OpaquePointer!) -> UnsafeMutablePointer<Float>?{
        return llama_get_logits(ctx);
    }

    public override func llm_eval(inputBatch:[ModelToken]) throws -> Bool{
        var eval_res:Int32 = 1        
        var exception = tryBlock {
            eval_res = llama_eval(self.context, inputBatch, Int32(inputBatch.count), min(self.contextParams.context, self.nPast), self.contextParams.numberOfThreads)
        }
        if exception != nil{
            return false
        }
        if eval_res != 0 {
            return false
        }
        return true
    }
    
    public override func llm_token_to_str(outputToken:Int32) -> String? {
        if let cStr = llama_token_to_str(context, outputToken){
//            print(String(cString: cStr))
            return String(cString: cStr)
        }
        return nil
    }
    
    public override func llm_token_nl() -> ModelToken{
        return llama_token_nl(self.context)
    }

    public override func llm_token_bos() -> ModelToken{
       return llama_token_bos(self.context)
    }
    
    public override func llm_token_eos() -> ModelToken{
        return llama_token_eos(self.context)
    }
    

    
    
    public override func llm_tokenize(_ input: String, bos: Bool = true, eos: Bool = false) -> [ModelToken] {
        if input.count == 0 {
            return []
        }

//        llama_tokenize(
//                struct llama_context * ctx,
//                          const char * text,
//                                 int   text_len,
//                         llama_token * tokens,
//                                 int   n_max_tokens,
//                                bool   add_bos)
        let n_tokens = Int32(input.utf8.count) + (bos == true ? 1 : 0)
        var embeddings: [llama_token] = Array<llama_token>(repeating: llama_token(), count: input.utf8.count)
        let n = llama_tokenize(context, input, Int32(input.utf8.count), &embeddings, n_tokens, bos)
        assert(n >= 0)
        embeddings.removeSubrange(Int(n)..<embeddings.count)
        
        if eos {
            embeddings.append(llama_token_eos(self.context))
        }
        
        return embeddings
    }
}

