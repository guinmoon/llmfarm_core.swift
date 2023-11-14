//
//  LLaMa_dadbed9.swift
//  Created by Guinmoon.

import Foundation 
import llmfarm_core_cpp

public class LLaMa_dadbed9: LLMBase {

    public var hardware_arch: String=""
    
    public override func llm_load_model(path: String = "", contextParams: ModelAndContextParams = .default, params:gpt_context_params ) throws -> Bool{
        var params = llama_dadbed9_context_default_params()
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
        self.context = llama_dadbed9_init_from_file(path, params)
        if self.context == nil {
            return false
        }
        return true
    }
    
    deinit {
        llama_dadbed9_free(context)
    }
    
    override func llm_get_n_ctx(ctx: OpaquePointer!) -> Int32{
        return llama_dadbed9_n_ctx(ctx)
    }
    
    override func llm_n_vocab(_ ctx: OpaquePointer!) -> Int32{
        return llama_dadbed9_n_vocab(ctx)
    }
        
    override func llm_get_logits(_ ctx: OpaquePointer!) -> UnsafeMutablePointer<Float>?{
        return llama_dadbed9_get_logits(ctx);
    }

    public override func llm_eval(inputBatch:[ModelToken]) throws -> Bool{
        if llama_dadbed9_eval(context, inputBatch, Int32(inputBatch.count), min(self.contextParams.context, self.nPast), contextParams.n_threads) != 0 {
            throw ModelError.failedToEval
        }
        return true
    }
    
    public override func llm_token_to_str(outputToken:Int32) -> String? {
//        var cStringPtr: UnsafeMutablePointer<CChar>? = nil
//        var cStr_len: Int32 = 0;
//        llama_dadbed9_token_to_str(context, outputToken,cStringPtr,cStr_len)
//        if cStr_len>0{
//            return String(cString: cStringPtr!)
//        }
        if let cStr = llama_dadbed9_token_to_str(context, outputToken){
            return String(cString: cStr)
        }
        return nil
    }
    
    public override func llm_token_nl() -> ModelToken{
//        return llama_dadbed9_token_nl(self.context)
        return llama_dadbed9_token_nl()
    }

    public override func llm_token_bos() -> ModelToken{
//        return llama_dadbed9_token_bos(self.context)
        return llama_dadbed9_token_bos()
    }
    
    public override func llm_token_eos() -> ModelToken{
//        return llama_dadbed9_token_eos(self.context)
        return llama_dadbed9_token_eos()
    }
    

    
    
    public override func llm_tokenize(_ input: String) -> [ModelToken] {
        if input.count == 0 {
            return []
        }

        var embeddings: [llama_dadbed9_token] = Array<llama_dadbed9_token>(repeating: llama_dadbed9_token(), count: input.utf8.count)
        let n = llama_dadbed9_tokenize(context, input, &embeddings, Int32(input.utf8.count), self.contextParams.add_bos_token)
        if n<=0{
            return []
        }
        embeddings.removeSubrange(Int(n)..<embeddings.count)
        
        if self.contextParams.add_eos_token {
//            embeddings.append(llama_dadbed9_token_eos(self.context))
            embeddings.append(llama_dadbed9_token_eos())
        }
        
        return embeddings
    }
}

