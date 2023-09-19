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
            params.use_mlock = false
        }
        self.hardware_arch = Get_Machine_Hardware_Name()// Disable Metal on intel Mac
        if self.hardware_arch=="x86_64"{
//            params.n_gpu_layers = 0
        }
        self.model = llama_load_model_from_file(path, params)
        if self.model == nil{
            return false
        }
        self.context = llama_new_context_with_model(model, params)
        if self.context == nil {
            return false
        }
        return true
    }
    
    deinit {
        llama_free(context)
        llama_free_model(model)
    }
    
    override func llm_n_vocab(_ ctx: OpaquePointer!) -> Int32{
        return llama_n_vocab(ctx)
    }
    
    override func llm_get_logits(_ ctx: OpaquePointer!) -> UnsafeMutablePointer<Float>?{
        return llama_get_logits(ctx);
    }

    public override func llm_eval(inputBatch:[ModelToken]) throws -> Bool{
        if llama_eval(context, inputBatch, Int32(inputBatch.count), nPast, contextParams.numberOfThreads) != 0 {
            throw ModelError.failedToEval
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

