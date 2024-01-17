//
//  GPT2.swift
//  Created by Guinmoon.

import Foundation
import llmfarm_core_cpp

public class GPT2: LLMBase {
    
    public var hardware_arch: String=""

    public override func llm_load_model(path: String = "", contextParams: ModelAndContextParams = .default, params:gpt_context_params ) throws -> Bool{        
        var context_params = gpt_context_default_params()
        context_params.n_ctx = contextParams.context
        //        params.n_parts = contextParams.parts
        context_params.seed = UInt32(contextParams.seed)
        context_params.f16_kv = contextParams.f16Kv
        context_params.logits_all = contextParams.logitsAll
        context_params.vocab_only = contextParams.vocabOnly
        context_params.use_mlock = contextParams.useMlock
        context_params.use_mmap = contextParams.useMMap
        var n_gpu_layers:Int32 = 0
        if contextParams.use_metal{
            n_gpu_layers = 1
        }
        self.hardware_arch = Get_Machine_Hardware_Name()// Disable Metal on intel Mac
        if self.hardware_arch=="x86_64"{
            n_gpu_layers = 0
        }
//TEMPORARY FIX
//        n_gpu_layers = 0
//        
        self.context = gpt2_init_from_file(path, context_params,n_gpu_layers)
        if self.context == nil {
            return false
        }
        self.contextParams.promptFormat = .None
        return true
    }
    
    deinit {
        gpt2_free(context)
    }
    
    public override func llm_eval(inputBatch:[ModelToken]) throws -> Bool{
        let res = gpt2_eval(context, inputBatch, Int32(inputBatch.count), nPast, contextParams.n_threads)
        if res != 0 {
            return false
        }
        return true
    }
    
}


