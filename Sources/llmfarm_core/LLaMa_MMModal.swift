//
//  LLaMa.swift
//  Created by Guinmoon.

import Foundation
import llmfarm_core_cpp

var LLaMaMM_obj_ptr:UnsafeMutableRawPointer? = nil

public class LLaMaMModal: LLaMa {
    
    public var clip_ctx: OpaquePointer?
    public var image_embed: UnsafeMutablePointer<llava_image_embed>?
    
    public override func llm_load_model(path: String = "", contextParams: ModelAndContextParams = .default, params:gpt_context_params,
                                        model_load_progress_callback:((Float)  -> (Bool))?) throws -> Bool{
        var context_params = llama_context_default_params()
        var model_params = llama_model_default_params()
        let clip_path = "/Users/guinmoon/dev/alpaca_llama_etc/mobilevlm-3b-mmproj-model-f16.gguf"
        let image_path = "/Users/guinmoon/dev/alpaca_llama_etc/Angelina-Jolie-Rome-Film-Fest.jpg"
        

        context_params.n_ctx = UInt32(contextParams.context)
        context_params.seed = UInt32(contextParams.seed)
        context_params.n_threads = UInt32(contextParams.n_threads)
        context_params.logits_all = contextParams.logitsAll
//        context_params.n_batch = contextParams.
        model_params.vocab_only = contextParams.vocabOnly
        model_params.use_mlock = contextParams.useMlock
        model_params.use_mmap = contextParams.useMMap
//        A C function pointer can only be formed from a reference to a 'func' or a literal closure
        self.progressCallback = model_load_progress_callback
        self.retain_new_self_ptr()
        model_params.progress_callback = { progress,b in
                let LLaMa_obj = Unmanaged<LLaMaMModal>.fromOpaque(LLaMaMM_obj_ptr!).takeRetainedValue()
                LLaMa_obj.retain_new_self_ptr()
                if (LLaMa_obj.progressCallback != nil){
                    let res = LLaMa_obj.progressCallback!(progress)
                    return res
                }
                
                return true
        }

        if contextParams.use_metal{
            model_params.n_gpu_layers = 100
        }else{
            model_params.n_gpu_layers = 0
        }
        self.hardware_arch = Get_Machine_Hardware_Name()// Disable Metal on intel Mac
        if self.hardware_arch=="x86_64"{
            model_params.n_gpu_layers = 0
        }
        
        if contextParams.lora_adapters.count>0{
            model_params.use_mmap = false
        }
                        
        self.clip_ctx = clip_model_load(clip_path, 1,model_params.n_gpu_layers);

        llama_backend_init(false)

        self.model = llama_load_model_from_file(path, model_params)
        if self.model == nil{
            return false
        }
        
        for lora in contextParams.lora_adapters{            
            llama_model_apply_lora_from_file(model,lora.0,lora.1,nil,6);
        }
        
        self.context = llama_new_context_with_model(self.model, context_params)
        if self.context == nil {
            return false
        }
//        var tokens_tmp: [llama_token] = [Int32](repeating: 0, count: 100000)
//        var tokens_count:Int = 0
//        llama_load_session_file(self.context,"/Users/guinmoon/Library/Containers/com.guinmoon.LLMFarm/Data/Documents/models/dump_state.bin",tokens_tmp.mutPtr, 100000,&tokens_count)
//        self.session_tokens.append(contentsOf: tokens_tmp[0..<tokens_count])
//        try? llm_eval(inputBatch:self.session_tokens)
//        llama_load_state(self.context,"/Users/guinmoon/Library/Containers/com.guinmoon.LLMFarm/Data/Documents/models/dump_state_.bin")
        
        self.image_embed = llava_image_embed_make_with_filename(self.clip_ctx, Int32(context_params.n_threads), image_path);
        if image_embed == nil
        {
            print("image load error")
            return false
        }
        return true
    }
    
    private func retain_new_self_ptr(){
        LLaMaMM_obj_ptr = Unmanaged.passRetained(self).toOpaque()
    }
    
    public override func destroy_objects(){
        print("destroy LLaMa")
        if batch != nil{
            llama_batch_free(batch!)
        }
        llama_free(context)
        llama_free_model(model)
        llama_backend_free()
    }
    
    deinit {
//        llama_save_state(self.context,"/Users/guinmoon/Library/Containers/com.guinmoon.LLMFarm/Data/Documents/models/dump_state_.bin")
//        llama_save_session_file(self.context,"/Users/guinmoon/Library/Containers/com.guinmoon.LLMFarm/Data/Documents/models/dump_state.bin",self.session_tokens, self.session_tokens.count)
        self.destroy_objects()
        print("deinit LLaMa")
    }
    
   

    public override func llm_eval_clip() throws -> Bool{
        llava_eval_image_embed(self.context, self.image_embed, 512, &self.nPast);
        // if llama_eval(self.context, mutable_inputBatch.mutPtr, Int32(inputBatch.count), min(self.contextParams.context, self.nPast)) != 0 {
        //     return false
        // }
        return true
    }

}

