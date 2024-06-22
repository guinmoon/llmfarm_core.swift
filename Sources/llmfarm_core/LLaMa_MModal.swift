//
//  LLaMa.swift
//  Created by Guinmoon.

import Foundation
import llmfarm_core_cpp

var LLaMaMM_obj:LLaMa_MModal? = nil

public class LLaMa_MModal: LLaMa {
    
    public var clip_ctx: OpaquePointer?
    public var image_embed: UnsafeMutablePointer<llava_image_embed>?
    
    
    public override func load_clip_model() -> Bool{
        if self.clip_ctx != nil {
            return true
        }
        if contextParams.clip_model == nil {
            return false
        }
        #if os(iOS)
        self.clip_ctx = clip_model_load(contextParams.clip_model, 1, 0 );
//        self.clip_ctx = clip_model_load(contextParams.clip_model, 1,contextParams.use_metal ? 1: 0);
        #else
        self.clip_ctx = clip_model_load(contextParams.clip_model, 1,contextParams.use_metal ? 1: 0);
        #endif
        return true
    }
    
    public override func deinit_clip_model(){
        clip_free(clip_ctx);
        self.clip_ctx = nil
    }
    
    public override func make_image_embed(_ image_path:String) -> Bool{
        self.image_embed = llava_image_embed_make_with_filename(self.clip_ctx, Int32(self.contextParams.n_threads), image_path);
        return true
    }
    
    private func retain_new_self_ptr(){
        LLaMaMM_obj = Unmanaged<LLaMa_MModal>.fromOpaque(Unmanaged.passRetained(self).toOpaque()).takeRetainedValue()
    }
    
    public override func destroy_clip(){
        if self.clip_ctx != nil{
            clip_free(clip_ctx);
        }
    }
    
    public override func llm_eval_clip() throws -> Bool{
        llava_eval_image_embed(self.context, self.image_embed, sampleParams.n_batch, &self.nPast);
        return true
    }
    
}

