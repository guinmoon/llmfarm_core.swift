//
//  LLaMA_FineTune.swift
//
//  Created by Guinmoon.
//

import Foundation
import llmfarm_core_cpp

var aaa = 1
var LLaMa_FineTune_obj_ptr:UnsafeMutableRawPointer? = nil

public class LLaMa_FineTune: FineTune {
    

    public override func finetune() throws{
        var args = ["progr_name", "--model-base", self.model_base, "--lora-out", self.lora_out, "--train-data", self.train_data,
            "--threads", String(self.threads), "--adam-iter", String(self.adam_iter), "--batch", String(self.batch), "--ctx", String(self.ctx)]        
        if self.use_checkpointing{
            args.append("--use-checkpointing")
        }
        print(args)
        var cargs = args.map { strdup($0) }        
        retain_new_self_ptr()
        let result = run_finetune(Int32(args.count), &cargs,
        { c_str in
            let LLaMa_FineTune_obj = Unmanaged<LLaMa_FineTune>.fromOpaque(LLaMa_FineTune_obj_ptr!).takeRetainedValue()
            if c_str != nil{
                let for_print = String(cString:c_str!)
                LLaMa_FineTune_obj.tune_log.append(for_print)
                print("\nProgress: \(for_print)")
            }
            LLaMa_FineTune_obj.retain_new_self_ptr()
            return LLaMa_FineTune_obj.cancel
        })            
        for ptr in cargs { free(ptr) }
    }
    
    private func retain_new_self_ptr(){
        LLaMa_FineTune_obj_ptr = Unmanaged.passRetained(self).toOpaque()
    }
    
}


