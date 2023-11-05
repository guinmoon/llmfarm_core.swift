//
//  FineTune.swift
//  Created by Guinmoon.
//

import Foundation
import llmfarm_core_cpp

public class FineTune {
        
    public var model_base: String = ""
    public var lora_out: String = ""
    public var train_data: String = ""
    public var threads:Int32 = Int32(ProcessInfo.processInfo.processorCount)/2
    public var adam_iter:Int32 = 1
    public var batch:Int32 = 512
    public var ctx:Int32 = 512
    public var use_checkpointing:Bool = true
    public var tune_log: [String] = []
    public var cancel: Bool = false

    let args = ["progr", "--model-base", "/Users/guinmoon/dev/alpaca_llama_etc/openllama-3b-v2-q8_0.gguf", "--lora-out", "/Users/guinmoon/dev/alpaca_llama_etc/lora-open-llama-3b-v2-q8_0-shakespeare-LLMFarm.bin", "--train-data", "/Users/guinmoon/dev/alpaca_llama_etc/pdf/shakespeare.txt",
            "--threads", "12", "--adam-iter", "30", "--batch", "4", "--ctx", "64", "--use-checkpointing"]
    
    public  init(_ model_base: String, _ lora_out: String,_ train_data:String, 
                    threads: Int32 = Int32(ProcessInfo.processInfo.processorCount)/2, 
                    adam_iter:Int32 = 30,batch:Int32 = 4, ctx:Int32 = 64,
                    use_checkpointing:Bool = true ) {  
        self.model_base = model_base
        self.lora_out = lora_out
        self.train_data = train_data
        self.threads = threads
        self.adam_iter = adam_iter
        self.batch = batch
        self.ctx = ctx
        self.use_checkpointing = use_checkpointing
    }
    
    public func finetune() throws{
        
    }
    
    deinit {
        
    }
    
    
}


