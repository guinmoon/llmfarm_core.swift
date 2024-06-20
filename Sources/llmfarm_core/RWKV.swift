//
//  RWKV.swift
//  Created by Guinmoon.

import Foundation
import llmfarm_core_cpp

public class RWKV: LLMBase {

    public var tokenizer_from_str:Tokenizer
    public var tokenizer_to_str:Tokenizer
    public var pointerToLogits:UnsafeMutablePointer<Float>? = nil
    public var pointerToStateIn:UnsafeMutablePointer<Float>? = nil
    public var pointerToStateOut:UnsafeMutablePointer<Float>? = nil
    
    
    
    public override init(path: String, contextParams: ModelAndContextParams = .default) throws {
        let core_resourses = get_core_bundle_path()
        let config_from_str = TokenizerConfig(
            vocab: URL(fileURLWithPath: core_resourses! + "/tokenizers/20B_tokenizer_vocab.json"),
            merges: URL(fileURLWithPath: core_resourses! + "/tokenizers/20B_tokenizer_merges.txt")
//            vocab: URL(fileURLWithPath: core_resourses! + "/tokenizers/MIDI_tokenizer_vocab.json"),
//            merges: URL(fileURLWithPath: core_resourses! + "/tokenizers/MIDI_tokenizer_merges.txt")
        )
        let config_to_str = TokenizerConfig(
            vocab: URL(fileURLWithPath: core_resourses! + "/tokenizers/20B_tokenizer_vocab.json"),
            merges: URL(fileURLWithPath: core_resourses! + "/tokenizers/20B_tokenizer_merges.txt")
//            vocab: URL(fileURLWithPath: core_resourses! + "/tokenizers/MIDI_tokenizer_vocab.json"),
//            merges: URL(fileURLWithPath: core_resourses! + "/tokenizers/MIDI_tokenizer_merges.txt")
        )
        self.tokenizer_from_str = Tokenizer(config: config_from_str)
        self.tokenizer_to_str = Tokenizer(config: config_to_str)
        try super.init(path: path, contextParams: contextParams)
        
    }
    
    public override func llm_load_model(path: String = "", contextParams: ModelAndContextParams = .default, params:gpt_context_params) throws -> Bool{
        self.context = rwkv_init_from_file(path, UInt32(contextParams.n_threads))
        if self.context == nil {
            return false
        }
//        rwkv_gpu_offload_layers(self.context,100)
//        self.promptFormat = .None
        
        return true
    }
    
    deinit {
        rwkv_free(context)
    }
    
    public override func llm_init_logits() throws -> Bool {
        do{
            let n_vocab = rwkv_get_logits_len(self.context);
            let n_state = rwkv_get_state_len(self.context);
            self.pointerToLogits = UnsafeMutablePointer<Float>.allocate(capacity: n_vocab)
            self.pointerToStateIn = UnsafeMutablePointer<Float>.allocate(capacity: n_state)
            self.pointerToStateOut = UnsafeMutablePointer<Float>.allocate(capacity: n_state)
            rwkv_init_state(self.context, pointerToStateIn);
            var inputs = [llm_token_bos(),llm_token_eos()]
            if try llm_eval(inputBatch: &inputs) == false {
                throw ModelError.failedToEval
            }
            return true
        }
        catch{
            print(error)
        }
        return false
    }
    
    public override func llm_eval(inputBatch: inout [ModelToken]) throws -> Bool{
//        for token in inputBatch{
//            rwkv_eval(self.context, UInt32(token), self.pointerToStateIn,self.pointerToStateIn, self.pointerToLogits)
//        }
        let token_chunks = inputBatch.chunked(into: 64)
        for chunk in token_chunks{
            rwkv_eval_sequence(self.context, chunk.map { UInt32($0) }, chunk.count, self.pointerToStateIn,self.pointerToStateIn, self.pointerToLogits)
        }
        return true
    }
    
    
    override func llm_n_vocab(_ ctx: OpaquePointer!) -> Int32{
        return Int32(rwkv_get_logits_len(self.context))
    }
    
    override func llm_get_logits(_ ctx: OpaquePointer!) -> UnsafeMutablePointer<Float>?{
        return self.pointerToLogits;
    }
   
    override func llm_get_n_ctx(ctx: OpaquePointer!) -> Int32{
        return 4096
    }
    
    public override func llm_token_to_str(outputToken:Int32) -> String? {
//        return tokenizer_from_str.decode(tokens: [outputToken])
        return tokenizer_to_str.decode(tokens: [outputToken])
    }
    
    public override func llm_tokenize(_ input: String, add_bos: Bool?, parse_special: Bool?) -> [ModelToken] {
        if input.count == 0 {
            return []
        }
        let tokens = tokenizer_from_str.encode(text: input)
        return tokens
    }
}

