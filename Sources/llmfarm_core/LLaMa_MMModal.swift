//
//  LLaMa.swift
//  Created by Guinmoon.

import Foundation
import llmfarm_core_cpp

var LLaMaM_obj_ptr:UnsafeMutableRawPointer? = nil

public class LLaMaMModal: LLMBase {
    
    public var model: OpaquePointer?
    private var batch: llama_batch?
    public var hardware_arch: String=""
    private var temporary_invalid_cchars: [CChar]  = []
    public var progressCallback: ((Float)  -> (Bool))? = nil
    
    public override func llm_load_model(path: String = "", contextParams: ModelAndContextParams = .default, params:gpt_context_params,
                                        model_load_progress_callback:((Float)  -> (Bool))?) throws -> Bool{
        
        
        return true
    }
    
    private func retain_new_self_ptr(){
        LLaMa_obj_ptr = Unmanaged.passRetained(self).toOpaque()
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
    
    override func llm_get_n_ctx(ctx: OpaquePointer!) -> Int32{
        return Int32(llama_n_ctx(self.context))
    }
    
    override func llm_n_vocab(_ ctx: OpaquePointer!) -> Int32{
        return llama_n_vocab(self.model)
    }
    
    override func llm_get_logits(_ ctx: OpaquePointer!) -> UnsafeMutablePointer<Float>?{
        return llama_get_logits(self.context);
    }

    public override func llm_eval(inputBatch:[ModelToken]) throws -> Bool{
        var mutable_inputBatch = inputBatch
        if llama_eval(self.context, mutable_inputBatch.mutPtr, Int32(inputBatch.count), min(self.contextParams.context, self.nPast)) != 0 {
            return false
        }
        return true
    }
    
    
    private func token_to_piece(token: Int32) -> [CChar] {
        let result = UnsafeMutablePointer<Int8>.allocate(capacity: 8)
        result.initialize(repeating: Int8(0), count: 8)
        defer {
            result.deallocate()
        }
        let nTokens = llama_token_to_piece(model, token, result, 8)
        
        if nTokens < 0 {
            let newResult = UnsafeMutablePointer<Int8>.allocate(capacity: Int(-nTokens))
            newResult.initialize(repeating: Int8(0), count: Int(-nTokens))
            defer {
                newResult.deallocate()
            }
            let nNewTokens = llama_token_to_piece(model, token, newResult, -nTokens)
            let bufferPointer = UnsafeBufferPointer(start: newResult, count: Int(nNewTokens))
            return Array(bufferPointer)
        } else {
            let bufferPointer = UnsafeBufferPointer(start: result, count: Int(nTokens))
            return Array(bufferPointer)
        }
    }
    
    public override func llm_token_to_str(outputToken:Int32) -> String? {
//        if let cStr = llama_token_to_str(context, outputToken){
//            return String(cString: cStr)
//        }
        //        return nil
        let new_token_cchars = token_to_piece(token: outputToken)
        temporary_invalid_cchars.append(contentsOf: new_token_cchars)
        let new_token_str: String
        if let string = String(validatingUTF8: temporary_invalid_cchars + [0]) {
            temporary_invalid_cchars.removeAll()
            new_token_str = string
        } else if (0 ..< temporary_invalid_cchars.count).contains(where: {$0 != 0 && String(validatingUTF8: Array(temporary_invalid_cchars.suffix($0)) + [0]) != nil}) {
            // in this case, at least the suffix of the temporary_invalid_cchars can be interpreted as UTF8 string
            let string = String(cString: temporary_invalid_cchars + [0])
            temporary_invalid_cchars.removeAll()
            new_token_str = string
        } else {
            new_token_str = ""
        }
        return new_token_str
    }
    
    public override func llm_token_nl() -> ModelToken{
        return llama_token_nl(self.model)
    }

    public override func llm_token_bos() -> ModelToken{
        return llama_token_bos(self.model)
    }
    
    public override func llm_token_eos() -> ModelToken{
        return llama_token_eos(self.model)
    }
    

    
    
    public override func llm_tokenize(_ input: String) -> [ModelToken] {
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
        let n_tokens = Int32(input.utf8.count) + (self.contextParams.add_bos_token == true ? 1 : 0)
        var embeddings: [llama_token] = Array<llama_token>(repeating: llama_token(), count: input.utf8.count)
        let n = llama_tokenize(self.model, input, Int32(input.utf8.count), &embeddings, n_tokens, self.contextParams.add_bos_token, self.contextParams.parse_special_tokens)
        if n<=0{
            return []
        }
        if Int(n) <= embeddings.count {
            embeddings.removeSubrange(Int(n)..<embeddings.count)
        }
        
        if self.contextParams.add_eos_token {
            embeddings.append(llama_token_eos(self.context))
        }
        
        return embeddings
    }
}

