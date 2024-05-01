//
//  LLaMa.swift
//  Created by Guinmoon.

import Foundation
import llmfarm_core_cpp

// var LLaMa_obj_ptr:UnsafeMutableRawPointer? = nil
var LLaMa_obj:LLaMa? = nil
//var LLaMa_ptr:UnsafeMutablePointer? = nil

public class LLaMa: LLMBase {
    
    public var model: OpaquePointer?
    public var ctx_sampling: OpaquePointer?
    public var batch: llama_batch?
    public var hardware_arch: String=""
    public var temporary_invalid_cchars: [CChar]  = []
    public var progressCallback: ((Float)  -> (Bool))? = nil    
//    public var sparams: llama_sampling_params_spm
    
    //  int32_t     n_prev                = 64;       // number of previous tokens to remember
    // int32_t     n_probs               = 0;        // if greater than 0, output the probabilities of top n_probs tokens.
    // int32_t     top_k                 = 40;       // <= 0 to use vocab size
    // float       top_p                 = 0.95f;    // 1.0 = disabled
    // float       min_p                 = 0.05f;    // 0.0 = disabled
    // float       tfs_z                 = 1.00f;    // 1.0 = disabled
    // float       typical_p             = 1.00f;    // 1.0 = disabled
    // float       temp                  = 0.80f;    // <= 0.0 to sample greedily, 0.0 to not output probabilities
    // float       dynatemp_range        = 0.00f;    // 0.0 = disabled
    // float       dynatemp_exponent     = 1.00f;    // controls how entropy maps to temperature in dynamic temperature sampler
    // int32_t     penalty_last_n        = 64;       // last n tokens to penalize (0 = disable penalty, -1 = context size)
    // float       penalty_repeat        = 1.10f;    // 1.0 = disabled
    // float       penalty_freq          = 0.00f;    // 0.0 = disabled
    // float       penalty_present       = 0.00f;    // 0.0 = disabled
    // int32_t     mirostat              = 0;        // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    // float       mirostat_tau          = 5.00f;    // target entropy
    // float       mirostat_eta          = 0.10f;    // learning rate
    // bool        penalize_nl           = true;  

//    public func init_sampling_param(){
//        sparams.n_prev = sampleParams.repeat_last_n
//        sparams.n_probs = 0;
//        sparams.top_k = sampleParams.top_k;
//        sparams.top_p = sampleParams.top_p;
//        sparams.min_p = sampleParams.min_p;
//        sparams.tfs_z = sampleParams.tfs_z;
//        sparams.typical_p = sampleParams.typical_p;
//        sparams.temp = sampleParams.temp;
//        sparams.dynatemp_range = 0;
//        sparams.dynatemp_exponent = 1;
//        sparams.penalty_last_n = sampleParams.repeat_last_n;
//        sparams.penalty_repeat = sampleParams.repeat_penalty;
//        sparams.penalty_freq = sampleParams.penalty_freq;
//        sparams.penalty_present = sampleParams.penalty_present;
//        sparams.mirostat = sampleParams.mirostat;
//        sparams.mirostat_tau = sampleParams.mirostat_tau;
//        sparams.mirostat_eta = sampleParams.mirostat_eta;
//        sparams.penalize_nl = sampleParams.penalize_nl;
//    }

    public override func llm_load_model(path: String = "", contextParams: ModelAndContextParams = .default, params:gpt_context_params,
                                        model_load_progress_callback:((Float)  -> (Bool))?) throws -> Bool{
        var context_params = llama_context_default_params()
        var model_params = llama_model_default_params()
//        init_sampling_param()
//        self.ctx_sampling = llama_sampling_init(sparams);
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
            //                let LLaMa_obj = Unmanaged<LLaMa>.fromOpaque(LLaMa_obj_ptr!).takeRetainedValue()
            //                let LLaMa_ptr = Unmanaged<LLaMa>.fromOpaque(LLaMa_obj!).takeRetainedValue()
//            LLaMa_obj?.retain_new_self_ptr()
            if (LLaMa_obj?.progressCallback != nil){
                let res = LLaMa_obj?.progressCallback!(progress)
                return res ?? false
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
        
#if targetEnvironment(simulator)
        model_params.n_gpu_layers = 0
        print("Running on simulator, force use n_gpu_layers = 0")
#endif
        
        if contextParams.lora_adapters.count>0{
            model_params.use_mmap = false
        }
        
        llama_backend_init()
        
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
        if !load_clip_model(){
            return false
        }
        self.batch = llama_batch_init(sampleParams.n_batch, 0, 1)
        return true
    }
    
    public func load_clip_model() -> Bool{
        return true
    }
    
    private func retain_new_self_ptr(){
        LLaMa_obj = Unmanaged<LLaMa>.fromOpaque(Unmanaged.passRetained(self).toOpaque()).takeRetainedValue()
        //        LLaMa_obj_ptr = Unmanaged.passRetained(self).toOpaque()
        //        LLaMa_obj_ptr = UnsafeMutablePointer(OpaquePointer(bitPattern: Unmanaged.passUnretained(self)))
        // LLaMa_ptr = Unmanaged<LLaMa_MModal>.fromOpaque(LLaMaMM_obj_ptr!).takeRetainedValue()
    }
    
    public override func destroy_objects(){
        print("destroy LLaMa")
        if batch != nil{
            llama_batch_free(batch!)
        }
        if context != nil{
            llama_free(context)
        }
        if model != nil{
            llama_free_model(model)
        }
        self.destroy_clip()
//        llama_backend_free()
    }
    
    public func destroy_clip(){
        
    }
    
    deinit {
        //        llama_save_state(self.context,"/Users/guinmoon/Library/Containers/com.guinmoon.LLMFarm/Data/Documents/models/dump_state_.bin")
        //        llama_save_session_file(self.context,"/Users/guinmoon/Library/Containers/com.guinmoon.LLMFarm/Data/Documents/models/dump_state.bin",self.session_tokens, self.session_tokens.count)       
        print("deinit LLaMa")
        self.destroy_objects()
        print("LLaMa deinited")
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
    
//    public func llm_eval_old(inputBatch:[ModelToken]) throws -> Bool{
//        var mutable_inputBatch = inputBatch
//        if llama_eval(self.context, mutable_inputBatch.mutPtr, Int32(inputBatch.count), min(self.contextParams.context, self.nPast)) != 0 {
//            return false
//        }
//        return true
//    }
    
     public override func llm_eval(inputBatch:[ModelToken]) throws -> Bool{
//       var mutable_inputBatch = inputBatch
//       if llama_eval(self.context, mutable_inputBatch.mutPtr, Int32(inputBatch.count), min(self.contextParams.context, self.nPast)) != 0 {
//           return false
//       }
        if self.nPast==0{
        // if self.nPast==0 || inputBatch.count>1{
             completion_init(tokens_list:inputBatch)
         }else{
             llama_batch_clear(&batch!)
             for i1 in 0..<inputBatch.count {
                 let i = Int(i1)
                 llama_batch_add(&batch!, inputBatch[i], Int32(i)+self.nPast, [0], true)
             }
 //            batch!.logits[Int(batch!.n_tokens) - 1] = 1
             if llama_decode(context, batch!) != 0 {
                 print("failed to evaluate llama!")
                 return false
             }
         }
        return true
    }
    
    func completion_init(tokens_list: [ModelToken]) {
//        print("attempting to complete \"\(text)\"")

        // tokens_list = tokenize(text: text, add_bos: true)
        temporary_invalid_cchars = []

//        let n_ctx = llama_n_ctx(context)
//        let n_kv_req = tokens_list.count + (Int(n_len) - tokens_list.count)
//
//        print("\n n_len = \(n_len), n_ctx = \(n_ctx), n_kv_req = \(n_kv_req)")
//
//        if n_kv_req > n_ctx {
//            print("error: n_kv_req > n_ctx, the required KV cache size is not big enough")
//        }

//        for id in tokens_list {
//            print(String(cString: token_to_piece(token: id) + [0]))
//        }

        llama_batch_clear(&batch!)

        for i1 in 0..<tokens_list.count {
            let i = Int(i1)
            llama_batch_add(&batch!, tokens_list[i], Int32(i), [0], false)
        }
        batch!.logits[Int(batch!.n_tokens) - 1] = 1

        if llama_decode(context, batch!) != 0 {
            print("llama_decode() failed")
        }

//        n_cur = batch.n_tokens
    }
        

    func sample_wip(){
        var new_token_id: llama_token = 0

       
    }

    override func llm_init_logits() throws -> Bool {
        return true
    }
    
    func llama_batch_clear(_ batch: inout llama_batch) {
     batch.n_tokens = 0
    }

    func llama_batch_add(_ batch: inout llama_batch, _ id: llama_token, _ pos: llama_pos, _ seq_ids: [llama_seq_id], _ logits: Bool) {
        batch.token   [Int(batch.n_tokens)] = id
        batch.pos     [Int(batch.n_tokens)] = pos
        batch.n_seq_id[Int(batch.n_tokens)] = Int32(seq_ids.count)
        for i in 0..<seq_ids.count {
            batch.seq_id[Int(batch.n_tokens)]![Int(i)] = seq_ids[i]
        }
        batch.logits  [Int(batch.n_tokens)] = logits ? 1 : 0

        batch.n_tokens += 1
    }
    
    func model_info() -> String {
        let result = UnsafeMutablePointer<Int8>.allocate(capacity: 256)
        result.initialize(repeating: Int8(0), count: 256)
        defer {
            result.deallocate()
        }

        // TODO: this is probably very stupid way to get the string from C

        let nChars = llama_model_desc(model, result, 256)
        let bufferPointer = UnsafeBufferPointer(start: result, count: Int(nChars))

        var SwiftString = ""
        for char in bufferPointer {
            SwiftString.append(Character(UnicodeScalar(UInt8(char))))
        }

        return SwiftString
    }

    

    private func token_to_piece(token: Int32) -> [CChar] {
        let result = UnsafeMutablePointer<Int8>.allocate(capacity: 8)
        result.initialize(repeating: Int8(0), count: 8)
        defer {
            result.deallocate()
        }
//        llama_token_to_piece(const struct llama_model * model, llama_token token, char * buf, int32_t length, bool special)
        let nTokens = llama_token_to_piece(model, token, result, 8,self.contextParams.parse_special_tokens)
        
        if nTokens < 0 {
            let newResult = UnsafeMutablePointer<Int8>.allocate(capacity: Int(-nTokens))
            newResult.initialize(repeating: Int8(0), count: Int(-nTokens))
            defer {
                newResult.deallocate()
            }
            let nNewTokens = llama_token_to_piece(model, token, newResult, -nTokens,self.contextParams.parse_special_tokens)
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
            embeddings.append(llm_token_eos())
        }
        
        return embeddings
    }
}

