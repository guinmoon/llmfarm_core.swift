//
//  Model.swift
//  Created by Guinmoon.

import Foundation
import llmfarm_core_cpp

public enum ModelInference {
    case LLama_bin
    case LLama_gguf
    case LLama_mm
    case GPTNeox
    case GPTNeox_gguf
    case GPT2
    case Replit
    case Starcoder
    case Starcoder_gguf
    case RWKV
}

public class AI {
    
    var aiQueue = DispatchQueue(label: "LLMFarm-Main", qos: .userInitiated, attributes: .concurrent, autoreleaseFrequency: .inherit, target: nil)
    
    //var model: Model!
    public var model: LLMBase?
    public var modelPath: String
    public var modelName: String
    public var chatName: String
    
    
    
    public var flagExit = false
    private(set) var flagResponding = false
    
    public init(_modelPath: String,_chatName: String) {
        self.modelPath = _modelPath
        self.modelName = NSURL(fileURLWithPath: _modelPath).lastPathComponent!
        self.chatName = _chatName
    }
    
    deinit {
        // self.model?.destroy_objects()
        self.model = nil
    }
    
    
    public func initModel(_ inference: ModelInference,contextParams: ModelAndContextParams = .default) {
        self.model = nil
        switch inference {
        case .LLama_bin:
            self.model = try? LLaMa_dadbed9(path: self.modelPath, contextParams: contextParams)
        case .LLama_gguf:
            self.model = try? LLaMa(path: self.modelPath, contextParams: contextParams)
        case .LLama_mm:
            self.model = try? LLaMa_MModal(path: self.modelPath, contextParams: contextParams)
        case .GPTNeox:
            self.model = try? GPTNeoX(path: self.modelPath, contextParams: contextParams)
        case .GPTNeox_gguf:
            self.model = try? LLaMa(path: self.modelPath, contextParams: contextParams)
        case .GPT2:
            self.model = try? GPT2(path: self.modelPath, contextParams: contextParams)
        case .Replit:
            self.model = try? Replit(path: self.modelPath, contextParams: contextParams)
        case .Starcoder:
            self.model = try? Starcoder(path: self.modelPath, contextParams: contextParams)
        case .Starcoder_gguf:
            self.model = try? LLaMa(path: self.modelPath, contextParams: contextParams)
        case .RWKV:
            self.model = try? RWKV(path: self.modelPath, contextParams: contextParams)
        }
    
    }


    public func loadModel_sync() throws {
         do{
            try ExceptionCather.catchException {
               try? self.model?.load_model()
            }
        }
        catch{
            print(error)
            throw error
        }
    }

    
    public func loadModel() {
        aiQueue.async {
            do{
                try ExceptionCather.catchException {
                    try? self.model?.load_model()
                }
            }
            catch{
                print(error)
                DispatchQueue.main.async {
                    if self.model?.modelLoadCompleteCallback != nil{
                        self.model?.modelLoadCompleteCallback!("[Error] \(error)")
                        self.model?.modelLoadCompleteCallback!("[Error] \(error)")
                    }
                }
            }
            DispatchQueue.main.async {
                _ = self.model?.modelLoadProgressCallback?(1.0)
                self.model?.modelLoadCompleteCallback?("[Done]")
            }
        }
    }


    public func conversation(_ input: String,  _ tokenCallback: ((String, Double)  -> ())?, _ completion: ((String) -> ())?,system_prompt:String?, img_path: String? = nil) {
        flagResponding = true
        aiQueue.async {
            guard let completion = completion else { return }
            
            
            if self.model == nil{
                DispatchQueue.main.async {
                    self.flagResponding = false
                    completion("[Error] Load Model")
                }
                return
            }
            
            // Model output
            var output:String? = ""
            do{
                try ExceptionCather.catchException {
                    output = try? self.model?.predict(input, { str, time in
                        if self.flagExit {
                            // Reset flag
                            self.flagExit = false
                            // Alert model of exit flag
                            return true
                        }
                        DispatchQueue.main.async {
                            tokenCallback?(str, time)
                        }
                        return false
                    },system_prompt:system_prompt,img_path:img_path)
                }
            }catch{
                print(error)
                DispatchQueue.main.async {
                    self.flagResponding = false
                    completion("[Error] \(error)")
                }
            }
            DispatchQueue.main.async {
                self.flagResponding = false
                completion(output ?? "[Error]")
            }
            
        }
    }
}


private typealias _ModelProgressCallback = (_ progress: Float, _ userData: UnsafeMutableRawPointer?) -> Void

public typealias ModelProgressCallback = (_ progress: Float, _ model: LLMBase) -> Void

func get_path_by_short_name(_ model_name:String?, dest:String = "lora_adapters") -> String? {
    //#if os(iOS) || os(watchOS) || os(tvOS)
    if model_name == nil{
        return nil
    }
    do {
        let fileManager = FileManager.default
        let documentsPath = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first
        let destinationURL = documentsPath!.appendingPathComponent(dest)
        try fileManager.createDirectory (at: destinationURL, withIntermediateDirectories: true, attributes: nil)
        let path = destinationURL.appendingPathComponent(model_name!).path
        if fileManager.fileExists(atPath: path){
            return path
        }else{
            return nil
        }
        
    } catch {
        print(error)
    }
    return nil
}

public func get_model_sample_param_by_config(_ model_config:Dictionary<String, AnyObject>) -> ModelSampleParams{
    var tmp_param = ModelSampleParams.default
    if (model_config["n_batch"] != nil){
        tmp_param.n_batch = model_config["n_batch"] as! Int32
    }
    if (model_config["temp"] != nil){
        tmp_param.temp = Float(model_config["temp"] as! Double)
    }
    if (model_config["top_k"] != nil){
        tmp_param.top_k = model_config["top_k"] as! Int32
    }
    if (model_config["top_p"] != nil){
        tmp_param.top_p = Float(model_config["top_p"] as! Double)
    }
    if (model_config["tfs_z"] != nil){
        tmp_param.tfs_z = Float(model_config["tfs_z"] as! Double)
    }
    if (model_config["typical_p"] != nil){
        tmp_param.typical_p = Float(model_config["typical_p"] as! Double)
    }
    if (model_config["repeat_penalty"] != nil){
        tmp_param.repeat_penalty = Float(model_config["repeat_penalty"] as! Double)
    }
    if (model_config["repeat_last_n"] != nil){
        tmp_param.repeat_last_n = model_config["repeat_last_n"] as! Int32
    }
    if (model_config["frequence_penalty"] != nil){
        tmp_param.frequence_penalty = Float(model_config["frequence_penalty"] as! Double)
    }
    if (model_config["presence_penalty"] != nil){
        tmp_param.presence_penalty = Float(model_config["presence_penalty"] as! Double)
    }
    if (model_config["mirostat"] != nil){
        tmp_param.mirostat = model_config["mirostat"] as! Int32
    }
    if (model_config["mirostat_tau"] != nil){
        tmp_param.mirostat_tau = Float(model_config["mirostat_tau"] as! Double)
    }
    if (model_config["mirostat_eta"] != nil){
        tmp_param.mirostat_eta = Float(model_config["mirostat_tau"] as! Double)
    }
    
    return tmp_param
}

extension String {
    func removingLeadingSpaces() -> String {
        guard let index = firstIndex(where: { !CharacterSet(charactersIn: String($0)).isSubset(of: .whitespaces) }) else {
            return self
        }
        return String(self[index...])
    }
}

public func get_system_prompt(_ prompt_format_in: String) -> (String,String){
    var prompt_format = prompt_format_in
    var system_prompt = ""
    let pattern = "[system]("
    if prompt_format_in.contains(pattern) {
        let beg_i = prompt_format_in.distance(of:pattern)! + pattern.count
        var end_i = -1
        for i in (beg_i...prompt_format_in.count-1){
            if prompt_format_in[i..<i+1] == ")"{
                end_i = i
                break
            }
        }
        if end_i != -1{
            system_prompt = prompt_format_in[beg_i...end_i-1]
        }
        prompt_format = String(prompt_format_in[end_i+2..<prompt_format_in.count]).removingLeadingSpaces()
    }
    return (prompt_format,system_prompt)
}



public func get_model_context_param_by_config(_ model_config:Dictionary<String, AnyObject>) -> ModelAndContextParams{
    var tmp_param = ModelAndContextParams.default
    if (model_config["context"] != nil){
        tmp_param.context = model_config["context"] as! Int32
    }
    if (model_config["numberOfThreads"] != nil && model_config["numberOfThreads"] as! Int32 != 0){
        tmp_param.n_threads = model_config["numberOfThreads"] as! Int32
    }
    if (model_config["n_predict"] != nil && model_config["n_predict"] as! Int32 != 0){
        tmp_param.n_predict = model_config["n_predict"] as! Int32
    }
    
    if model_config["lora_adapters"] != nil{
        let tmp_adapters = model_config["lora_adapters"]! as? [Dictionary<String, Any>]
        if tmp_adapters != nil{
            for adapter in tmp_adapters!{
                var adapter_file: String? = nil
                var scale: Float? = nil
                if adapter["adapter"] != nil{
                    adapter_file = adapter["adapter"]! as? String
                }
                if adapter["scale"] != nil{
                    scale = adapter["scale"]! as? Float
                }
                if adapter_file != nil && scale != nil{
                    let adapter_path = get_path_by_short_name(adapter_file!)
                    if adapter_path != nil{
                        tmp_param.lora_adapters.append((adapter_path!,scale!))
                    }
                }
            }
        }            
    }
    if model_config["clip_model"] != nil {
        tmp_param.clip_model = get_path_by_short_name(model_config["clip_model"]! as? String,dest:"models")
    }
    if (model_config["reverse_prompt"] != nil){
        let splited_revrse_prompt = String(model_config["reverse_prompt"]! as! String).components(separatedBy: [","])
        for word in splited_revrse_prompt{
            let trimed_word = word.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimed_word==""{
                continue
            }
            var exist = false
            for r_word in tmp_param.reverse_prompt{
                if r_word == trimed_word{
                    exist = true
                    break
                }
            }
            if !exist{
                tmp_param.reverse_prompt.append(trimed_word)
            }
        }
    }   
    
    if (model_config["skip_tokens"] != nil){
        tmp_param.skip_tokens_str = model_config["skip_tokens"]! as! String
    }

    if (model_config["prompt_format"] != nil && model_config["prompt_format"]! as! String != "auto"
            && model_config["prompt_format"]! as! String != "{prompt}"){
            tmp_param.custom_prompt_format = model_config["prompt_format"]! as! String
            (tmp_param.custom_prompt_format,tmp_param.system_prompt) = get_system_prompt(tmp_param.custom_prompt_format)
            tmp_param.promptFormat = .Custom
    }
    
    if (model_config["use_metal"] != nil){
        tmp_param.use_metal = model_config["use_metal"] as! Bool
    }
    if (model_config["mlock"] != nil){
        tmp_param.useMlock = model_config["mlock"] as! Bool
    }
    if (model_config["mmap"] != nil){
        tmp_param.useMMap = model_config["mmap"] as! Bool
    }
    if (model_config["flash_attn"] != nil){
        tmp_param.flash_attn = model_config["flash_attn"] as! Bool
    }
    if (model_config["add_bos_token"] != nil){
        tmp_param.add_bos_token = model_config["add_bos_token"] as! Bool
    }
    if (model_config["add_eos_token"] != nil){
        tmp_param.add_eos_token = model_config["add_eos_token"] as! Bool
    }
    if (model_config["parse_special_tokens"] != nil){
        tmp_param.parse_special_tokens = model_config["parse_special_tokens"] as! Bool
    }
    if (model_config["save_load_state"] != nil){
        tmp_param.save_load_state = model_config["save_load_state"] as! Bool
    }
    

    if (model_config["model"] as! String).hasSuffix(".gguf"){
            tmp_param.model_inference = ModelInference.LLama_gguf
    }else{
        if model_config["model_inference"] as! String == "llama"{
            tmp_param.model_inference = ModelInference.LLama_bin
        }
        if model_config["model_inference"] as! String == "gptneox" {
            tmp_param.model_inference = ModelInference.GPTNeox
        }
        if model_config["model_inference"] as! String == "rwkv" {
            tmp_param.model_inference = ModelInference.RWKV
        }
        if model_config["model_inference"] as! String == "gpt2" {
            tmp_param.model_inference = ModelInference.GPT2
        }
        if model_config["model_inference"] as! String == "replit" {
            tmp_param.model_inference = ModelInference.Replit
        }
        if model_config["model_inference"] as! String == "starcoder" {
            tmp_param.model_inference = ModelInference.Starcoder
        }
    }
    
    if tmp_param.clip_model != nil{
        tmp_param.model_inference = ModelInference.LLama_mm
    }
    
    return tmp_param
}

public struct ModelAndContextParams {
    public var model_inference = ModelInference.LLama_gguf

    public var context: Int32 = 512    // text context
    public var parts: Int32 = -1   // -1 for default
    public var seed: UInt32 = 0xFFFFFFFF      // RNG seed, 0 for random
    public var n_threads: Int32 = 1
    public var n_predict: Int32 = 0
    public var lora_adapters: [(String,Float)] = []
    public var state_dump_path: String = ""
    public var skip_tokens: [Int32] = []
    public var skip_tokens_str: String = ""

    public var promptFormat: ModelPromptStyle = .None
    public var custom_prompt_format = ""
    public var system_prompt = ""
    
    public var f16Kv = true         // use fp16 for KV cache
    public var logitsAll = false    // the llama_eval() call computes all logits, not just the last one
    public var vocabOnly = false    // only load the vocabulary, no weights
    public var useMlock = false     // force system to keep model in RAM
    public var useMMap = true     // if disabled dont use MMap file
    public var embedding = false    // embedding mode only
    public var processorsConunt  = Int32(ProcessInfo.processInfo.processorCount)
    public var use_metal = false
    public var grammar_path:String? = nil
    public var add_bos_token = true
    public var add_eos_token = false
    public var parse_special_tokens = true
    public var flash_attn = false
    public var save_load_state = true
    
    
    public var warm_prompt = "\n\n\n"

    public var reverse_prompt: [String] = []
    
    public var clip_model:String? = nil

    public static let `default` = ModelAndContextParams()
    
    public init(    context: Int32 = 2048 /*512*/,
                    parts: Int32 = -1, 
                    seed: UInt32 = 0xFFFFFFFF, 
                    numberOfThreads: Int32 = 0,
                    f16Kv: Bool = true, 
                    logitsAll: Bool = false, 
                    vocabOnly: Bool = false,
                    useMlock: Bool = false,
                    useMMap: Bool = true, 
                    embedding: Bool = false) {
        self.context = context
        self.parts = parts
        self.seed = seed
        // Set numberOfThreads to processorCount, processorCount is actually thread count of cpu
        self.n_threads = Int32(numberOfThreads) == Int32(0) ? processorsConunt : numberOfThreads
        //        self.numberOfThreads = processorsConunt
        self.f16Kv = f16Kv
        self.logitsAll = logitsAll
        self.vocabOnly = vocabOnly
        self.useMlock = useMlock
        self.useMMap = useMMap
        self.embedding = embedding
    }
}

public struct ModelSampleParams {
    public var n_batch: Int32
    public var temp: Float
    public var top_k: Int32
    public var top_p: Float
    public var min_p: Float
    public var tfs_z: Float
    public var typical_p: Float
    public var repeat_penalty: Float
    public var repeat_last_n: Int32
    public var frequence_penalty: Float
    public var presence_penalty: Float
    public var mirostat: Int32
    public var mirostat_tau: Float
    public var mirostat_eta: Float
    public var penalize_nl: Bool
    
    public static let `default` = ModelSampleParams(
        n_batch: 512,
        temp: 0.9,
        top_k: 40,
        top_p: 0.95,
        min_p: 0,
        tfs_z: 1.0,
        typical_p: 1.0,
        repeat_penalty: 1.1,
        repeat_last_n: 64,
        frequence_penalty: 0.0,
        presence_penalty: 0.0,
        mirostat: 0,
        mirostat_tau: 5.0,
        mirostat_eta: 0.1,
        penalize_nl: true
    )
    
    public init(n_batch: Int32 = 512,
                temp: Float = 0.8,
                top_k: Int32 = 40,
                top_p: Float = 0.95,
                min_p: Float = 0.0,
                tfs_z: Float = 1.0,
                typical_p: Float = 1.0,
                repeat_penalty: Float = 1.1,
                repeat_last_n: Int32 = 64,
                frequence_penalty: Float = 0.0,
                presence_penalty: Float = 0.0,
                mirostat: Int32 = 0,
                mirostat_tau: Float = 5.0,
                mirostat_eta: Float = 0.1,
                penalize_nl: Bool = true,
                use_metal:Bool = false) {
        self.n_batch = n_batch
        self.temp = temp
        self.top_k = top_k
        self.top_p = top_p
        self.tfs_z = tfs_z
        self.typical_p = typical_p
        self.repeat_penalty = repeat_penalty
        self.repeat_last_n = repeat_last_n
        self.frequence_penalty = frequence_penalty
        self.presence_penalty = presence_penalty
        self.mirostat = mirostat
        self.mirostat_tau = mirostat_tau
        self.mirostat_eta = mirostat_eta
        self.penalize_nl = penalize_nl
        self.min_p = 0
    }
}

public enum ModelError: Error {
    case modelNotFound(String)
    case inputTooLong
    case failedToEval
    case contextLimit
}

public enum ModelPromptStyle {
    case None
    case Custom
    // case ChatBase
    // case OpenAssistant
    // case StableLM_Tuned
    // case LLaMa
    // case LLaMa_QA
    // case Dolly_b3
    // case RedPajama_chat
}

public typealias ModelToken = Int32

