//
//  File.swift
//
//
//  Created by guinmoon on 26.08.2023.
//

import Foundation
import llmfarm_core

let maxOutputLength = 512
var total_output = 0

func mainCallback(_ str: String, _ time: Double) -> Bool {
    print("\(str)",terminator: "")
    total_output += str.count
    if(total_output>maxOutputLength){
        return true
    }
    return false
}

var input_text = "From fairest creatures"

let ai = AI(_modelPath: "alpaca_llama_etc/openllama-3b-v2-q8_0.gguf",_chatName: "chat")
var params:ModelAndContextParams = .default
params.use_metal = true
params.promptFormat = .Custom
params.custom_prompt_format = "### Instruction:{{prompt}}### Response:"
params.lora_adapters.append(("lora-open-llama-3b-v2-q8_0-shakespeare-LATEST.bin",1.0 ))

try? ai.loadModel(ModelInference.LLama_gguf,contextParams: params)


let output = try? ai.model.predict(input_text, mainCallback)
