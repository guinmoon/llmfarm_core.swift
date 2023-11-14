//
//  File.swift
//  
//
//  Created by guinmoon on 26.08.2023.
//
import Foundation
import llmfarm_core

let maxOutputLength = 256
var total_output = 0

func mainCallback(_ str: String, _ time: Double) -> Bool {
    print("\(str)",terminator: "")
    total_output += str.count
    if(total_output>maxOutputLength){
        return true
    }
    return false
}

var input_text = "State the meaning of life."

let ai = AI(_modelPath: "orca-mini-3b.ggmlv3.q4_1.gguf",_chatName: "chat")
var params:ModelAndContextParams = .default

params.promptFormat = .Custom
params.custom_prompt_format = "### Instruction:{{prompt}}### Response:"

try? ai.loadModel(ModelInference.LLama_gguf,contextParams: params)



let output = try? ai.model.predict(input_text, mainCallback)


