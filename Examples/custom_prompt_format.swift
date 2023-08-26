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

let ai = AI(_modelPath: "orca-mini-3b.ggmlv3.q4_1.bin",_chatName: "chat")
var params:ModelContextParams = .default

try? ai.loadModel(ModelInference.LLamaInference,contextParams: params)

ai.model.promptFormat = .Custom
ai.model.custom_prompt_format = "### Instruction:{{prompt}}### Response:"

let output = try? ai.model.predict(input_text, mainCallback)


