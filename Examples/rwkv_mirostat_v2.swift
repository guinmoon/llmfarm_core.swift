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

let ai = AI(_modelPath: "q4_1-RWKV-4-Raven-1B5-v12-Eng98%-Other2%-20230520-ctx4096.bin",_chatName: "chat")
var params:ModelAndContextParams = .default
params.promptFormat = .None
ai.model.sampleParams.mirostat = 2
ai.model.sampleParams.mirostat_eta = 0.1
ai.model.sampleParams.mirostat_tau = 5.0

try? ai.loadModel(ModelInference.RWKV,contextParams: params)

let output = try? ai.model.predict(input_text, mainCallback)
