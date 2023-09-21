//
//  File.swift
//  
//
//  Created by guinmoon on 21.09.2023.
//

import Foundation


extension Array {
    func chunked(into size: Int) -> [[Element]] {
        return stride(from: 0, to: count, by: size).map {
            Array(self[$0 ..< Swift.min($0 + size, count)])
        }
    }
}

extension String {
    
    // Convenience
    func binURL(_ dir: URL) -> URL {
        dir.appendingPathComponent(self).appendingPathExtension("bin")
    }
    
    // Used with a filename to produce a url for saving to
    func localModelSaveURL() -> URL {
        let supportDir = try! FileManager.default.url(for: .applicationSupportDirectory, in: .allDomainsMask, appropriateFor: nil, create: true)
        return supportDir.appendingPathComponent(self)
    }
    
}

extension String: Error {
}

