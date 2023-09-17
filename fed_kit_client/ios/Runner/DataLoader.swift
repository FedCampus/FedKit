//
//  DataLoader.swift
//  FlowerSDK
//
//  Created by Daniel Nugraha on 25.05.22.
//  Adapted from https://github.com/JacopoMangiavacchi/MNIST-CoreML-Training
//  Modified by Steven Hé to adapt to fed_kit.

import Compression
import CoreML
import Foundation

let appDirectory = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!

private let dataset = "MNIST"
private let shapeData: [NSNumber] = [1, 28, 28]
private let lengthEntry = shapeData.reduce(1) { acc, value in
    Int(truncating: value) * acc
}

private let normalization: Float = 255.0
private let shapeTarget: [NSNumber] = [1]

func trainBatchProvider(
    _ partitionId: Int,
    inputName: String,
    outputName: String,
    progressHandler: @escaping (Int) -> Void
) async throws -> MLBatchProvider {
    return try await prepareMLBatchProvider(
        filePath: extract("_train"),
        inputName: inputName,
        outputName: outputName,
        progressHandler: progressHandler
    ) { index in
        index / 6000 + 1 == partitionId
    }
}

func testBatchProvider(
    inputName: String,
    outputName: String,
    progressHandler: @escaping (Int) -> Void
) async throws -> MLBatchProvider {
    return try await prepareMLBatchProvider(
        filePath: extract("_test"),
        inputName: inputName,
        outputName: outputName,
        progressHandler: progressHandler
    )
}

private func extract(_ set: String) throws -> URL {
    let sourceURL = Bundle.main.url(forResource: dataset + set, withExtension: "csv.lzfse")!
    let destinationURL = appDirectory.appendingPathComponent(dataset + set + ".csv")
    return try extractFile(from: sourceURL, to: destinationURL)
}

/// Extract file
///
/// - parameter sourceURL:URL of source file
/// - parameter destinationFilename: Choosen destination filename
///
/// - returns: Temporary path of extracted file
private func extractFile(from sourceURL: URL, to destinationURL: URL) throws -> URL {
    let fileManager = FileManager.default
    if fileManager.fileExists(atPath: destinationURL.path) {
        return destinationURL
    }
    var isDir: ObjCBool = true
    if !fileManager.fileExists(atPath: appDirectory.path, isDirectory: &isDir) {
        try fileManager.createDirectory(at: appDirectory, withIntermediateDirectories: true)
    }
    fileManager.createFile(atPath: destinationURL.path, contents: nil, attributes: nil)

    let destinationFileHandle = try FileHandle(forWritingTo: destinationURL)
    defer {
        destinationFileHandle.closeFile()
    }

    let filter = try OutputFilter(.decompress, using: .lzfse) { data in
        if let data = data {
            destinationFileHandle.write(data)
        }
    }

    let sourceFileHandle = try FileHandle(forReadingFrom: sourceURL)
    defer {
        sourceFileHandle.closeFile()
    }
    let bufferSize = 65536
    while true {
        let data = sourceFileHandle.readData(ofLength: bufferSize)

        try filter.write(data)
        if data.count < bufferSize {
            break
        }
    }
    try filter.finalize()

    return destinationURL
}

private func prepareMLBatchProvider(
    filePath: URL,
    inputName: String,
    outputName: String,
    progressHandler: @escaping (Int) -> Void,
    indexFilter: ((Int) -> Bool)? = nil
) async throws -> MLBatchProvider {
    var count = 0
    let featureProviders = try await withThrowingTaskGroup(of: MLDictionaryFeatureProvider.self) { group in
        let (bytes, _) = try await URLSession.shared.bytes(from: filePath)
        for try await line in bytes.lines {
            count += 1
            if indexFilter.map({ !$0(count) }) ?? false { continue }
            let countNow = count
            group.addTask {
                let splits = line.split(separator: ",")
                let imageMultiArr = try! MLMultiArray(shape: shapeData, dataType: .float32)
                let outputMultiArr = try! MLMultiArray(shape: shapeTarget, dataType: .int32)
                for i in 0 ..< lengthEntry {
                    imageMultiArr[i] = (Float(String(splits[i]))! / normalization) as NSNumber
                }
                outputMultiArr[0] = NSNumber(value: Float(String(splits.last!))!)
                let imageValue = MLFeatureValue(multiArray: imageMultiArr)
                let outputValue = MLFeatureValue(multiArray: outputMultiArr)
                let dataPointFeatures: [String: MLFeatureValue] =
                    [inputName: imageValue, outputName: outputValue]
                progressHandler(countNow)
                return try! MLDictionaryFeatureProvider(dictionary: dataPointFeatures)
            }
        }

        var featureProviders = [MLFeatureProvider]()
        for try await provider in group {
            featureProviders.append(provider)
        }
        return featureProviders
    }

    return MLArrayBatchProvider(array: featureProviders)
}
