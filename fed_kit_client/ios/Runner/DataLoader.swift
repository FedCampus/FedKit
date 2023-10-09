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
private let shapeData: [NSNumber] = [1, 28, 28, 1]
private let lengthEntry = shapeData.reduce(1) { acc, value in
    Int(truncating: value) * acc
}

private let normalization: Float = 255.0
private let nClasses = 10
private let shapeTarget: [NSNumber] = [nClasses as NSNumber]

func trainBatchProvider(
    _ dataDir: String,
    _ partitionId: Int,
    inputName: String,
    outputName: String,
    progressHandler: @escaping (Int) -> Void
) async throws -> MLBatchProvider {
    return try await prepareMLBatchProvider(
        filePath: "\(dataDir)/MNIST_train.csv",
        inputName: inputName,
        outputName: outputName,
        progressHandler: progressHandler
    ) { index in
        index / 6000 + 1 == partitionId
    }
}

func testBatchProvider(
    _ dataDir: String,
    inputName: String,
    outputName: String,
    progressHandler: @escaping (Int) -> Void
) async throws -> MLBatchProvider {
    return try await prepareMLBatchProvider(
        filePath: "\(dataDir)/MNIST_test.csv",
        inputName: inputName,
        outputName: outputName,
        progressHandler: progressHandler
    )
}

private func prepareMLBatchProvider(
    filePath: String,
    inputName: String,
    outputName: String,
    progressHandler: @escaping (Int) -> Void,
    indexFilter: ((Int) -> Bool)? = nil
) async throws -> MLBatchProvider {
    var count = 0
    let featureProviders = try await withThrowingTaskGroup(of: MLDictionaryFeatureProvider.self) { group in
        let (bytes, _) =
            try await URLSession.shared.bytes(from: URL(fileURLWithPath: filePath))
        for try await line in bytes.lines {
            count += 1
            if indexFilter.map({ !$0(count) }) ?? false { continue }
            let countNow = count
            group.addTask {
                let splits = line.split(separator: ",")
                let imageMultiArr = try! MLMultiArray(shape: shapeData, dataType: .float32)
                let outputMultiArr = try! MLMultiArray(shape: shapeTarget, dataType: .int32)
                for i in 0 ..< lengthEntry {
                    imageMultiArr[i] = (Float(String(splits[i + 1]))! / normalization) as NSNumber
                }
                for i in 0 ..< outputMultiArr.count {
                    outputMultiArr[i] = 0
                }
                outputMultiArr[Int(String(splits[0]))!] = 1
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
