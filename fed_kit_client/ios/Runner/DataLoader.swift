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
import os
import UIKit

let appDirectory = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!

enum DataLoader {
    private static let dataset = "MNIST"
    private static let shapeData: [NSNumber] = [1, 28, 28]
    private static let lengthEntry = shapeData.reduce(1) { acc, value in
        Int(truncating: value) * acc
    }

    private static let normalization: Float = 255.0
    private static let shapeTarget: [NSNumber] = [1]
    static let log = logger(String(describing: DataLoader.self))

    static func trainBatchProvider(
        progressHandler: @escaping (Int) -> Void
    ) async throws -> MLBatchProvider {
        return try await prepareMLBatchProvider(
            filePath: extractTrainFile(dataset: dataset), progressHandler: progressHandler
        )
    }

    static func testBatchProvider(
        progressHandler: @escaping (Int) -> Void
    ) async throws -> MLBatchProvider {
        return try await prepareMLBatchProvider(
            filePath: extractTestFile(dataset: dataset), progressHandler: progressHandler
        )
    }

    /// Extract file
    ///
    /// - parameter sourceURL:URL of source file
    /// - parameter destinationFilename: Choosen destination filename
    ///
    /// - returns: Temporary path of extracted file
    fileprivate static func extractFile(from sourceURL: URL, to destinationURL: URL) -> URL {
        let sourceFileHandle = try! FileHandle(forReadingFrom: sourceURL)
        var isDir: ObjCBool = true
        let fileManager = FileManager.default
        if !fileManager.fileExists(atPath: appDirectory.path, isDirectory: &isDir) {
            try! fileManager.createDirectory(at: appDirectory, withIntermediateDirectories: true)
        }
        if fileManager.fileExists(atPath: destinationURL.path) {
            return destinationURL
        }
        FileManager.default.createFile(atPath: destinationURL.path,
                                       contents: nil,
                                       attributes: nil)

        let destinationFileHandle = try! FileHandle(forWritingTo: destinationURL)
        let bufferSize = 65536

        let filter = try! OutputFilter(.decompress, using: .lzfse, bufferCapacity: 655_360) { data in
            if let data = data {
                destinationFileHandle.write(data)
            }
        }

        while true {
            let data = sourceFileHandle.readData(ofLength: bufferSize)

            try! filter.write(data)
            if data.count < bufferSize {
                break
            }
        }

        sourceFileHandle.closeFile()
        destinationFileHandle.closeFile()

        return destinationURL
    }

    /// Extract train file
    ///
    /// - returns: Temporary path of extracted file
    private static func extractTrainFile(dataset: String) -> URL {
        let sourceURL = Bundle.main.url(forResource: dataset + "_train", withExtension: "csv.lzfse")!
        let destinationURL = appDirectory.appendingPathComponent(dataset + "_train.csv")
        return extractFile(from: sourceURL, to: destinationURL)
    }

    /// Extract test file
    ///
    /// - returns: Temporary path of extracted file
    private static func extractTestFile(dataset: String) -> URL {
        let sourceURL = Bundle.main.url(forResource: dataset + "_test", withExtension: "csv.lzfse")!
        let destinationURL = appDirectory.appendingPathComponent(dataset + "_test.csv")
        return extractFile(from: sourceURL, to: destinationURL)
    }

    private static func prepareMLBatchProvider(
        filePath: URL, progressHandler: @escaping (Int) -> Void
    ) async throws -> MLBatchProvider {
        var count = 0
        let featureProviders = try await withThrowingTaskGroup(of: MLDictionaryFeatureProvider.self) { group in
            let (bytes, _) = try await URLSession.shared.bytes(from: filePath)
            for try await line in bytes.lines {
                let splits = line.split(separator: ",")
                count += 1
                let countNow = count
                group.addTask {
                    let imageMultiArr = try! MLMultiArray(shape: shapeData, dataType: .float32)
                    let outputMultiArr = try! MLMultiArray(shape: shapeTarget, dataType: .int32)
                    for i in 0 ..< lengthEntry {
                        imageMultiArr[i] = (Float(String(splits[i]))! / normalization) as NSNumber
                    }
                    outputMultiArr[0] = NSNumber(value: Float(String(splits.last!))!)
                    let imageValue = MLFeatureValue(multiArray: imageMultiArr)
                    let outputValue = MLFeatureValue(multiArray: outputMultiArr)
                    let dataPointFeatures: [String: MLFeatureValue] = ["image": imageValue,
                                                                       "output_true": outputValue]
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
}
