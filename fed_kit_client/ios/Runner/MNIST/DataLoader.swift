//
//  DataLoader.swift
//  FlowerSDK
//
//  Created by Daniel Nugraha on 25.05.22.
//  Adapted from https://github.com/JacopoMangiavacchi/MNIST-CoreML-Training

import Compression
import CoreML
import Foundation
import os
import UIKit

let appDirectory = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!

enum DataLoader {
    private static let dataset = "MNIST"
    private static let shapeData: [NSNumber] = [1, 28, 28]
    private static let normalization: Float = 255.0
    private static let shapeTarget: [NSNumber] = [1]

    static func trainBatchProvider(progressHandler: @escaping (Int) -> Void) -> MLBatchProvider {
        return prepareMLBatchProvider(filePath: extractTrainFile(dataset: dataset), progressHandler: progressHandler)
    }

    static func testBatchProvider(progressHandler: @escaping (Int) -> Void) -> MLBatchProvider {
        return prepareMLBatchProvider(filePath: extractTestFile(dataset: dataset), progressHandler: progressHandler)
    }

    /// Extract file
    ///
    /// - parameter sourceURL:URL of source file
    /// - parameter destinationFilename: Choosen destination filename
    ///
    /// - returns: Temporary path of extracted file
    fileprivate static func extractFile(from sourceURL: URL, to destinationURL: URL) -> String {
        let sourceFileHandle = try! FileHandle(forReadingFrom: sourceURL)
        var isDir: ObjCBool = true
        let fileManager = FileManager.default
        if !fileManager.fileExists(atPath: appDirectory.path, isDirectory: &isDir) {
            try! fileManager.createDirectory(at: appDirectory, withIntermediateDirectories: true)
        }
        if fileManager.fileExists(atPath: destinationURL.path) {
            return destinationURL.path
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

        return destinationURL.path
    }

    /// Extract train file
    ///
    /// - returns: Temporary path of extracted file
    private static func extractTrainFile(dataset: String) -> String {
        let sourceURL = Bundle.main.url(forResource: dataset + "_train", withExtension: "csv.lzfse")!
        let destinationURL = appDirectory.appendingPathComponent(dataset + "_train.csv")
        return extractFile(from: sourceURL, to: destinationURL)
    }

    /// Extract test file
    ///
    /// - returns: Temporary path of extracted file
    private static func extractTestFile(dataset: String) -> String {
        let sourceURL = Bundle.main.url(forResource: dataset + "_test", withExtension: "csv.lzfse")!
        let destinationURL = appDirectory.appendingPathComponent(dataset + "_test.csv")
        return extractFile(from: sourceURL, to: destinationURL)
    }

    private static func prepareMLBatchProvider(filePath: String, progressHandler: @escaping (Int) -> Void) -> MLBatchProvider {
        var featureProviders = [MLFeatureProvider]()

        var count = 0
        errno = 0

        if freopen(filePath, "r", stdin) == nil {
            print("error opening file")
        }
        var lengthEntry = 1
        shapeData.enumerated().forEach { _, value in
            lengthEntry = Int(truncating: value) * lengthEntry
        }

        // MARK: Fails if commas occur in the values of csv

        while let line = readLine()?.split(separator: ",") {
            if count > 5000 {
                break
            }
            count += 1
            progressHandler(count)
            let imageMultiArr = try! MLMultiArray(shape: shapeData, dataType: .float32)
            let outputMultiArr = try! MLMultiArray(shape: shapeTarget, dataType: .int32)
            for i in 0 ..< lengthEntry {
                imageMultiArr[i] = NSNumber(value: Float(String(line[i]))! / normalization)
            }
            outputMultiArr[0] = NSNumber(value: Float(String(line.last!))!)
            let imageValue = MLFeatureValue(multiArray: imageMultiArr)
            let outputValue = MLFeatureValue(multiArray: outputMultiArr)
            let dataPointFeatures: [String: MLFeatureValue] = ["image": imageValue,
                                                               "output_true": outputValue]
            if let provider = try? MLDictionaryFeatureProvider(dictionary: dataPointFeatures) {
                featureProviders.append(provider)
            }
        }

        return MLArrayBatchProvider(array: featureProviders)
    }

    static func predictionBatchProvider() -> MLBatchProvider {
        var featureProviders = [MLFeatureProvider]()

        var count = 0
        errno = 0

        let testFilePath = extractTestFile(dataset: dataset)
        if freopen(testFilePath, "r", stdin) == nil {
            print("error opening file")
        }
        var lengthEntry = 1
        shapeData.enumerated().forEach { _, value in
            lengthEntry = Int(truncating: value) * lengthEntry
        }
        while let line = readLine()?.split(separator: ",") {
            count += 1
            let imageMultiArr = try! MLMultiArray(shape: shapeData, dataType: .float32)
            for i in 0 ..< lengthEntry {
                imageMultiArr[i] = NSNumber(value: Float(String(line[i]))! / normalization)
            }
            let imageValue = MLFeatureValue(multiArray: imageMultiArr)
            let dataPointFeatures: [String: MLFeatureValue] = ["image": imageValue]
            if let provider = try? MLDictionaryFeatureProvider(dictionary: dataPointFeatures) {
                featureProviders.append(provider)
            }
        }
        return MLArrayBatchProvider(array: featureProviders)
    }
}
