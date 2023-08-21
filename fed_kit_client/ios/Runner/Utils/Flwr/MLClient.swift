//
//  MLFlwrClient.swift
//
//
//  Created by Daniel Nugraha on 18.01.23.
//  Simplified by Steven Hé to adapt to fed_kit.
//

import CoreML
import Foundation
import NIOCore
import NIOPosix
import os

public enum MLTask {
    case train
    case test
}

enum MLClientErr: Error {
    case NoParamUpdate
    case ParamsNil
    case ParamNotMultiArray
}

public class MLClient {
    let layerNames: [String]
    var parameters: [MLMultiArray]?
    var dataLoader: MLDataLoader
    var compiledModelUrl: URL
    var tempModelUrl: URL
    private var paramUpdate: [[Float]]?

    init(_ layerNames: [String], _ dataLoader: MLDataLoader, _ compiledModelUrl: URL) {
        self.layerNames = layerNames
        self.dataLoader = dataLoader
        self.compiledModelUrl = compiledModelUrl

        let modelFileName = compiledModelUrl.deletingPathExtension().lastPathComponent
        tempModelUrl = appDirectory.appendingPathComponent("temp\(modelFileName).mlmodelc")
    }

    func getParameters() async throws -> [[Float]] {
        if parameters == nil {
            try await fit()
        }
        guard let parameters else {
            throw MLClientErr.ParamsNil
        }
        return try parameters.map { layer in
            let pointer = try UnsafeBufferPointer<Float>(layer)
            return Array(pointer)
        }
    }

    func updateParameters(parameters: [[Float]]) {
        paramUpdate = parameters
    }

    func fit() async throws {
        let config = try config()
        let updateContext = try await updateModelAsync(
            forModelAt: tempModelUrl, trainingData: dataLoader.trainBatchProvider, configuration: config
        )
        parameters = try layerNames.map { name in
            let paramKey = MLParameterKey.weights.scoped(to: name)
            guard let weightsMultiArray = try updateContext.model.parameterValue(for: paramKey) as? MLMultiArray else {
                throw MLClientErr.ParamNotMultiArray
            }
            return weightsMultiArray
        }
        try saveModel(updateContext)
    }

    func evaluate() async throws -> (Double, Double) {
        let config = try config()
        config.parameters![MLParameterKey.epochs] = 1
        let updateContext = try await updateModelAsync(
            forModelAt: tempModelUrl, trainingData: dataLoader.testBatchProvider, configuration: config
        )
        let loss = updateContext.metrics[.lossValue] as! Double
        return (loss, (1.0 - loss) * 100)
    }

    private func config() throws -> MLModelConfiguration {
        let config = MLModelConfiguration()
        if let paramUpdate {
            for (index, weightsArray) in paramUpdate.enumerated() {
                let layerParams = try MLMultiArray(weightsArray)
                let paramKey = MLParameterKey.weights.scoped(to: layerNames[index])
                config.parameters?[paramKey] = layerParams
            }
            self.paramUpdate = nil
        }
        return config
    }

    private func saveModel(_ updateContext: MLUpdateContext) throws {
        let updatedModel = updateContext.model
        let fileManager = FileManager.default
        try fileManager.createDirectory(at: tempModelUrl, withIntermediateDirectories: true, attributes: nil)
        try updatedModel.write(to: tempModelUrl)
        _ = try fileManager.replaceItemAt(compiledModelUrl, withItemAt: tempModelUrl)
    }
}
