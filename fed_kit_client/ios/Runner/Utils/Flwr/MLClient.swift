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
    let layers: [Layer]
    var parameters: [MLMultiArray]?
    var dataLoader: MLDataLoader
    var compiledModelUrl: URL
    var tempModelUrl: URL
    private var paramUpdate: [[Float]]?

    init(_ layers: [Layer], _ dataLoader: MLDataLoader, _ compiledModelUrl: URL) {
        self.layers = layers
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
            forModelAt: compiledModelUrl, trainingData: dataLoader.trainBatchProvider, configuration: config
        )
        parameters = try layers.map { layer in
            let paramKey = MLParameterKey.weights.scoped(to: layer.name)
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
            forModelAt: compiledModelUrl, trainingData: dataLoader.testBatchProvider, configuration: config
        )
        let loss = updateContext.metrics[.lossValue] as! Double
        return (loss, (1.0 - loss) * 100)
    }

    /// Guarantee that the config returned has non-nil `parameters`.
    private func config() throws -> MLModelConfiguration {
        let config = MLModelConfiguration()
        if config.parameters == nil {
            config.parameters = [:]
        }
        if let paramUpdate {
            for (index, weightsArray) in paramUpdate.enumerated() {
                let shapedArray = MLShapedArray(scalars: weightsArray, shape: layers[index].shape)
                let layerParams = MLMultiArray(shapedArray)
                let paramKey = MLParameterKey.weights.scoped(to: layers[index].name)
                config.parameters![paramKey] = layerParams
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
