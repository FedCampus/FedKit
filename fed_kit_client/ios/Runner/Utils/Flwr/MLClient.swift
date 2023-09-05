//
//  MLClient.swift
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
    var rewriteModelUrl: URL
    var tempModelUrl: URL
    var mlModel: CoreML_Specification_Model
    private var paramUpdate: [[Float]]?

    let log = logger(String(describing: MLClient.self))

    init(_ layers: [Layer], _ dataLoader: MLDataLoader, _ modelUrl: URL) async throws {
        self.layers = layers
        self.dataLoader = dataLoader
        compiledModelUrl = try await MLModel.compileModel(at: modelUrl)
        log.error("Compiled model URL: \(compiledModelUrl).")
        let modelFileName = compiledModelUrl.deletingPathExtension().lastPathComponent
        tempModelUrl = appDirectory.appendingPathComponent("temp\(modelFileName).mlmodelc")
        rewriteModelUrl = appDirectory.appendingPathComponent("rewrite\(modelFileName).mlmodel")

        let content = try Data(contentsOf: modelUrl)
        mlModel = try CoreML_Specification_Model(serializedData: content)
        try mlModel.serializedData().write(to: rewriteModelUrl)
        compiledModelUrl = try await MLModel.compileModel(at: rewriteModelUrl)
    }

    func getParameters() async throws -> [[Float]] {
        return try await parameters().map { layer in
            let pointer = try UnsafeBufferPointer<Float>(layer)
            return Array(pointer)
        }
    }

    func updateParameters(parameters: [[Float]]) {
        paramUpdate = parameters
    }

    func fit(epochs: Int? = nil) async throws {
        let config = try await config()
        if epochs != nil {
            config.parameters![MLParameterKey.epochs] = epochs
        }
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
        let config = try await config()
        config.parameters![MLParameterKey.epochs] = 1
        let updateContext = try await updateModelAsync(
            forModelAt: compiledModelUrl, trainingData: dataLoader.testBatchProvider, configuration: config
        )
        let loss = updateContext.metrics[.lossValue] as! Double
        return (loss, (1.0 - loss) * 100)
    }

    /// Guarantee that the config returned has non-nil `parameters`.
    private func config() async throws -> MLModelConfiguration {
        let config = MLModelConfiguration()
        if config.parameters == nil {
            config.parameters = [:]
        }
        if let paramUpdate {
            for (index, weightsArray) in paramUpdate.enumerated() {
                let layer = layers[index]
                let shapedArray = MLShapedArray(scalars: weightsArray, shape: layer.shape)
                let layerParams = MLMultiArray(shapedArray)
                log.error("MLClient: layerParams for \(layer.name) shape: \(layerParams.shape) count: \(layerParams.count) is float: \(layerParams.dataType == .float).")
                let paramKey = MLParameterKey.weights.scoped(to: layer.name)
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

    private func parameters() async throws -> [MLMultiArray] {
        if parameters == nil {
            try await fit(epochs: 1)
        }
        guard let parameters else {
            throw MLClientErr.ParamsNil
        }
        return parameters
    }
}
