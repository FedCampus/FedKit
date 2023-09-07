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
    case UnexpectedLayer(String)
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

    init(_ layers: [Layer], _ dataLoader: MLDataLoader, _ modelUrl: URL) throws {
        self.layers = layers
        self.dataLoader = dataLoader
        // Initial models.
        let url = try MLModel.compileModel(at: modelUrl)
        log.error("Compiled model URL: \(url).")
        compiledModelUrl = url
        let modelFileName = compiledModelUrl.deletingPathExtension().lastPathComponent
        tempModelUrl = appDirectory.appendingPathComponent("temp\(modelFileName).mlmodelc")
        rewriteModelUrl = appDirectory.appendingPathComponent("rewrite\(modelFileName).mlmodel")
        // ProtoBuf representation.
        let content = try Data(contentsOf: modelUrl)
        mlModel = try CoreML_Specification_Model(serializedData: content)
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
            let paramKey = layer.type.scoped(to: layer.name)
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
    /// Update `compiledModelUrl` to match new parameters.
    private func config() async throws -> MLModelConfiguration {
        let config = MLModelConfiguration()
        if config.parameters == nil {
            config.parameters = [:]
        }
        if let paramUpdate {
            try mlModel.neuralNetwork.layers.forEachMut { nnLayer in
                let name = nnLayer.name
                for (index, layer) in layers.enumerated() {
                    if layer.name != name { continue }
                    switch (nnLayer.layer!, layer.type) {
                    case (.convolution, MLParameterKey.weights):
                        nnLayer.convolution.weights.floatValue = paramUpdate[index]
                    case (.innerProduct, MLParameterKey.weights):
                        nnLayer.innerProduct.weights.floatValue = paramUpdate[index]
                    case (.convolution, MLParameterKey.biases):
                        nnLayer.convolution.bias.floatValue = paramUpdate[index]
                    case (.innerProduct, MLParameterKey.biases):
                        nnLayer.innerProduct.bias.floatValue = paramUpdate[index]
                    default: throw MLClientErr.UnexpectedLayer(name)
                    }
                    log.error("Updated layer \(name) with weights of \(paramUpdate[index].count).")
                }
            }
            try recompile()
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

    private func recompile() throws {
        try mlModel.serializedData().write(to: rewriteModelUrl)
        compiledModelUrl = try MLModel.compileModel(at: rewriteModelUrl)
    }
}
