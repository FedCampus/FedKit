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

enum MLClientErr: Error {
    case ParamsNil
    case ParamNotMultiArray
    case UpdateContextNoModel
    case FeatureNoValue
    case UnexpectedLayer(String)
}

public class MLClient {
    let layers: [Layer]
    var dataLoader: MLDataLoader
    var paramUpdate = false
    var compiledModelUrl: URL
    var modelUrl: URL
    let tempModelUrl: URL
    var modelProto: ModelProto
    private var parameters: [[Float]]

    init(
        _ layers: [Layer], _ dataLoader: MLDataLoader, _ modelUrl: URL, _ modelProto: ModelProto
    ) throws {
        self.layers = layers
        self.dataLoader = dataLoader
        self.modelUrl = modelUrl
        self.modelProto = modelProto
        parameters = try modelProto.parameters(layers: layers)
        // Initial models.
        let url = try MLModel.compileModel(at: modelUrl)
        log.error("Compiled model URL: \(url).")
        compiledModelUrl = url
        let modelFileName = compiledModelUrl.deletingPathExtension().lastPathComponent
        tempModelUrl = appDirectory.appendingPathComponent("temp\(modelFileName).mlmodelc")
    }

    func getParameters() -> [[Float]] {
        return parameters
    }

    func updateParameters(parameters: [[Float]]) {
        self.parameters = parameters
        paramUpdate = true
    }

    func fit(epochs: Int? = nil, callback: ((Float) -> Void)? = nil) async throws {
        let config = try await config()
        if epochs != nil {
            config.parameters![MLParameterKey.epochs] = epochs
        }
        let updateContext = try await updateModelAsync(
            forModelAt: compiledModelUrl,
            trainingData: dataLoader.trainBatchProvider,
            configuration: config,
            progressHandler: callback.map { callback in
                { contextProgress in
                    let loss = contextProgress.metrics[.lossValue] as? Float ?? -1.0
                    callback(loss)
                }
            }
        )
        if updateContext.model == nil {
            throw MLClientErr.UpdateContextNoModel
        }
        for (index, layer) in layers.enumerated() {
            if !layer.updatable { continue }

            let paramKey = layer.type.scoped(to: layer.name)
            guard let weightsMultiArray = try updateContext.model.parameterValue(for: paramKey) as? MLMultiArray else {
                throw MLClientErr.ParamNotMultiArray
            }

            let pointer = try UnsafeBufferPointer<Float>(weightsMultiArray)
            parameters[index] = Array(pointer)
        }
        try saveModel(updateContext)
    }

    /// Currently, calculates Mean Square Error and category correctness.
    func evaluate() async throws -> (Float, Float) {
        let config = try await config()
        let model = try MLModel(contentsOf: compiledModelUrl)
        let batch = dataLoader.testBatchProvider
        let predictions = try model.predictions(fromBatch: batch)

        var totalLoss: Float = 0.0
        var nCorrect = 0
        for index in 0 ..< predictions.count {
            guard let pred =
                predictions.features(at: index).featureValue(for: modelProto.output)
            else {
                throw MLClientErr.FeatureNoValue
            }
            let actl = batch.features(at: index).featureValue(for: modelProto.target)!
            let prediction = try pred.multiArrayValue!.toArray(type: Float.self)
            let actual = try actl.multiArrayValue!.toArray(type: Double.self)
            totalLoss += meanSquareErrors(prediction, actual)
            if (actual[0] == 0 && prediction[0] < 0.5) || (actual[0] != 0 && prediction[0] >= 0.5) {
                nCorrect += 1
            }
        }

        let total = Float(predictions.count)
        let loss = totalLoss / total
        let accuracy = Float(nCorrect) / total
        return (loss, accuracy)
    }

    /// Guarantee that the config returned has non-nil `parameters`.
    /// Update `compiledModelUrl` to match new parameters.
    private func config() async throws -> MLModelConfiguration {
        let config = MLModelConfiguration()
        if config.parameters == nil {
            config.parameters = [:]
        }
        if paramUpdate {
            try modelProto.model.neuralNetwork.layers.forEachMut { nnLayer in
                let name = nnLayer.name
                for (index, layer) in layers.enumerated() {
                    if layer.name != name { continue }

                    let parameter = parameters[index]
                    switch (nnLayer.layer!, layer.type) {
                    case (.convolution, MLParameterKey.weights):
                        nnLayer.convolution.weights.floatValue = parameter
                    case (.innerProduct, MLParameterKey.weights):
                        nnLayer.innerProduct.weights.floatValue = parameter
                    case (.convolution, MLParameterKey.biases):
                        nnLayer.convolution.bias.floatValue = parameter
                    case (.innerProduct, MLParameterKey.biases):
                        nnLayer.innerProduct.bias.floatValue = parameter
                    default: throw MLClientErr.UnexpectedLayer(name)
                    }
                    log.error("Updated layer \(name) with weights of \(parameter.count).")
                }
            }
            try recompile()
            log.error("Recompiled.")
            paramUpdate = false
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

    private func recompile() throws {
        try! modelProto.model.serializedData().write(to: modelUrl)
        compiledModelUrl = try! MLModel.compileModel(at: modelUrl)
    }
}

struct MLDataLoader {
    let trainBatchProvider: MLBatchProvider
    let testBatchProvider: MLBatchProvider
}

struct ModelProto {
    var model: CoreML_Specification_Model
    let input: String
    let output: String
    let target: String

    init(mlModel: CoreML_Specification_Model) throws {
        model = mlModel
        input = try modelInput(model)
        output = try modelOutput(model)
        target = try modelTarget(model)
    }

    init(data: Data) throws {
        let model = try CoreML_Specification_Model(serializedData: data)
        try self.init(mlModel: model)
    }

    func parameters(layers: [Layer]) throws -> [[Float]] {
        var parameters = [[Float]]()
        for nnLayer in model.neuralNetwork.layers {
            let name = nnLayer.name
            for layer in layers {
                if layer.name != name { continue }
                switch (nnLayer.layer!, layer.type) {
                case (.convolution, MLParameterKey.weights):
                    parameters.append(nnLayer.convolution.weights.floatValue)
                case (.innerProduct, MLParameterKey.weights):
                    parameters.append(nnLayer.innerProduct.weights.floatValue)
                case (.convolution, MLParameterKey.biases):
                    parameters.append(nnLayer.convolution.bias.floatValue)
                case (.innerProduct, MLParameterKey.biases):
                    parameters.append(nnLayer.innerProduct.bias.floatValue)
                default: throw MLClientErr.UnexpectedLayer(name)
                }
            }
        }
        log.error("Model ProtoBuf: \(parameters.count) layers.")
        return parameters
    }
}

private func modelInput(_ mlModel: CoreML_Specification_Model) throws -> String {
    let inputs = mlModel.description_p.input
    if inputs.count != 1 {
        throw MLClientErr.UnexpectedLayer("Number of inputs \(inputs.count) is not 1")
    }
    let inputName = inputs[0].name
    log.error("Model input: \(inputName)")
    return inputName
}

private func modelOutput(_ mlModel: CoreML_Specification_Model) throws -> String {
    let outputs = mlModel.description_p.output
    if outputs.count != 1 {
        throw MLClientErr.UnexpectedLayer("Number of outputs \(outputs.count) is not 1")
    }
    let outputName = outputs[0].name
    log.error("Model output: \(outputName)")
    return outputName
}

private func modelTarget(_ mlModel: CoreML_Specification_Model) throws -> String {
    let lossLayers = mlModel.neuralNetwork.updateParams.lossLayers
    if lossLayers.count != 1 {
        throw MLClientErr.UnexpectedLayer("Number of lossLayers \(lossLayers.count) is not 1")
    }
    let lossLayer = lossLayers[0]
    let target: String
    switch lossLayer.lossLayerType {
    case let .categoricalCrossEntropyLossLayer(layer): target = layer.target
    case let .meanSquaredErrorLossLayer(layer): target = layer.target
    case .none: throw MLClientErr.UnexpectedLayer("No lossLayerType for \(lossLayer.name)")
    }
    log.error("Model target: \(target)")
    return target
}
