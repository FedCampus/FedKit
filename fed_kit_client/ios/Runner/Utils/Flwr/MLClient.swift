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
}

public class MLClient {
    var parameters: MLParameter
    var dataLoader: MLDataLoader
    var compiledModelUrl: URL
    var tempModelUrl: URL
    private var paramUpdate: [[Float]]?

    init(_ layerWrappers: [MLLayerWrapper], _ dataLoader: MLDataLoader, _ compiledModelUrl: URL) {
        parameters = MLParameter(layerWrappers: layerWrappers)
        self.dataLoader = dataLoader
        self.compiledModelUrl = compiledModelUrl

        let modelFileName = compiledModelUrl.deletingPathExtension().lastPathComponent
        tempModelUrl = appDirectory.appendingPathComponent("temp\(modelFileName).mlmodelc")
    }

    func getParameters() -> [[Float]] {
        return parameters.layerWrappers.compactMap { $0.weights }
    }

    func updateParameters(parameters: [[Float]]) {
        paramUpdate = parameters
    }

    func fit() async throws {
        var configuration: MLModelConfiguration?
        if let paramUpdate {
            configuration = parameters.parametersToWeights(parameters: paramUpdate)
        }
        let updateContext = try await updateModelAsync(
            forModelAt: tempModelUrl, trainingData: dataLoader.trainBatchProvider, configuration: configuration
        )
        parameters.updateLayerWrappers(context: updateContext)
        try saveModel(updateContext)
    }

    func evaluate() async throws -> (Double, Double) {
        guard let paramUpdate else {
            throw MLClientErr.NoParamUpdate
        }
        let configuration = parameters.parametersToWeights(parameters: paramUpdate)
        configuration.parameters![MLParameterKey.epochs] = 1
        let updateContext = try await updateModelAsync(
            forModelAt: tempModelUrl, trainingData: dataLoader.testBatchProvider, configuration: configuration
        )
        let loss = updateContext.metrics[.lossValue] as! Double
        return (loss, (1.0 - loss) * 100)
    }

    private func saveModel(_ updateContext: MLUpdateContext) throws {
        let updatedModel = updateContext.model
        let fileManager = FileManager.default
        try fileManager.createDirectory(at: tempModelUrl, withIntermediateDirectories: true, attributes: nil)
        try updatedModel.write(to: tempModelUrl)
        _ = try fileManager.replaceItemAt(compiledModelUrl, withItemAt: tempModelUrl)
    }
}
