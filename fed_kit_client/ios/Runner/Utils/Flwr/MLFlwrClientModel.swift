//
//  File.swift
//
//
//  Created by Daniel Nugraha on 20.01.23.
//

import CoreML
import Foundation
import os

public struct MLDataLoader {
    public let trainBatchProvider: MLBatchProvider
    public let testBatchProvider: MLBatchProvider

    public init(trainBatchProvider: MLBatchProvider, testBatchProvider: MLBatchProvider) {
        self.trainBatchProvider = trainBatchProvider
        self.testBatchProvider = testBatchProvider
    }
}

public struct MLLayerWrapper {
    let shape: [Int16]
    let name: String
    var weights: [Float]
    let isUpdatable: Bool

    public init(shape: [Int16], name: String, weights: [Float], isUpdatable: Bool) {
        self.shape = shape
        self.name = name
        self.weights = weights
        self.isUpdatable = isUpdatable
    }
}

struct MLResult {
    let loss: Double
    let numSamples: Int
    let accuracy: Double
}

public class MLParameter {
    var layerWrappers: [MLLayerWrapper]
    private let log = Logger(subsystem: Bundle.main.bundleIdentifier ?? "flwr.Flower",
                             category: String(describing: MLParameter.self))

    /// Inits MLParameter class that contains information about the model parameters and implements routines for their update and transformation.
    ///
    /// - Parameters:
    ///   - layerWrappers: Information about the layer provided with primitive data types.
    public init(layerWrappers: [MLLayerWrapper]) {
        self.layerWrappers = layerWrappers
    }

    /// - Returns: Specification of the machine learning model configuration in the CoreML structure.
    public func parametersToWeights(parameters: [[Float]]) -> MLModelConfiguration {
        let config = MLModelConfiguration()

        guard parameters.count == layerWrappers.count else {
            log.info("parameters received is not valid")
            return config
        }

        for (index, weightsArray) in parameters.enumerated() {
            let expectedNumberOfElements = layerWrappers[index].shape.map { Int($0) }.reduce(1, *)
            guard weightsArray.count == expectedNumberOfElements else {
                log.info("array received has wrong number of elements")
                continue
            }
            layerWrappers[index].weights = weightsArray
            if layerWrappers[index].isUpdatable {
                if let weightsMultiArray = try? MLMultiArray(
                    shape: layerWrappers[index].shape as [NSNumber], dataType: .float
                ) {
                    for (index, element) in weightsArray.enumerated() {
                        weightsMultiArray[index] = element as NSNumber
                    }
                    let paramKey = MLParameterKey.weights.scoped(to: layerWrappers[index].name)
                    config.parameters?[paramKey] = weightsMultiArray
                }
            }
        }

        return config
    }

    /// Updates the layers given the CoreML update context
    ///
    /// - Parameters:
    ///   - context: The context of the update procedure of the CoreML model.
    public func updateLayerWrappers(context: MLUpdateContext) {
        for (index, layer) in layerWrappers.enumerated() {
            if layer.isUpdatable {
                let paramKey = MLParameterKey.weights.scoped(to: layer.name)
                if let weightsMultiArray = try? context.model.parameterValue(for: paramKey) as? MLMultiArray {
                    let weightsShape = Array(weightsMultiArray.shape.map { Int16(truncating: $0) }.drop(while: { $0 < 2 }))
                    guard weightsShape == layer.shape else {
                        log.info("shape \(weightsShape) is not the same as \(layer.shape)")
                        continue
                    }

                    if let pointer = try? UnsafeBufferPointer<Float>(weightsMultiArray) {
                        let array = pointer.compactMap { $0 }
                        layerWrappers[index].weights = array
                    }
                }
            }
        }
    }
}
