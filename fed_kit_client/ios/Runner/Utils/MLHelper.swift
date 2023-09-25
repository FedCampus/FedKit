import CoreML
import Foundation

func updateModelAsync(
    forModelAt: URL,
    trainingData: MLBatchProvider,
    configuration: MLModelConfiguration? = nil,
    progressHandler: ((MLUpdateContext) -> Void)? = nil
) async throws -> MLUpdateContext {
    let log = logger("updateModelAsync")

    let progressHandler = progressHandler ?? defaultProgressHandler

    return try await withCheckedThrowingContinuation { continuation in
        let completionHandler: (MLUpdateContext) -> Void = { updateContext in
            continuation.resume(with: Result { updateContext })
        }
        let progressHandlers = MLUpdateProgressHandlers(
            forEvents: [.epochEnd],
            progressHandler: progressHandler,
            completionHandler: completionHandler
        )
        do {
            let updateTask = try MLUpdateTask(
                forModelAt: forModelAt, trainingData: trainingData, configuration: configuration, progressHandlers: progressHandlers
            )
            updateTask.resume()
        } catch {
            continuation.resume(throwing: error)
        }
    }
}

func defaultProgressHandler(contextProgress: MLUpdateContext) {
    let loss = contextProgress.metrics[.lossValue] as! Float
    log.error("Epoch \(contextProgress.metrics[.epochIndex] as! Int + 1) finished with loss \(loss)")
}

enum LayerConversionErr: Error {
    case missingValue(key: String, dict: [String: Any?])
    case unknownType(String)
}

struct Layer {
    let name: String
    let type: MLParameterKey
    let updatable: Bool

    init(dictionary: [String: Any?]) throws {
        guard let name = dictionary["name"] as? String else {
            throw LayerConversionErr.missingValue(key: "name", dict: dictionary)
        }

        guard let type = dictionary["type"] as? String else {
            throw LayerConversionErr.missingValue(key: "shape", dict: dictionary)
        }
        self.name = name
        switch type {
        case "weights": self.type = MLParameterKey.weights
        case "bias": self.type = MLParameterKey.biases
        default: throw LayerConversionErr.unknownType(type)
        }

        if let updatable = dictionary["updatable"] as? Bool {
            self.updatable = updatable
        } else {
            throw LayerConversionErr.missingValue(key: "updatable", dict: dictionary)
        }
    }
}

/// Assuming the two have the same length.
func meanSquareErrors(_ a: [Float], _ b: [Double]) -> Float {
    var sum: Float = 0.0
    for (index, value) in a.enumerated() {
        let diff = value - Float(b[index])
        sum += diff * diff
    }
    return sum
}
