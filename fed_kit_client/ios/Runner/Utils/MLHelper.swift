import CoreML
import Foundation

func updateModelAsync(forModelAt: URL, trainingData: MLBatchProvider, configuration: MLModelConfiguration? = nil) async throws -> MLUpdateContext {
    let log = logger("updateModelAsync")

    let progressHandler: (MLUpdateContext) -> Void = { contextProgress in
        let loss = contextProgress.metrics[.lossValue] as! Double
        log.error("Epoch \(contextProgress.metrics[.epochIndex] as! Int + 1) finished with loss \(loss)")
    }

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

enum LayerConversionErr: Error {
    case missingValue(key: String)
    case unknownType(String)
}

struct Layer {
    let name: String
    let type: MLParameterKey

    init(dictionary: [String: Any?]) throws {
        guard let name = dictionary["name"] as? String else {
            throw LayerConversionErr.missingValue(key: "name")
        }

        guard let type = dictionary["type"] as? String else {
            throw LayerConversionErr.missingValue(key: "shape")
        }
        self.name = name
        switch type {
        case "weights": self.type = MLParameterKey.weights
        case "bias": self.type = MLParameterKey.biases
        default: throw LayerConversionErr.unknownType(type)
        }
    }
}
