import CoreML
import Foundation

func updateModelAsync(forModelAt: URL, trainingData: MLBatchProvider, configuration: MLModelConfiguration? = nil) async throws -> MLUpdateContext {
    let log = logger("updateModelAsync")

    let progressHandler: (MLUpdateContext) -> Void = { contextProgress in
        let loss = String(format: "%.4f", contextProgress.metrics[.lossValue] as! Double)
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

enum ConversionError: Error {
    case missingValue(key: String)
    case invalidValue(key: String, expectedType: Any.Type)
}

struct Layer {
    let name: String
    let shape: [NSNumber]

    init(dictionary: [String: Any?]) throws {
        guard let name = dictionary["name"] as? String else {
            throw ConversionError.missingValue(key: "name")
        }

        guard let shape = dictionary["shape"] as? [NSNumber] else {
            throw ConversionError.missingValue(key: "shape")
        }
        self.name = name
        self.shape = shape
    }
}
