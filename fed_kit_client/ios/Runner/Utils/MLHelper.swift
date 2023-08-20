import CoreML
import Foundation

func updateModelAsync(forModelAt: URL, trainingData: MLBatchProvider, configuration: MLModelConfiguration? = nil) async throws -> MLUpdateContext {
    let log = logger("updateModelAsync")

    let progressHandler: (MLUpdateContext) -> Void = { contextProgress in
        let loss = String(format: "%.4f", contextProgress.metrics[.lossValue] as! Double)
        log.debug("Epoch \(contextProgress.metrics[.epochIndex] as! Int + 1) finished with loss \(loss)")
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
