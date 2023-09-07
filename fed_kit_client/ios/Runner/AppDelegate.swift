import CoreML
import Flutter
import UIKit

let log = logger(String(describing: AppDelegate.self))

@UIApplicationMain
@objc class AppDelegate: FlutterAppDelegate, FlutterStreamHandler {
    var eventSink: FlutterEventSink?
    func onListen(withArguments _: Any?, eventSink events: @escaping FlutterEventSink) -> FlutterError? {
        eventSink = events
        return nil
    }

    func onCancel(withArguments _: Any?) -> FlutterError? {
        eventSink = nil
        return nil
    }

    var mlClient: MLClient?
    private var dataLoader: MLDataLoader?
    private var partitionId = -1

    var ready = false

    override func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
    ) -> Bool {
        GeneratedPluginRegistrant.register(with: self)
        register()
        return super.application(application, didFinishLaunchingWithOptions: launchOptions)
    }

    func register() {
        let controller = window?.rootViewController as! FlutterViewController
        let messenger = controller.binaryMessenger
        FlutterMethodChannel(
            name: "fed_kit_client_cifar10_ml_client", binaryMessenger: messenger
        ).setMethodCallHandler(handle)
        FlutterEventChannel(
            name: "fed_kit_client_cifar10_ml_client_log", binaryMessenger: messenger
        ).setStreamHandler(self)
    }

    func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        switch call.method {
        case "getPlatformVersion": result("iOS " + UIDevice.current.systemVersion)
        case "evaluate": evaluate(result)
        case "fit": fit(call, result)
        case "getParameters": getParameters(result)
        case "ready": ready(result)
        case "testSize": result(Int(mlClient!.dataLoader.testBatchProvider.count))
        case "trainingSize": result(Int(mlClient!.dataLoader.trainBatchProvider.count))
        case "updateParameters": updateParameters(call, result)
        case "initML": initML(call, result)
        default: result(FlutterMethodNotImplemented)
        }
    }

    func evaluate(_ result: @escaping FlutterResult) {
        runAsync(result) {
            let (loss, accuracy) = try await self.mlClient!.evaluate()
            let lossAccuracy = [Float(loss), Float(accuracy)]
            return FlutterStandardTypedData(float32: Data(fromArray: lossAccuracy))
        }
    }

    func fit(_ call: FlutterMethodCall, _ result: @escaping FlutterResult) {
        runAsync(result) {
            let args = call.arguments as! [String: Any]
            let epochs = args["epochs"] as? Int
            try await self.mlClient?.fit(epochs: epochs) { loss in
                DispatchQueue.main.async {
                    self.eventSink.map { $0([loss]) }
                }
            }
            return nil
        }
    }

    func getParameters(_ result: @escaping FlutterResult) {
        runAsync(result) {
            try await self.mlClient!.getParameters().map { layer in
                FlutterStandardTypedData(bytes: Data(fromArray: layer))
            }
        }
    }

    func ready(_ result: @escaping FlutterResult) {
        result(ready)
    }

    func updateParameters(_ call: FlutterMethodCall, _ result: @escaping FlutterResult) {
        let args = call.arguments as! [String: Any]
        let params = args["parameters"] as! [FlutterStandardTypedData]
        let parameters = params.map { layer in
            layer.data.toArray(type: Float.self)
        }
        mlClient?.updateParameters(parameters: parameters)
        result(nil)
    }

    func initML(_ call: FlutterMethodCall, _ result: @escaping FlutterResult) {
        runAsync(result) {
            let args = call.arguments as! [String: Any]
            let modelDir = args["modelDir"] as! String
            let layers = try (args["layersSizes"] as! [[String: Any]]).map(Layer.init)
            let partitionId = (args["partitionId"] as! NSNumber).intValue
            let url = URL(fileURLWithPath: modelDir)
            log.error("Accessing: \(url.startAccessingSecurityScopedResource())")
            log.error("Model URL: \(url).")
            self.mlClient = try MLClient(layers, await self.dataLoader(partitionId), url)
            self.ready = true
            return nil
        }
    }

    private func runAsync(_ result: @escaping FlutterResult, _ task: @escaping () async throws -> Any?) {
        DispatchQueue.global(qos: .default).async {
            Task {
                do {
                    let output = try await task()
                    DispatchQueue.main.async { result(output) }
                } catch {
                    let e = FlutterError(code: "\(error)", message: error.localizedDescription, details: nil)
                    DispatchQueue.main.async { result(e) }
                }
            }
        }
    }

    private func dataLoader(_ partitionId: Int) async throws -> MLDataLoader {
        if dataLoader != nil && self.partitionId == partitionId {
            return dataLoader!
        }
        let trainBatchProvider = try await trainBatchProvider(partitionId) { count in
            if count % 1000 == 999 {
                log.error("Prepared \(count) training data points.")
            }
        }
        log.error("trainBatchProvider: \(trainBatchProvider.count)")

        let testBatchProvider = try await testBatchProvider { count in
            if count % 1000 == 999 {
                log.error("Prepared \(count) test data points.")
            }
        }
        log.error("testBatchProvider: \(testBatchProvider.count)")

        dataLoader = MLDataLoader(trainBatchProvider: trainBatchProvider, testBatchProvider: testBatchProvider)
        self.partitionId = partitionId
        return dataLoader!
    }
}
