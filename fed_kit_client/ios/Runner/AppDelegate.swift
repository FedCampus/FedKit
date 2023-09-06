import CoreML
import Flutter
import UIKit

enum AppErr: Error {
    case ModelNotFound
}

let log = logger(String(describing: AppDelegate.self))

@UIApplicationMain
@objc class AppDelegate: FlutterAppDelegate {
    var mlClient: MLClient?
    private var dataLoader: MLDataLoader?

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
        let controller: FlutterViewController = window?.rootViewController as! FlutterViewController
        let channel = FlutterMethodChannel(name: "fed_kit_client_cifar10_ml_client", binaryMessenger: controller.binaryMessenger)
        channel.setMethodCallHandler(handle)
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
            try await self.mlClient?.fit(epochs: epochs)
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
            let partitionId = (args["partitionId"] as! NSNumber).int32Value
            let url = URL(fileURLWithPath: modelDir)
            log.error("Accessing: \(url.startAccessingSecurityScopedResource())")
            log.error("Model URL: \(url).")
            try self.checkModel(url)
            self.mlClient = try MLClient(layers, await self.dataLoader(), url)
            self.ready = true
            return nil
        }
    }

    private func checkModel(_ url: URL) throws {
        let content = try Data(contentsOf: url)
        let layers = try MLModelInspect(serializedData: content).getLayerWrappers()
        for layer in layers {
            log.error("\(layer.name), updatable: \(layer.isUpdatable), shape: \(layer.shape).")
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

    private func dataLoader() async throws -> MLDataLoader {
        if dataLoader != nil {
            return dataLoader!
        }
        let trainBatchProvider = try await trainBatchProvider { count in
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
        return dataLoader!
    }
}
