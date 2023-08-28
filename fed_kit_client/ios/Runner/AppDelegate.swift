import CoreML
import Flutter
import UIKit

enum AppErr: Error {
    case ModelNotFound
}

@UIApplicationMain
@objc class AppDelegate: FlutterAppDelegate {
    var mlClient: MLClient?
    let log = logger(String(describing: AppDelegate.self))

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

    func fit(_: FlutterMethodCall, _ result: @escaping FlutterResult) {
        // TODO: Real fit.
        result(nil)
    }

    func getParameters(_ result: @escaping FlutterResult) {
        runAsync(result) {
            try (await self.mlClient?.getParameters().map { layer in
                FlutterStandardTypedData(float32: Data(fromArray: layer))
            })!
        }
    }

    func ready(_ result: @escaping FlutterResult) {
        // TODO: Test.
        result(true)
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
            let layersNames = args["layersNames"] as! [String]
            let partitionId = (args["partitionId"] as! NSNumber).int32Value
            let trainBatchProvider = DataLoader.trainBatchProvider { count in
                if count % 500 == 499 {
                    self.log.error("Prepared \(count) training data points.")
                }
            }
            self.log.error("trainBatchProvider: \(trainBatchProvider.count)")

            let testBatchProvider = DataLoader.testBatchProvider { count in
                if count % 500 == 499 {
                    self.log.error("Prepared \(count) test data points.")
                }
            }
            self.log.error("testBatchProvider: \(testBatchProvider.count)")

            let dataLoader = MLDataLoader(trainBatchProvider: trainBatchProvider, testBatchProvider: testBatchProvider)
            let url = URL(fileURLWithPath: modelDir)
            self.log.error("Accessing: \(url.startAccessingSecurityScopedResource())")
            self.log.error("Model URL: \(url).")
            try self.checkModel(url)
            let compiledModelUrl = try MLModel.compileModel(at: url)
            self.log.error("Compiled model URL: \(compiledModelUrl).")
            self.mlClient = MLClient(layersNames, dataLoader, compiledModelUrl)
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
}
