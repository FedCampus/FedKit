import CoreML
import Flutter
import UIKit

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
        case "testSize": result(Int(9)) // TODO: Actual test size.
        case "trainingSize": result(Int(99)) // TODO: Actual training size.
        case "updateParameters": updateParameters(call, result)
        case "initML": initML(call, result)
        default: result(FlutterMethodNotImplemented)
        }
    }

    func evaluate(_ result: @escaping FlutterResult) {
        // TODO: Real evaluation.
        var fakeData = [Float32(0.9), Float32(0.1)]
        result(FlutterStandardTypedData(float32: Data(fromArray: fakeData)))
    }

    func fit(_: FlutterMethodCall, _ result: @escaping FlutterResult) {
        // TODO: Real fit.
        result(nil)
    }

    func getParameters(_ result: @escaping FlutterResult) {
        let parameters = mlClient?.getParameters().compactMap { layer in
            FlutterStandardTypedData(float32: Data(fromArray: layer))
        }
        result(parameters)
    }

    func ready(_ result: @escaping FlutterResult) {
        // TODO: Test.
        result(true)
    }

    func updateParameters(_ call: FlutterMethodCall, _ result: @escaping FlutterResult) {
        let args = call.arguments as! [String: Any]
        let params = args["parameters"] as! [FlutterStandardTypedData]
        let parameters = params.compactMap { layer in
            layer.data.toArray(type: Float.self)
        }
        mlClient?.updateParameters(parameters: parameters)
        result(nil)
    }

    func initML(_ call: FlutterMethodCall, _ result: @escaping FlutterResult) {
        let args = call.arguments as! [String: Any]
        let modelDir = args["modelDir"] as! String
        let layersSizes = (args["layersSizes"] as! [NSNumber]).compactMap { $0.int32Value }
        let partitionId = (args["partitionId"] as! NSNumber).int32Value
        DispatchQueue.global(qos: .default).async {
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
            guard let url = URL(string: modelDir) else {
                self.log.error("Model file not at \(modelDir).")
                let e = FlutterError(code: "Model file not at \(modelDir).", message: nil, details: nil)
                DispatchQueue.main.async { result(e) }
                return
            }
            do {
                self.log.error("Model URL: \(url).")
                let compiledModelUrl = try MLModel.compileModel(at: url)
                self.log.error("Compiled model URL: \(compiledModelUrl).")
                let modelData = try Data(contentsOf: url)
                self.log.error("Model data of \(modelData.count).")
                let modelInspect = try MLModelInspect(serializedData: modelData)
                self.log.error("Model initialized inspection.")
                let layerWrappers = modelInspect.getLayerWrappers()
                self.mlClient = MLClient(layerWrappers, dataLoader, compiledModelUrl)
                DispatchQueue.main.async { result(nil) }
            } catch {
                let e = FlutterError(code: "\(error)", message: error.localizedDescription, details: nil)
                DispatchQueue.main.async { result(e) }
            }
        }
    }
}
