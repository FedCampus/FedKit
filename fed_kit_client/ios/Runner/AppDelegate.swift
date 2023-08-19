import Flutter
import UIKit

@UIApplicationMain
@objc class AppDelegate: FlutterAppDelegate {
    private var parameters: [FlutterStandardTypedData]?
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
        result(FlutterStandardTypedData(float32: Data(buffer: UnsafeBufferPointer(start: fakeData, count: 2))))
    }

    func fit(_: FlutterMethodCall, _ result: @escaping FlutterResult) {
        // TODO: Real fit.
        result(nil)
    }

    func getParameters(_ result: @escaping FlutterResult) {
        // TODO: Real parameters.
        result(parameters)
    }

    func ready(_ result: @escaping FlutterResult) {
        // TODO: Test.
        result(true)
    }

    func updateParameters(_ call: FlutterMethodCall, _ result: @escaping FlutterResult) {
        let args = call.arguments as! [String: Any]
        parameters = args["parameters"] as? [FlutterStandardTypedData]
        // TODO: Real updates.
        result(nil)
    }

    func initML(_: FlutterMethodCall, _ result: @escaping FlutterResult) {
        // TODO: Initialize model and load data.
        result(nil)
    }
}
