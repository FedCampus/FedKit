import Flutter
import UIKit

@UIApplicationMain
@objc class AppDelegate: FlutterAppDelegate {
    override func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
    ) -> Bool {
        GeneratedPluginRegistrant.register(with: self)
        register()
        return super.application(application, didFinishLaunchingWithOptions: launchOptions)
    }

    public func register() {
        let controller: FlutterViewController = window?.rootViewController as! FlutterViewController
        let channel = FlutterMethodChannel(name: "fed_kit_flutter", binaryMessenger: controller.binaryMessenger)
        channel.setMethodCallHandler(handle)
    }

    public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        switch call.method {
        case "getPlatformVersion": result("iOS " + UIDevice.current.systemVersion)
        default: result(FlutterMethodNotImplemented)
        }
    }
}
