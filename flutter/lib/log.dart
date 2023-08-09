import 'package:logger/logger.dart';

//https://stackoverflow.com/questions/70145480/dart-singleton-with-parameters-global-app-logger
Logger get logger => Log.instance;

class Log extends Logger {
  Log._() : super(printer: PrettyPrinter(printTime: true));
  static final instance = Log._();
}
