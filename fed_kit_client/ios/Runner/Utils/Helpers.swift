import Foundation
import os

func logger(_ category: String) -> Logger {
    return Logger(subsystem: Bundle.main.bundleIdentifier ?? "fed_kit_client", category: category)
}

/// From <https://stackoverflow.com/questions/24196820/nsdata-from-byte-array-in-swift>
extension Data {
    init<T>(fromArray values: [T]) {
        var values = values
        self.init(buffer: UnsafeBufferPointer(start: &values, count: values.count))
    }

    func toArray<T>(type: T.Type) -> [T] {
        let value = self.withUnsafeBytes {
            $0.baseAddress?.assumingMemoryBound(to: T.self)
        }
        return [T](UnsafeBufferPointer(start: value, count: self.count / MemoryLayout<T>.stride))
    }
}
