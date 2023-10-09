import CoreML
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

    func toArray<T>(type _: T.Type) -> [T] {
        let value = withUnsafeBytes {
            $0.baseAddress?.assumingMemoryBound(to: T.self)
        }
        return [T](UnsafeBufferPointer(start: value, count: count / MemoryLayout<T>.stride))
    }
}

extension MLMultiArray {
    func toArray<T>(type _: T.Type) throws -> [T] {
        return try Array(UnsafeBufferPointer<T>(self))
    }
}

extension MutableCollection {
    /// <https://forums.swift.org/t/inout-variables-in-for-in-loops/61380/6>
    mutating func forEachMut(_ body: (inout Element) throws -> Void) rethrows {
        var i = startIndex
        while i < endIndex {
            try body(&self[i])
            formIndex(after: &i)
        }
    }
}

extension Array where Element == Float {
    /// Assuming at least one element.
    func argmax() -> Int {
        var maxIndex = 0
        var maxValue = self[0]

        for (index, value) in enumerated() {
            if value > maxValue {
                maxIndex = index
                maxValue = value
            }
        }

        return maxIndex
    }
}

extension Array where Element == Int32 {
    /// Assuming at least one element.
    func argmax() -> Int {
        var maxIndex = 0
        var maxValue = self[0]

        for (index, value) in enumerated() {
            if value > maxValue {
                maxIndex = index
                maxValue = value
            }
        }

        return maxIndex
    }
}
