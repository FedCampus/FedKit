import Foundation
import os

func logger(_ category: String) -> Logger {
    return Logger(subsystem: Bundle.main.bundleIdentifier ?? "fed_kit_client", category: category)
}
