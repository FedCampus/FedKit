// DO NOT EDIT.
// swift-format-ignore-file
//
// Generated by the Swift generator plugin for the protocol buffer compiler.
// Source: VisionFeaturePrint.proto
//
// For information on using the generated types, please see the documentation:
//   https://github.com/apple/swift-protobuf/

// Copyright (c) 2018, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in LICENSE.txt or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import SwiftProtobuf

// If the compiler emits an error on this type, it is because this file
// was generated by a version of the `protoc` Swift plug-in that is
// incompatible with the version of SwiftProtobuf to which you are linking.
// Please ensure that you are building against the same version of the API
// that was used to generate this file.
private struct _GeneratedWithProtocGenSwiftVersion: SwiftProtobuf.ProtobufAPIVersionCheck {
    struct _2: SwiftProtobuf.ProtobufAPIVersion_2 {}
    typealias Version = _2
}

/// *
/// A model which takes an input image and outputs array(s) of features
/// according to the specified feature types
struct CoreML_Specification_CoreMLModels_VisionFeaturePrint {
    // SwiftProtobuf.Message conformance is added in an extension below. See the
    // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
    // methods supported on all messages.

    /// Vision feature print type
    var visionFeaturePrintType: CoreML_Specification_CoreMLModels_VisionFeaturePrint.OneOf_VisionFeaturePrintType?

    var scene: CoreML_Specification_CoreMLModels_VisionFeaturePrint.Scene {
        get {
            if case let .scene(v)? = visionFeaturePrintType { return v }
            return CoreML_Specification_CoreMLModels_VisionFeaturePrint.Scene()
        }
        set { visionFeaturePrintType = .scene(newValue) }
    }

    var objects: CoreML_Specification_CoreMLModels_VisionFeaturePrint.Objects {
        get {
            if case let .objects(v)? = visionFeaturePrintType { return v }
            return CoreML_Specification_CoreMLModels_VisionFeaturePrint.Objects()
        }
        set { visionFeaturePrintType = .objects(newValue) }
    }

    var unknownFields = SwiftProtobuf.UnknownStorage()

    /// Vision feature print type
    enum OneOf_VisionFeaturePrintType: Equatable {
        case scene(CoreML_Specification_CoreMLModels_VisionFeaturePrint.Scene)
        case objects(CoreML_Specification_CoreMLModels_VisionFeaturePrint.Objects)

        #if !swift(>=4.1)
            static func == (lhs: CoreML_Specification_CoreMLModels_VisionFeaturePrint.OneOf_VisionFeaturePrintType, rhs: CoreML_Specification_CoreMLModels_VisionFeaturePrint.OneOf_VisionFeaturePrintType) -> Bool {
                // The use of inline closures is to circumvent an issue where the compiler
                // allocates stack space for every case branch when no optimizations are
                // enabled. https://github.com/apple/swift-protobuf/issues/1034
                switch (lhs, rhs) {
                case (.scene, .scene): return {
                        guard case let .scene(l) = lhs, case let .scene(r) = rhs else { preconditionFailure() }
                        return l == r
                    }()
                case (.objects, .objects): return {
                        guard case let .objects(l) = lhs, case let .objects(r) = rhs else { preconditionFailure() }
                        return l == r
                    }()
                default: return false
                }
            }
        #endif
    }

    /// Scene extracts features useful for identifying contents of natural images
    /// in both indoor and outdoor environments
    struct Scene {
        // SwiftProtobuf.Message conformance is added in an extension below. See the
        // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
        // methods supported on all messages.

        var version: CoreML_Specification_CoreMLModels_VisionFeaturePrint.Scene.SceneVersion = .invalid

        var unknownFields = SwiftProtobuf.UnknownStorage()

        enum SceneVersion: SwiftProtobuf.Enum {
            typealias RawValue = Int
            case invalid // = 0

            /// VERSION_1 is available on iOS,tvOS 12.0+, macOS 10.14+
            /// It uses a 299x299 input image and yields a 2048 float feature vector
            case sceneVersion1 // = 1
            case UNRECOGNIZED(Int)

            init() {
                self = .invalid
            }

            init?(rawValue: Int) {
                switch rawValue {
                case 0: self = .invalid
                case 1: self = .sceneVersion1
                default: self = .UNRECOGNIZED(rawValue)
                }
            }

            var rawValue: Int {
                switch self {
                case .invalid: return 0
                case .sceneVersion1: return 1
                case let .UNRECOGNIZED(i): return i
                }
            }
        }

        init() {}
    }

    /// Objects extracts features useful for identifying and localizing
    /// objects in natural images
    struct Objects {
        // SwiftProtobuf.Message conformance is added in an extension below. See the
        // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
        // methods supported on all messages.

        var version: CoreML_Specification_CoreMLModels_VisionFeaturePrint.Objects.ObjectsVersion = .invalid

        ///
        /// Stores the names of the output features according to the
        /// order of them being computed from the neural network, i.e.,
        /// the first element in the output is the earliest being
        /// computed, while the last is the latest being computed. In
        /// general, the order reflects the resolution of the feature.
        /// The earlier it is computed, the higher the feature resolution.
        var output: [String] = []

        var unknownFields = SwiftProtobuf.UnknownStorage()

        enum ObjectsVersion: SwiftProtobuf.Enum {
            typealias RawValue = Int
            case invalid // = 0

            /// VERSION_1 is available on iOS,tvOS 14.0+, macOS 11.0+
            /// It uses a 299x299 input image and yields two multiarray
            /// features: one at high resolution of shape (288, 35, 35)
            /// the other at low resolution of shape (768, 17, 17)
            case objectsVersion1 // = 1
            case UNRECOGNIZED(Int)

            init() {
                self = .invalid
            }

            init?(rawValue: Int) {
                switch rawValue {
                case 0: self = .invalid
                case 1: self = .objectsVersion1
                default: self = .UNRECOGNIZED(rawValue)
                }
            }

            var rawValue: Int {
                switch self {
                case .invalid: return 0
                case .objectsVersion1: return 1
                case let .UNRECOGNIZED(i): return i
                }
            }
        }

        init() {}
    }

    init() {}
}

#if swift(>=4.2)

    extension CoreML_Specification_CoreMLModels_VisionFeaturePrint.Scene.SceneVersion: CaseIterable {
        // The compiler won't synthesize support with the UNRECOGNIZED case.
        static var allCases: [CoreML_Specification_CoreMLModels_VisionFeaturePrint.Scene.SceneVersion] = [
            .invalid,
            .sceneVersion1,
        ]
    }

    extension CoreML_Specification_CoreMLModels_VisionFeaturePrint.Objects.ObjectsVersion: CaseIterable {
        // The compiler won't synthesize support with the UNRECOGNIZED case.
        static var allCases: [CoreML_Specification_CoreMLModels_VisionFeaturePrint.Objects.ObjectsVersion] = [
            .invalid,
            .objectsVersion1,
        ]
    }

#endif // swift(>=4.2)

// MARK: - Code below here is support for the SwiftProtobuf runtime.

private let _protobuf_package = "CoreML.Specification.CoreMLModels"

extension CoreML_Specification_CoreMLModels_VisionFeaturePrint: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
    static let protoMessageName: String = _protobuf_package + ".VisionFeaturePrint"
    static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
        20: .same(proto: "scene"),
        21: .same(proto: "objects"),
    ]

    mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
        while let fieldNumber = try decoder.nextFieldNumber() {
            // The use of inline closures is to circumvent an issue where the compiler
            // allocates stack space for every case branch when no optimizations are
            // enabled. https://github.com/apple/swift-protobuf/issues/1034
            switch fieldNumber {
            case 20: try {
                    var v: CoreML_Specification_CoreMLModels_VisionFeaturePrint.Scene?
                    var hadOneofValue = false
                    if let current = self.visionFeaturePrintType {
                        hadOneofValue = true
                        if case let .scene(m) = current { v = m }
                    }
                    try decoder.decodeSingularMessageField(value: &v)
                    if let v = v {
                        if hadOneofValue { try decoder.handleConflictingOneOf() }
                        self.visionFeaturePrintType = .scene(v)
                    }
                }()
            case 21: try {
                    var v: CoreML_Specification_CoreMLModels_VisionFeaturePrint.Objects?
                    var hadOneofValue = false
                    if let current = self.visionFeaturePrintType {
                        hadOneofValue = true
                        if case let .objects(m) = current { v = m }
                    }
                    try decoder.decodeSingularMessageField(value: &v)
                    if let v = v {
                        if hadOneofValue { try decoder.handleConflictingOneOf() }
                        self.visionFeaturePrintType = .objects(v)
                    }
                }()
            default: break
            }
        }
    }

    func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
        // The use of inline closures is to circumvent an issue where the compiler
        // allocates stack space for every if/case branch local when no optimizations
        // are enabled. https://github.com/apple/swift-protobuf/issues/1034 and
        // https://github.com/apple/swift-protobuf/issues/1182
        switch visionFeaturePrintType {
        case .scene?: try {
                guard case let .scene(v)? = self.visionFeaturePrintType else { preconditionFailure() }
                try visitor.visitSingularMessageField(value: v, fieldNumber: 20)
            }()
        case .objects?: try {
                guard case let .objects(v)? = self.visionFeaturePrintType else { preconditionFailure() }
                try visitor.visitSingularMessageField(value: v, fieldNumber: 21)
            }()
        case nil: break
        }
        try unknownFields.traverse(visitor: &visitor)
    }

    static func == (lhs: CoreML_Specification_CoreMLModels_VisionFeaturePrint, rhs: CoreML_Specification_CoreMLModels_VisionFeaturePrint) -> Bool {
        if lhs.visionFeaturePrintType != rhs.visionFeaturePrintType { return false }
        if lhs.unknownFields != rhs.unknownFields { return false }
        return true
    }
}

extension CoreML_Specification_CoreMLModels_VisionFeaturePrint.Scene: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
    static let protoMessageName: String = CoreML_Specification_CoreMLModels_VisionFeaturePrint.protoMessageName + ".Scene"
    static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
        1: .same(proto: "version"),
    ]

    mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
        while let fieldNumber = try decoder.nextFieldNumber() {
            // The use of inline closures is to circumvent an issue where the compiler
            // allocates stack space for every case branch when no optimizations are
            // enabled. https://github.com/apple/swift-protobuf/issues/1034
            switch fieldNumber {
            case 1: try try decoder.decodeSingularEnumField(value: &version)
            default: break
            }
        }
    }

    func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
        if version != .invalid {
            try visitor.visitSingularEnumField(value: version, fieldNumber: 1)
        }
        try unknownFields.traverse(visitor: &visitor)
    }

    static func == (lhs: CoreML_Specification_CoreMLModels_VisionFeaturePrint.Scene, rhs: CoreML_Specification_CoreMLModels_VisionFeaturePrint.Scene) -> Bool {
        if lhs.version != rhs.version { return false }
        if lhs.unknownFields != rhs.unknownFields { return false }
        return true
    }
}

extension CoreML_Specification_CoreMLModels_VisionFeaturePrint.Scene.SceneVersion: SwiftProtobuf._ProtoNameProviding {
    static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
        0: .same(proto: "SCENE_VERSION_INVALID"),
        1: .same(proto: "SCENE_VERSION_1"),
    ]
}

extension CoreML_Specification_CoreMLModels_VisionFeaturePrint.Objects: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
    static let protoMessageName: String = CoreML_Specification_CoreMLModels_VisionFeaturePrint.protoMessageName + ".Objects"
    static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
        1: .same(proto: "version"),
        100: .same(proto: "output"),
    ]

    mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
        while let fieldNumber = try decoder.nextFieldNumber() {
            // The use of inline closures is to circumvent an issue where the compiler
            // allocates stack space for every case branch when no optimizations are
            // enabled. https://github.com/apple/swift-protobuf/issues/1034
            switch fieldNumber {
            case 1: try try decoder.decodeSingularEnumField(value: &version)
            case 100: try try decoder.decodeRepeatedStringField(value: &output)
            default: break
            }
        }
    }

    func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
        if version != .invalid {
            try visitor.visitSingularEnumField(value: version, fieldNumber: 1)
        }
        if !output.isEmpty {
            try visitor.visitRepeatedStringField(value: output, fieldNumber: 100)
        }
        try unknownFields.traverse(visitor: &visitor)
    }

    static func == (lhs: CoreML_Specification_CoreMLModels_VisionFeaturePrint.Objects, rhs: CoreML_Specification_CoreMLModels_VisionFeaturePrint.Objects) -> Bool {
        if lhs.version != rhs.version { return false }
        if lhs.output != rhs.output { return false }
        if lhs.unknownFields != rhs.unknownFields { return false }
        return true
    }
}

extension CoreML_Specification_CoreMLModels_VisionFeaturePrint.Objects.ObjectsVersion: SwiftProtobuf._ProtoNameProviding {
    static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
        0: .same(proto: "OBJECTS_VERSION_INVALID"),
        1: .same(proto: "OBJECTS_VERSION_1"),
    ]
}