# FedKit Flutter Package

WIP

## Code generation

This is needed if either the ProtoBuf or `@freezed` portion changed.

### Set up for ProtoBuf

[Install `protoc`](https://grpc.io/docs/protoc-installation/).

Install `protoc` Dart plugin:

```sh
dart pub global activate protoc_plugin
export PATH="$PATH":"$HOME/.pub-cache/bin"
```

### Generate JSON parsing code and ProtoBuf code

```sh
sh gen_code.sh
```
