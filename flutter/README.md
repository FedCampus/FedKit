# FedKit Flutter Package

WIP

## Development

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
