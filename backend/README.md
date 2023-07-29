# Backend Server

## Set up

### Install the required dependencies

```sh
python3 -m pip install -r requirements.txt
```

### Run migrations

```sh
python3 manage.py migrate
```

### Download TFLite models

Download the model file from <https://github.com/FedCampus/dyn_flower_android_drf/files/11858642/cifar10.zip> to `static/cifar10.tflite`.

- Alternatively, generate it using the script in `../gen_tflite/cifar10_eg/`.

### Create TFLite model in database

Pipe the seed script into a Django shell:

```sh
cat seed.py | python3 manage.py shell
```

The above script adds the CIFAR10 training data type and model to database.

## Development

To test on physical devices in development, run with

```sh
python3 manage.py runserver 0.0.0.0:8000
```

Find you local IP in your system settings for the physical device to connect to.

## Adding custom model

To add a new TFLite model to the backend database, follow the following steps:

1. Following `../gen_tflite/README.md`, generate the `.tflite` file.
1. Move the generated `.tflite` file to `static/`.
1. Use a script referencing `seed.py`.
    Replace `sizes` with the `Model parameter sizes in bytes` printout from the model generation script. Replace `d` with the proper `TrainingDataType`.

    ```python
    m = TFLiteModel(
        name="my_models_name",
        file_path="/static/my_model.tflite",
        layers_sizes=sizes,
        data_type=d,
    )
    m.save()
    ```

## Docker Deployment

This project can be deployed using Docker, which ensures that it will run the same way on every machine, regardless of the local setup.

### Pull the Docker Image

- Note that the latest tag may not be avaliable before the PR gets merged, but it will come soon enough that I put it in here to minimize further modification of the README file

The Docker image for this project is hosted and actively updated on Docker Hub via the CI/CD pipeline. To pull the image to your local machine, run the following command:

```sh
docker pull fedcampus/dyn_flower_android_drf:latest
```

### Prepare the model

To prepare the model, follow the **"Download TFLite models"** and use `pwd` to get the directory of `/static`, we need to use that later

### Run the Docker Image

Before running the Docker image, please ensure that the models are located on local machine as bind mount will be used to pass the tflite files to the docker image to allow different models to be used.

To run the Docker image, use the `docker run` command along with the appropriate port bindings. The application uses ports 8000 and 8080. Here's how you can run the Docker image:

```sh
docker run --name name-of-your-choice \
-p 8000:8000 \
-p 8080:8080 \
-v /path/to/static:/app/static \
fedcampus/dyn_flower_android_drf:latest
```

This command maps port 8000 (the value on the right) inside the Docker container to port 8000 (the value on the left) on your host machine, and does the same for port 8080.

The part behind `-v` is used for bind mount (Docker Volume is also supported, to use Docker Volume, replace the left side of the bind mount with the name of the Volume)

### Testing with Docker

Once the Docker container is running, you can test the application by sending POST requests to the paths `/train/server` and `/train/get_advertised` on your localhost, using ports 8000 or 8080, depending on the service you're trying to reach. You can use curl or another tool to send these requests.

For example, to make a POST request to `/train/server` using curl, you would run the following command:

```sh
curl -X POST http://localhost:8000/train/server
```

And to make a POST request to `/train/advertised`, you would run:

```sh
curl -X POST http://localhost:8000/train/advertised
```

Remember to replace `localhost` and `8000` with the appropriate values if you're running Docker on a different host or port.
