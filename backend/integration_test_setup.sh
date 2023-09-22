pip install -r requirements.txt
python manage.py migrate
mkdir static
touch static/cifar10.tflite
cat seed.py | python manage.py shell
