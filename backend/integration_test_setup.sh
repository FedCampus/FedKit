pip install -r requirements.txt
pythob manage.py migrate
mkdir static
touch static/cifar10.tflite
cat seed.py | python manage.py shell
python manage.py runserver
