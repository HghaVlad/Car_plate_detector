# Car number plate detector and recogniser

## RU

#### Детектор номерных знаков 

##### Инструкция

1. Скачать репозиторий: `git clone https://github.com/HghaVlad/Car_plate_detector`

2. Установить библиотеки: `pip3 install -r requirements.txt`

3. Запустить приложение: `python3 app.py`

4. Выберите режим, который будете использовать

##### Обнаружение на одном изображении(Debug mode)

1. В файле [main.py](main.py) изменить путь к исходному изображению `ORGINAL_IMG_PATH`

2. Запустить `python3 main.py`

3. В папке [images](images) появятся изображения номеров и новое фото(`frame_end.png`) с подписанными и выделенными номерами

#### Использование камеры компьютера для real-time detection

1. В файле [realtime.py](realtime.py) изменить настройки камеры

2. Запустить `python3 realtime.py`

#### Использование записанного видео для создания нового видеоряда

1. В файле [process_video.py](process_video.py) изменить путь к исходному видео `ORGINAL_VIDEO_PATH`

2. Запустить `python3 process_video.py`