from detection import Transform, Detector, Trackers, Horizon, Vertical
from visualization import overlap, rectangle
import matplotlib.pyplot as plt
import cv2


#===================#
# TARGET            #
#===================#

# вводим путь до видео
video         = ''


#===================#
# MODELS            #
#===================#

# yolov4 работает не на всех версиях cv2
# более легкая, но менее точная модель
# weights       = 'yolo/yolov4-tiny.weights'
# config        = 'yolo/yolov4-tiny.cfg'

weights       = 'yolo/yolov4.weights'
config        = 'yolo/yolov4.cfg'

# Тут определяем объекты, которые хотим детектировать
# Нужно задать номера из файла yolo/coco.json
# Например: 2 = легковые автомобили, 5 = автобусы, 7 = грузовики
target        = [2, 5, 7]

# Это один из самых выжных параметров. Он отвечает за частоту
# использования детектора. Например, если стоит число 25, то
# детектор срабатывает раз в 25 кадров. В остальное время работают
# только трекеры.
drop_n_frames = 25


#===================#
# INITIALIZATION    #
#===================#

detector      = Detector(weights, config, target=target)
capture       = cv2.VideoCapture(video)

# Детектор работает только с квадратными изображениями,
# Поэтому тут мы указываем часть изображения, за которой будем следить
fragment      = [300, 0, 1020, 720]

# Указываем размер исходных изображений (разрешение камеры)
input_shape   = [1280, 720]
transform     = Transform(input_shape, fragment=fragment)


#===================#
# TEXT              #
#===================#

# Только для отрисовки
bottom_org    = (50, 620)
top_org       = (50, 520)
font          = cv2.FONT_HERSHEY_SIMPLEX 
fontScale     = 1
fontColor     = (255, 255, 255)
thickness     = 2


#===================#
# BOUNDARY          #
#===================#

# Прямоугольная граница, за пересечением которой, мы следим
boundary      = [350, 50, 970, 670]

# Для вертикального детектирования нужно заменить Horizon на Vertical
boundary_checker = Horizon(boundary)

if isinstance(boundary_checker, Horizon):
    upward_label   = 'top'
    downward_label = 'bottom'
else:
    upward_label   = 'left'
    downward_label = 'right'


#===================#
# TRACKERS          #
#===================#

trackers      = Trackers()
num_of_frame  = 0
upward        = 0
downward      = 0


#===================#
# OUTPUT VIDEO      #
#===================#

# Формат и разрешение полученного видео
fourcc        = cv2.VideoWriter_fourcc(*'mp4v')
frame_rate    = 25.0
shape         = (1280, 720)
out           = cv2.VideoWriter('mall.mp4', fourcc, frame_rate, shape)


# Подсчет трафика и создане видео с обнаруженными объектами

while capture.isOpened():
    
    state, frame = capture.read()
    
    if state is False:
        break
    
    # Этап обнаружения и добавления в трекеров
    # т.к детектор тяжелый, производим проверку
    # один раз на каждые 'drop_n_frames' кадров.
    # В остальное время всю работу выполняет трекер
    if num_of_frame % drop_n_frames == 0:
        blob     = transform(frame)
        detected = detector(blob, threshold=0.6)
        update   = trackers.update(frame)
    
        for box in detected[1]:
            origin = transform.convert_to_origin(box)
            nested = boundary_checker.is_nested(origin)
            
            # Проверяем, находится ли объект в детектируемой области
            if not nested:
                continue
            
            # Проверяем, ведет ли какой-то трекер этот объект
            index = trackers.get_index(update, origin)
        
            if index == None:
                size    = trackers.convert_box_to_size(origin)
                
                # В разных случаях можно ипользовать разные виды трекеров
                # Если выдает ошибку, сторит установить расширенную версию cv2
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, size)
                trackers.add_tracker(tracker)
    
    # Передаем трекерам новый кадр (обновляем положения объектов)
    updates   = trackers.update(frame)
    
    # Отрисовывает рамки на видео
    over = overlap(frame, [0, 0, 1280, 50], white=False)
    over = overlap(over, [0, 50, 350, 670], white=False)
    over = overlap(over, [0, 670, 1280, 720], white=False)
    over = overlap(over, [970, 50, 1280, 670], white=False)

    for index, update in updates.items():
        status    = update[0]
        box       = update[1]
        direction = trackers.directions[index]
        
        # По одному кадру трудно определить направлени,
        # Поэтому для новых объектов direction = None
        if direction != None:
            
            # Проверка пересечения граничных линий
            cross  = boundary_checker.is_crossed(box, direction)
            status = cross[0]
            line   = cross[1]
            
            # Если пересек, изменяем счетчики
            if status is True:
                
                # Какую линию мы пересекли?
                # Для Vertical: left и right
                if line == upward_label:
                    upward += 1
                elif line == downward_label:
                    downward += 1
                
                # закрываем трекер
                trackers.drop_tracker(index)
                continue
        
        # Отрисовка границ объекта
        left, top, right, bottom = box
        over = cv2.rectangle(over, (left, top), (right, bottom), (8, 67, 226), 2)
    
    
    upward_text =   f'{upward_label}:   {upward}'
    over = cv2.putText(over, upward_text, top_org, font, fontScale, fontColor, thickness, cv2.LINE_AA)
    
    downward_text = f'{downward_label}: {downward}'
    over = cv2.putText(over, downward_text, bottom_org, font, fontScale, fontColor, thickness, cv2.LINE_AA)
    
    out.write(over)
    
    print('=======================')
    print(f'{upward_label}:    {upward}')
    print(f'{downward_label}:  {downward}')
    print('')
    
    num_of_frame += 1

# Результаты по видео
print('Сводка по видео')
print(f'{upward_label}: {upward}')
print(f'{downward_label}: {downward}')

# Сохраняем видео
out.release()