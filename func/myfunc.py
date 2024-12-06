import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics, model_selection
from sklearn.base import TransformerMixin, BaseEstimator
import pickle

# Создадим функцию для получения значений из поля характеристик    
def get_characters(characters):        
    """Функция принимает строку с характеристиками автомобиля. Преобразует её в список, 
    разбивая по знакам табуляции. Пытается найти индексы соответсвующих характеристик 
    и увеличив их на единицу получает значения этих характеристик. Если индекс не найден 
    в значение характеристики записывается 'None'. Функция возвращает полученные значения 
    характеристик в виде кортежа.
    Args:
        characters (str): исходная строка с характеристиками
    
    Returns:
        typle: кортеж с полученными значениями характеристик
    """
    # Разделяем строку по знаку табуляции в спмсок
    characters = characters.split('\n')
    
    # Пытаемся получить значение категории характеристики выполняя поиск индекса наименования
    # категории и присваивая соответствующей переменной значение следующего за наименованием 
    # категории характеристики элемента. Если категория не обнаружена - переменной присваивается 
    # значение 'None' 
    try:
        availability = characters[characters.index('Наличие')+1]
    except:
        availability = 'None'
        
    try:
        generation = characters[characters.index('Поколение')+1]
    except:
        generation = 'None'
        
    try:
        year = characters[characters.index('Год выпуска')+1]
    except:
        year = 'None'
        
    try:
        mileage = characters[characters.index('Пробег')+1]
    except:
        mileage = 'None'
        
    try:
        body = characters[characters.index('Кузов')+1]
    except:
        body = 'None'
        
    try:
        color = characters[characters.index('Цвет')+1]    
    except:
        color = 'None'
        
    try:
        engine = characters[characters.index('Двигатель')+1] 
    except:
        engine = 'None'
        
    try:
        equipment = characters[characters.index('Комплектация')+1]
    except:
        equipment = 'None'
        
    try:
        tax = characters[characters.index('Налог')+1]
    except:
        tax = 'None'
        
    try:
        transmission = characters[characters.index('Коробка')+1]
    except:
        transmission = 'None'
        
    try:
        drive = characters[characters.index('Привод')+1]
    except:
        drive = 'None'
        
    try:
        steering_wheel = characters[characters.index('Руль')+1]
    except:
        steering_wheel = 'None'
        
    try:
        condition = characters[characters.index('Состояние')+1]
    except:
        condition = 'None'
        
    try:
        owners  = characters[characters.index('Владельцы')+1]
    except:
        owners = 'None'
        
    try:
        pts = characters[characters.index('ПТС')+1]
    except:
        pts = 'None'
        
    try:
        possession = characters[characters.index('Владение')+1]
    except:
        possession = 'None'
        
    try:
        customs =  characters[characters.index('Таможня')+1]
    except:
        customs = 'None'
        
    try:
        exchange = characters[characters.index('Обмен')+1]
    except:
        exchange = 'None'
        
    try:    
        vin = characters[characters.index('VIN')+1]
    except:
        vin = 'None'
        
    try:
        state_number = characters[characters.index('Госномер')+1]
    except:
        state_number = 'None'
    
    # Формируем кортеж из значений категорий характеристик
    result = (availability,
                generation,
                year,
                mileage,
                body,
                color,
                engine,
                equipment,
                tax,
                transmission,
                drive,
                steering_wheel,
                condition,
                owners,
                pts,
                possession,
                customs,
                exchange,
                vin,
                state_number
                )
    
    # Возвращаем результат
    return result


# Функция для разделения признака 'engine' на составляющие: мощность, объем и тип двигателя.
def get_engine(eng_str):
    """Функция принимает на вход строку с характеристиками двигателя
    разбивает ее на список по символу "/". Перебирает полученный список.
    В каждом элементе выполняется поиск подстроки содержащей мощность в 
    лошадиных силах или тип двигателя. При нахождении этих подстрок элемент 
    списка заносится в заранее определенную переменную. Для мощности в 
    лошадиных силах выпоняется преобразование в число. Если подстроки не 
    обнаружены - элемент заносится в переменную обозначающую объем 
    двигателя (так сделано потому что в данных присутстсвуют в том 
    числе автомобили с электрическим двигателем)

    Args:
        eng_str (str): строка с характеристиками двигателя
    Returns:
        tuple: кортеж с результатами разделения характеристик двигателя
    """
    # Определяем переменные для хранения извлеченных данных
    eng_hpower = 'None'
    eng_vol = 'None'
    eng_type = 'None'
    
    # В цикле перебираем список полученный разбиением строки по знаку табуляции
    for elem in eng_str.split('/'):
        
        # Ищем подстроку "л.с."
        if re.search('л.с.', elem.lower()):
            
            # Если нашли - заносим в переменную значение в виде числа, 
            # отбросив размерность
            eng_hpower = int(elem.strip().split(' ')[0])
            
        # Ищем подстроку с типом двигателя 
        elif re.search('гибрид|дизель|электро|бензин|газ', elem.lower()):
            
            # Если нашли - записываем в соответствующую переменную
            eng_type = elem.strip()
        
        # Если поиск по этим условиям не дал результата - значит текущий элемент 
        # содержит информацию об объеме двигателя для двигателей внутреннего сгорания 
        # или мощность в КВт для электрических двигателей
        else:
            eng_vol = elem.strip()
            
    # возвращаем значения переменных в виде кортежа
    return eng_hpower, eng_vol, eng_type


# Функция для извлечения марки автомобиля.
def get_brand(info_str):
    """Функция принимает на вход строку из поля info. Разбивает её на список 
    по знаку табуляции. Первый элемент этого списка разбивается в список по знаку 
    пробела. Функция сравнивает первый элемент полученного списка с ключами заранее
    определнного словаря, содержащего составные наименования марок машин. Если ключ
    найден - возвращается значение из словаря, если нет - первый элемент полученного 
    списка.

    Args:
        info_str (str): _description_

    Returns:
        str: наименование марки автомобиля
    """
    # Задаем словарь для марок автомобилей, состоящих из двух слов
    brand_dict = {'Land':'Land Rover',
                'Great':'Great Wall',
                'Lynk':'Lynk & Co',
                'Alfa':'Alfa Romeo',
                'Iran':'Iran Khodro',
                'Aston':'Aston Martin',
                'DW':'DW Hower'}
    
    # Проверяем являетсяли нулевой элемент списка ключом нашего словаря
    if info_str.split('\n')[0].split(' ')[0] in list(brand_dict.keys()):
        
        # Если да - вощвращаем значение из словаря по ключу
        return brand_dict[info_str.split('\n')[0].split(' ')[0]]
    
    # Иначе - нулевой элемент возвращается как наименование марки автомобиля
    else:
        return info_str.split('\n')[0].split(' ')[0]
    
    
    
# Функция для извлечения наименования модели
def get_model(data):
    """Функция принимает на вход строку датафрейма. Преобразует поле info в список 
    по знаку табуляции. Первый элемент этого списка разбивается на список по знаку
    пробела и заносится в переменную info. Функция создает списки brand и generation
    разбив однноименные поля строки датафрейма по знаку пробела. Далее создается 
    список model содержащий элементы отсутствующие в списках brand и generation.
    Функция возвращает объединный через знак пробела в строку итоговый список model.  

    Args:
        df (pd.Series): строка датафрейма

    Returns:
        str: наименование модели автомобиля
    """
    # Формируем список info содержащий марку, поколение и модель автомобиля
    info = data['info'].split('\n')[0].split(' ')
    
    # Создаем списки разбив значения строк с маркой и поколением автомобиля по знаку пробела
    brand = data['brand'].split(' ')
    generation = data['generation'].split(' ')
    
    # В данных есть наименование модели, которое полностью присутствует в наименовании поколения. 
    # Такой случай всего один, обработаем его отдельно
    if info == 'Fiat Punto III Grande Punto'.split(' '):
        return info[1]
    
    # Формируем список model оставляя только те элементы списка info, 
    # которые отсутствуют в списке brand
    else:
        model = [item for item in info if item not in brand]
    
        # Переопределяем список  model оставляя только те элементы, которые отсутствуют 
        # в списке generation 
        model = [item for item in model if item not in generation]
    
        # Возвращаем строку с наименованием модели, объединив элементы списка model через знак пробела
        return ' '.join(model)



# Функция для получения номера объявления 
def get_id(head_str):
    """Функция принимает на вход строку из поля head.
    Преобразует её в список по знаку табуляции. В цикле перебираем список и выполняем проверку
    каждого первого элемента для текущего значения. Если оно является знаком № - значит это тот 
    элемент списка, который мы ищем. Возвращаем его значение.

    Args:
        head_str (str): исходная строка с информацией

    Returns:
        str: номер объявления
    """
    # Перебираем элементы списка, полученные разбиением строки из поля "head" 
    # по знаку табуляции
    for elem in (head_str.split('\n')):         
        # Проверяем первый элемент текущего значения на соответствие условию
        if elem[0]=='№':
            # Возращаем текущее значение, если условие выполнено
            return elem
      


# Функция для восстановления пропущенных значений пробега автомобиля
def restore_mileage(df):
    """Функция принимает на вход строку датафрейма. Проверяет значение в поле
    с пробегом. Если пробег (признак 'mileage') не указан - строка из поля 'info' 
    разбивается на список по знаку табуляции. Элементы списка перебираются в цикле. 
    На каждой итерации цикла выполняется поиск подстроки 'км'. В случае нахождения 
    функция возращает этот элемент списка. Если пробег был указан в признаке 'mileage'
    функция вернет это значение.

    Args:
        df (pd.series): строка датафрейма

    Returns:
       str: полученное из поля 'info' значение пробега
    """
    # Проверяем, что значение пробега отсутствует
    if df['mileage'] == 'None':
        
        # Перебираем элементы списка, полученные разбиением строки поля 'info'
        # по знаку табуляции
        for elem in df['info'].split('\n'):
            
            # Каждый элемент разбиваем в список по знаку пробела и ищем подстроку 'км', 
            # а также отсутствие подстроки "Заряд" т.к. для электромобилей указывается еще 
            # запас хода в км
            if 'Заряд' not in elem.split() and 'км' in elem.split(): 
                
                # При нахождении - возвращаем текущий элемент
                return elem
            
    # Если пробег уже указан возвращаем его текущее значение
    else:
        return df['mileage']



# Функция для разделения поля 'options' по категориям
def get_cat_options(options_str):
    """Функция принимает на вход строку с опциями. В строке содержаться 
    фиксированные наименования категорий опций и сами опции. Функция разбивает строку 
    на список по знаку табуляции и создает список с индексами наименований категорий
    опций из исходного списка. Далее в цикле перебирается список индексов и для соответствующей 
    категории опции в заранее определенную переменную записываются сами опции через срез исходного
    списка по полученным индексам. Функция возвращает значения переменных (сами опции) в виде кортежа

    Args:
        options_str (str): исходная строка содержащая катеогрии опций и сами опции

    Returns:
        tuple: значения опций распределенный по категориям
    """
    # Определяем переменные для категорий опций
    option_media = 'Не указано'
    option_interior = 'Не указано'
    option_exterior = 'Не указано'
    option_visibility = 'Не указано'
    option_safety = 'Не указано'
    option_comfort ='Не указано'
    option_protection = 'Не указано'
    option_other = 'Не указано'
    
    # Задаем список возможных категорий
    cat_ls = ['Мультимедиа',
              'Салон',
              'Элементы экстерьера',
              'Обзор',
              'Безопасность',
              'Комфорт',
              'Защита от угона',
              'Прочее'
              ]
    
    # Если опции не указаны возвращаем заранее определенные значения 
    # для каждой категории опций - 'Не указано'
    if options_str is np.nan:        
        return [option_media, 
                option_interior,
                option_exterior,
                option_visibility,
                option_safety,
                option_comfort,
                option_protection,
                option_other
                ]
        
    else:
        # Иначе разделяем строку по знаку табуляции в список опций
        options_ls = options_str.split('\n')
        
        # Записываем найденные индекс и наименование категории в отдельный список  
        ls_ind = [[i, x] for i, x in enumerate(options_ls) if x in cat_ls]
       
        # В цикле проходим по всем найденным категориям из отдельного списка    
        for num, elem in enumerate(ls_ind):         
                                    
            # Проверяем текущее наименование категории и записываем значение 
            # в соответствующую переменную с помощью среза для случаев нахождения 
            # категории в теле или в конце списка опций. И удаляем лишний символ '•'.          
            if elem[1] == 'Мультимедиа':
                try:
                    option_media = [i for i in options_ls[elem[0]+1:ls_ind[num+1][0]] if i != '•']
                except:
                    option_media = [i for i in options_ls[elem[0]+1:] if i != '•']
                                
            elif elem[1] == 'Салон':
                try:
                    option_interior = [i for i in options_ls[elem[0]+1:ls_ind[num+1][0]] if i != '•']
                except:
                    option_interior = [i for i in options_ls[elem[0]+1:] if i != '•']
                    
            elif elem[1] == 'Элементы экстерьера':
                try:
                    option_exterior = [i for i in options_ls[elem[0]+1:ls_ind[num+1][0]] if i != '•']
                except:
                    option_exterior = [i for i in options_ls[elem[0]+1:] if i != '•']
            
            elif elem[1] == 'Обзор':
                try:
                    option_visibility = [i for i in options_ls[elem[0]+1:ls_ind[num+1][0]] if i != '•']
                except:
                    option_visibility = [i for i in options_ls[elem[0]+1:] if i != '•']
                    
            elif elem[1] == 'Безопасность':
                try:
                    option_safety = [i for i in options_ls[elem[0]+1:ls_ind[num+1][0]] if i != '•']
                except:
                    option_safety = [i for i in options_ls[elem[0]+1:] if i != '•']
            
            elif elem[1] == 'Комфорт':
                try:
                    option_comfort = [i for i in options_ls[elem[0]+1:ls_ind[num+1][0]] if i != '•']
                except:
                    option_comfort = [i for i in options_ls[elem[0]+1:] if i != '•']
            
            elif elem[1] == 'Защита от угона':
                try:
                    option_protection = [i for i in options_ls[elem[0]+1:ls_ind[num+1][0]] if i != '•']
                except:
                    option_protection = [i for i in options_ls[elem[0]+1:] if i != '•']
                    
            elif elem[1] == 'Прочее':
                try:
                    option_other = [i for i in options_ls[elem[0]+1:ls_ind[num+1][0]] if i != '•']
                except:
                    option_other = [i for i in options_ls[elem[0]+1:] if i != '•']
                  
        # Возвращаем найденные значения различных категорий опций
        return option_media, option_interior, option_exterior, option_visibility,\
            option_safety, option_comfort, option_protection, option_other
                

# Функция для извлечения опций из категории безопасность.
def get_safety(cat_options):
    """Функция принимает на вход список с опциями из категории "безопасность".   
    В цикле перебираются элементы полученного списка. На каждой итерации выполняется
    проверка на соответствие конкретной опции. В случае нахождения соответствия текущий
    элемент записывается в заранее определенную переменную (по умолчанию все переменные 
    имеют значение 'Не указано'), описывающую конкретную опцию. Функция возвращает кортеж с 
    полученными значениями переменных (опций). 
    Args:
        cat_options (list): список с опциями конкретной категории

    Returns:
        tuple: кортеж со значениями опций в конкретной категории
    """
    airbags_driv = 'Не указано'
    airbags_pass = 'Не указано'
    airbags_side = 'Не указано'
    airbags_s_rear = 'Не указано'
    airbags_knees_driv = 'Не указано'
    airbags_knees_pass = 'Не указано'
    airbags_wind = 'Не указано'
    collision_warn = 'Не указано'
    collision_avoid = 'Не указано'
    lane_depart_warn = 'Не указано'
    lane_holds = 'Не указано'
    traffic_jam_assist = 'Не указано'
    driv_fatigue_sens = 'Не указано'
    road_sign_recogn = 'Не указано'
    asr = 'Не указано'
    vsm = 'Не указано'
    bas_ebd  = 'Не указано'
    hill_start_assist = 'Не указано'
    descent_assist = 'Не указано'
    blind_spot_monitor = 'Не указано'
    assist_revers_parking = 'Не указано'
    night_vision = 'Не указано'
    isofix_rear = 'Не указано'
    isofix_front = 'Не указано'    
    pressure_sensor = 'Не указано'
    abs_sys = 'Не указано'
    esp = 'Не указано'
    lockout = 'Не указано'
    era_glonass = 'Не указано'
    armored_body = 'Не указано'
        
    for elem in cat_options:
        
        # Проверяем, что в текущей строке есть значения
        if elem != 'Не указано':               
            if re.search("Подушка безопасности водителя", elem):
                airbags_driv = elem
            
            elif re.search("Подушка безопасности пассажира", elem):
                airbags_pass = elem 
            
            # При точном поиске дополнительно ищем вариант с символом "•"    
            elif (elem == 'Подушки безопасности боковые') or (elem == '•Подушки безопасности боковые'):
                airbags_side = elem                 
            
            elif re.search("Подушки безопасности боковые задние", elem):
                airbags_s_rear = elem  
            
            elif re.search("Подушка безопасности для защиты коленей водителя", elem):
                airbags_knees_driv = elem
            
            elif re.search("Подушка безопасности для защиты коленей пассажира", elem):
                airbags_knees_pass = elem
                
            elif re.search("шторки", elem):
                airbags_wind = elem   
            
            elif re.search('предупреждения о столкновении', elem.lower()):
                collision_warn = elem
            
            elif re.search('предотвращения столкновения', elem.lower()):            
                collision_avoid = elem
                
            elif re.search('предупреждения о выезде из полосы', elem.lower()):            
                lane_depart_warn = elem
                
            elif re.search('удержания в полосе', elem.lower()):
                lane_holds = elem
                
            elif re.search('движения в пробке', elem.lower()):
                traffic_jam_assist = elem 
                
            elif re.search('усталости водителя', elem.lower()):
                driv_fatigue_sens = elem
                
            elif re.search('дорожных знаков', elem.lower()):
                road_sign_recogn = elem
                
            elif re.search('антипробуксовочная', elem.lower()):
                asr = elem
                
            elif re.search('стабилизации рулевого управления|vsm', elem.lower()):
                vsm = elem
                
            elif re.search('распределения тормозных усилий|bas|ebd', elem.lower()):
                bas_ebd = elem
                
            elif re.search('при старте в гору', elem.lower()):
                hill_start_assist = elem
                
            elif re.search('при спуске', elem.lower()):
                descent_assist = elem
                
            elif re.search('слепых зон', elem.lower()):
                blind_spot_monitor = elem
                
            elif re.search('при выезде с парковки', elem.lower()):
                assist_revers_parking = elem
                
            elif re.search('ночного видения', elem.lower()):
                night_vision = elem  
        
            elif re.search('(задний ряд)', elem.lower()):
                isofix_rear = elem
            
            elif re.search('(передний ряд)', elem.lower()): 
                isofix_front = elem            
                
            elif re.search('давления в шинах', elem.lower()):
                pressure_sensor = elem
                
            elif re.search('abs', elem.lower()):
                abs_sys = elem
                
            elif re.search('esp', elem.lower()):
                esp = elem
                
            elif re.search('блокировка замков', elem.lower()):
                lockout = elem
                
            elif re.search('эра', elem.lower()):
                era_glonass = elem
                
            elif re.search('бронированный кузов', elem.lower()):
                armored_body = elem
                
    return  airbags_driv, airbags_pass, airbags_side, airbags_s_rear, airbags_knees_driv, airbags_knees_pass,\
        airbags_wind, collision_warn, collision_avoid, lane_depart_warn, lane_holds, traffic_jam_assist,\
        driv_fatigue_sens, road_sign_recogn, asr, vsm, bas_ebd, hill_start_assist, descent_assist,\
        blind_spot_monitor, assist_revers_parking, night_vision, isofix_rear, isofix_front, pressure_sensor,\
        abs_sys, esp, lockout, era_glonass, armored_body
        
        
        
# Функция для извлечения опций из категории "обзор".
def get_visibility(cat_options):
    """Функция принимает на вход список с опциями из категории "обзор".   
    В цикле перебираются элементы полученного списка. На каждой итерации выполняется
    проверка на соответствие конкретной опции. В случае нахождения соответствия текущий
    элемент записывается в заранее определенную переменную (по умолчанию все переменные 
    имеют значение 'Не указано'), описывающую конкретную опцию. Функция возвращает кортеж с 
    полученными значениями переменных (опций). 
    Args:
        cat_options (list): список с опциями конкретной категории

    Returns:
        tuple: кортеж со значениями опций в конкретной категории
    """

    laser_headlights = 'Не указано'
    led_headlights = 'Не указано'
    xenon_headlights = 'Не указано'
    heat_wind_wipers = 'Не указано'
    heat_wind = 'Не указано'
    heat_mirrors = 'Не указано'
    heat_wind_washer = 'Не указано'        
    anti_frog = 'Не указано'
    adaptive_light = 'Не указано'
    rain_sens = 'Не указано'
    light_sens = 'Не указано'
    day_lights = 'Не указано'
    auto_headlight_corr = 'Не указано'
    auto_high_beam = 'Не указано'
    headlight_washer = 'Не указано'
        
    for elem in cat_options:
        
        # Проверяем, что в текущей строке есть значения
        if elem != 'Не указано':  
            if re.search('лазер', elem.lower()):
                laser_headlights = elem 
            
            elif re.search('диод', elem.lower()):    
                led_headlights  = elem 
                
            elif re.search('ксенон', elem.lower()):    
                xenon_headlights = elem           
            
            elif re.search('стеклоочист', elem.lower()):
                heat_wind_wipers = elem
                
            elif re.search('лобового', elem.lower()):
                heat_wind = elem
                
            elif re.search('боков', elem.lower()):
                heat_mirrors = elem
                
            elif re.search('стеклоомыв', elem.lower()):
                heat_wind_washer = elem
                
            elif re.search('противотуман', elem.lower()):
                anti_frog = elem
                
            elif re.search('адаптив', elem.lower()):
                adaptive_light = elem
                
            elif re.search('дожд', elem.lower()):
                rain_sens = elem
                
            elif re.search('датчик свет', elem.lower()):
                light_sens = elem
                
            elif re.search('дневные', elem.lower()):
                day_lights = elem
                
            elif re.search('корректор', elem.lower()):
                auto_headlight_corr = elem
                
            elif re.search('дальн', elem.lower()):
                auto_high_beam = elem
                
            elif re.search('омыват', elem.lower()):
                headlight_washer = elem
          
    return  laser_headlights, led_headlights, xenon_headlights, heat_wind_wipers, heat_wind, heat_mirrors,\
        heat_wind_washer, anti_frog, adaptive_light, rain_sens, light_sens, day_lights, auto_headlight_corr,\
        auto_high_beam, headlight_washer

       
        
# Функция для извлечения опций из категории "мультимедиа"
def get_media(cat_options):
    """Функция принимает на вход список с опциями из категории "мультимедиа".   
    В цикле перебираются элементы полученного списка. На каждой итерации выполняется
    проверка на соответствие конкретной опции. В случае нахождения соответствия текущий
    элемент записывается в заранее определенную переменную (по умолчанию все переменные 
    имеют значение 'Не указано'), описывающую конкретную опцию. Функция возвращает кортеж с 
    полученными значениями переменных (опций). 
    Args:
        cat_options (list): список с опциями конкретной категории

    Returns:
        tuple: кортеж со значениями опций в конкретной категории
    """
    premium_audio = 'Не указано'
    audio = 'Не указано'
    audio_prep = 'Не указано'    
    ru_menu = 'Не указано'
    lcd_screen = 'Не указано'
    media_rear_pass = 'Не указано'
    remote_control = 'Не указано'
    wireless_charg = 'Не указано'
    usb = 'Не указано'
    navi_sys = 'Не указано'
    voice_control = 'Не указано'
    android_auto = 'Не указано'
    carplay = 'Не указано'
    yandex_auto = 'Не указано'
    aux = 'Не указано'
    bluetooth = 'Не указано'
    socket_12v = 'Не указано'
    socket_220v = 'Не указано'    
        
    for elem in cat_options:
        
        # Проверяем, что в текущей строке есть значения
        if elem != 'Не указано':
            
            # При точом поиске будем искать дополнительный вариант с символом "•"
            if elem == 'Премиальная аудиосистема' or elem == '•Премиальная аудиосистема':
                premium_audio = elem
                
            elif elem == 'Аудиосистема' or elem == '•Аудиосистема':
                audio = elem
                
            elif elem == 'Аудиоподготовка' or elem == '•Аудиоподготовка':
                audio_prep = elem
                
            elif re.search('русиф', elem.lower()):
                ru_menu = elem
                
            elif re.search('экран', elem.lower()):
                lcd_screen = elem
                
            elif re.search('задних', elem.lower()):
                media_rear_pass = elem
                
            elif re.search('дистанц', elem.lower()):
                remote_control = elem
                
            elif re.search('заряд', elem.lower()):
                wireless_charg = elem
                
            elif re.search('usb', elem.lower()):
                usb = elem
                
            elif re.search('навигац', elem.lower()):
                navi_sys = elem
                
            elif re.search('голосов', elem.lower()):
                voice_control = elem
                
            elif re.search('android', elem.lower()):
                android_auto = elem
                
            elif re.search('carplay', elem.lower()):
                carplay = elem
                
            elif re.search('яндекс', elem.lower()):
                yandex_auto = elem
                
            elif re.search('aux', elem.lower()):
                aux = elem
                
            elif re.search('bluetooth', elem.lower()):
                bluetooth = elem
                
            elif re.search('12', elem.lower()):
                socket_12v = elem
                
            elif re.search('220', elem.lower()):
                socket_220v = elem
                      
    return premium_audio, audio, audio_prep, ru_menu, lcd_screen, media_rear_pass, remote_control, wireless_charg,\
        usb, navi_sys, voice_control, android_auto, carplay, yandex_auto, aux, bluetooth,\
        socket_12v, socket_220v

   
   
# Функция для извлечения опций из категории "комфорт"
def get_comfort(cat_options):
    """Функция принимает на вход список с опциями из категории "комфорт".   
    В цикле перебираются элементы полученного списка. На каждой итерации выполняется
    проверка на соответствие конкретной опции. В случае нахождения соответствия текущий
    элемент записывается в заранее определенную переменную (по умолчанию все переменные 
    имеют значение 'Не указано'), описывающую конкретную опцию. Функция возвращает кортеж с 
    полученными значениями переменных (опций). 
    Args:
        cat_options (list): список с опциями конкретной категории

    Returns:
        tuple: кортеж со значениями опций в конкретной категории
    """
    climate_1_zone = 'Не указано'
    climate_2_zone = 'Не указано'
    climate_m_zone = 'Не указано'
    air_cond = 'Не указано'
    cam_360 = 'Не указано'
    cam_front = 'Не указано'
    cam_rear = 'Не указано'
    e_wind_font = 'Не указано'
    e_wind_rear = 'Не указано'
    cruise_control = 'Не указано'
    a_cruise_control = 'Не указано'
    s_wheel_adj_height = 'Не указано'
    s_wheel_adj_reach = 'Не указано'
    s_wheel_adj_elect = 'Не указано'
    s_wheel_adj_mem = 'Не указано'
    amplifier_wheel = 'Не указано'
    a_amplifier_wheel = 'Не указано'
    front_park = 'Не указано'
    rear_park = 'Не указано'
    auto_park = 'Не указано'
    head_up_disp = 'Не указано'
    drive_mode = 'Не указано'
    remote_start = 'Не указано'
    trunk_without_hands = 'Не указано'
    multifunc_wheel = 'Не указано'
    on_board_comp = 'Не указано'
    electr_instr_panel = 'Не указано'
    keyless_entry_sys = 'Не указано'
    engine_button = 'Не указано'
    prog_pre_heater = 'Не указано'
    electric_trunk = 'Не указано'
    electr_fold_mirror = 'Не указано'
    mirror_memory = 'Не указано'
    paddle_shifters = 'Не указано'
    pedal_assembly = 'Не указано'
    start_stop_sys = 'Не указано'
    electric_mirror = 'Не указано'
    glove_box = 'Не указано'
    door_closer = 'Не указано'
    lighter_ashtray = 'Не указано'
    
    for elem in cat_options:
        
        # Проверяем, что в текущей строке есть значения
        if elem != 'Не указано':
            
            if re.search('1-зонный', elem.lower()):
                climate_1_zone = elem
                
            elif re.search('2-зонный', elem.lower()):
                climate_2_zone = elem
                
            elif re.search('многозонный', elem.lower()):
                climate_m_zone = elem
                
            elif re.search('конд', elem.lower()):
                air_cond = elem            
                    
            elif elem == 'Камера 360°' or elem == '•Камера 360°':
                cam_360 = elem
            
            elif elem == 'Камера передняя' or elem == '•Камера передняя':
                cam_front = elem
            
            elif elem == 'Камера задняя' or elem == '•Камера задняя':
                cam_rear = elem               
                                
            elif elem == 'Электростеклоподъемники передние' or elem == '•Электростеклоподъемники передние':
                e_wind_font = elem
            
            elif elem == 'Электростеклоподъемники задние' or elem == '•Электростеклоподъемники задние':
                e_wind_rear = elem
            
            elif elem == 'Круиз-контроль' or elem == '•Круиз-контроль':
                cruise_control = elem
                
            elif elem == 'Адаптивный круиз-контроль' or elem == '•Адаптивный круиз-контроль':
                a_cruise_control  = elem
        
            elif elem == 'Регулировка руля по высоте' or elem == '•Регулировка руля по высоте':
                s_wheel_adj_height = elem
                
            elif elem == 'Регулировка руля по вылету' or elem == '•Регулировка руля по вылету':
                s_wheel_adj_reach = elem
                
            elif elem == 'Электрорегулировка руля' or elem == '•Электрорегулировка руля':
                s_wheel_adj_elect = elem
                
            elif elem == 'Рулевая колонка с памятью положения' or elem == '•Рулевая колонка с памятью положения':
                s_wheel_adj_mem = elem
            
            elif elem == 'Усилитель руля' or elem == '•Усилитель руля':
                amplifier_wheel = elem
            
            elif elem == 'Активный усилитель руля' or elem == '•Активный усилитель руля':
                a_amplifier_wheel = elem
            
            elif elem == 'Парктроник передний' or elem == '•Парктроник передний':
                front_park = elem
                
            elif elem == 'Парктроник задний' or elem == '•Парктроник задний':
                rear_park = elem
            
            elif elem == 'Система автоматической парковки' or elem == '•Система автоматической парковки':
                auto_park = elem       
                
            elif re.search('проекц', elem.lower()):
                head_up_disp = elem
                
            elif re.search('выбор', elem.lower()):
                drive_mode = elem
                
            elif re.search('дистан', elem.lower()):
                remote_start = elem
                
            elif re.search('без помощи рук', elem.lower()): 
                trunk_without_hands = elem
                
            elif re.search('мультифунк', elem.lower()):
                multifunc_wheel = elem
                
            elif re.search('компьютер', elem.lower()):
                on_board_comp = elem  
                
            elif re.search('приборн', elem.lower()):
                electr_instr_panel = elem
                
            elif re.search('без ключ', elem.lower()):
                keyless_entry_sys = elem
                
            elif re.search('кноп', elem.lower()): 
                engine_button = elem
                
            elif re.search('отопит', elem.lower()):
                prog_pre_heater = elem
                
            elif re.search('привод крышки', elem.lower()):
                electric_trunk = elem
                
            elif re.search('электроскладывание', elem.lower()):    
                electr_fold_mirror = elem
                
            elif re.search('память боковых зеркал', elem.lower()):
                mirror_memory = elem
                
            elif re.search('лепестки', elem.lower()):
                paddle_shifters = elem
                
            elif re.search('педаль', elem.lower()):
                pedal_assembly = elem
                
            elif re.search('старт-стоп', elem.lower()):
                start_stop_sys = elem
                
            elif re.search('электропривод зеркал', elem.lower()):
                electric_mirror = elem
                
            elif re.search('перчат', elem.lower()):
                glove_box = elem
                
            elif re.search('доводчик дверей', elem.lower()):
                door_closer = elem
                
            elif re.search('пепельница', elem.lower()):
                lighter_ashtray = elem
            
    return climate_1_zone, climate_2_zone, climate_m_zone, air_cond, cam_360, cam_front, cam_rear,\
        e_wind_font, e_wind_rear, cruise_control, a_cruise_control, s_wheel_adj_height, s_wheel_adj_reach,\
        s_wheel_adj_elect, s_wheel_adj_mem, amplifier_wheel, a_amplifier_wheel, front_park, rear_park,\
        auto_park, head_up_disp, drive_mode, remote_start, trunk_without_hands, multifunc_wheel, on_board_comp,\
        electr_instr_panel, keyless_entry_sys, engine_button, prog_pre_heater, electric_trunk, electr_fold_mirror,\
        mirror_memory, paddle_shifters, pedal_assembly, start_stop_sys, electric_mirror, glove_box, door_closer,\
        lighter_ashtray

  
        
# Функция для извлечения опций из категории "экстерьер"
def get_exterior(cat_options):
    """Функция принимает на вход список с опциями из категории "экстерьер".   
    В цикле перебираются элементы полученного списка. На каждой итерации выполняется
    проверка на соответствие конкретной опции. В случае нахождения соответствия текущий
    элемент записывается в заранее определенную переменную (по умолчанию все переменные 
    имеют значение 'Не указано'), описывающую конкретную опцию. Функция возвращает кортеж с 
    полученными значениями переменных (опций). 
    Args:
        cat_options (list): список с опциями конкретной категории

    Returns:
        tuple: кортеж со значениями опций в конкретной категории
    """
    d_type_steel = 'Не указано'
    d_type_light = 'Не указано'   
    d_size_12 = 'Не указано'
    d_size_13 = 'Не указано'
    d_size_14 = 'Не указано'
    d_size_15 = 'Не указано'
    d_size_16 = 'Не указано'
    d_size_17 = 'Не указано'
    d_size_18 = 'Не указано'
    d_size_19 = 'Не указано'
    d_size_20 = 'Не указано'
    d_size_21 = 'Не указано'
    d_size_22 = 'Не указано'
    d_size_23 = 'Не указано'
    d_size_24 = 'Не указано'
    d_size_25 = 'Не указано'
    d_size_26 = 'Не указано'
    d_size_27 = 'Не указано'
    d_size_28 = 'Не указано'    
    two_paint = 'Не указано'
    body_kit = 'Не указано'
    roof_rails = 'Не указано'
    airbrush = 'Не указано'
    moldings = 'Не указано'
    
    for elem in cat_options:
        
        # Проверяем, что в текущей строке есть значения
        if elem != 'Не указано':
            
            if re.search('стальные', elem.lower()):            
                d_type_steel = elem
                
            elif re.search('легкосплавные', elem.lower()):
                d_type_light = elem
                
            elif re.search('12', elem.lower()):    
                d_size_12 = elem 
                
            elif re.search('13', elem.lower()):
                d_size_13 = elem 
                
            elif re.search('14', elem.lower()):
                d_size_14 = elem 
                
            elif re.search('15', elem.lower()):
                d_size_15 = elem 
            
            elif re.search('16', elem.lower()):
                d_size_16 = elem 
                
            elif re.search('17', elem.lower()):
                d_size_17 = elem 
                
            elif re.search('18', elem.lower()):
                d_size_18 = elem 
            
            elif re.search('19', elem.lower()):
                d_size_19 = elem 
            
            elif re.search('20', elem.lower()):
                d_size_20 = elem 
                
            elif re.search('21', elem.lower()):
                d_size_21 = elem 
                
            elif re.search('22', elem.lower()):
                d_size_22 = elem 
                
            elif re.search('23', elem.lower()):
                d_size_23 = elem 
                
            elif re.search('24', elem.lower()):
                d_size_24 = elem 
            
            elif re.search('25', elem.lower()):
                d_size_25 = elem 
                
            elif re.search('26', elem.lower()):
                d_size_26 = elem 
                
            elif re.search('27', elem.lower()):
                d_size_27 = elem 
            
            elif re.search('28', elem.lower()):
                d_size_28 = elem 
                
            elif re.search('окраска кузова', elem.lower()):
                two_paint = elem
            
            elif re.search('обвес кузова', elem.lower()):
                body_kit = elem
            
            elif re.search('рейлинг', elem.lower()):
                roof_rails = elem
                
            elif re.search('аэрография', elem.lower()):
                airbrush = elem
            
            elif re.search('молдинг', elem.lower()):
                moldings = elem    
  
    return d_type_steel, d_type_light, d_size_12, d_size_13, d_size_14, d_size_15, d_size_16, d_size_17, d_size_18,\
        d_size_19, d_size_20, d_size_21, d_size_22, d_size_23, d_size_24, d_size_25, d_size_26, d_size_27, d_size_28,\
        two_paint, body_kit, roof_rails, airbrush, moldings

        
        
# Функция для извлечения опций из категории "защита от угона"
def get_protection(cat_options):
    """Функция принимает на вход список с опциями из категории "защита от угона".   
    В цикле перебираются элементы полученного списка. На каждой итерации выполняется
    проверка на соответствие конкретной опции. В случае нахождения соответствия текущий
    элемент записывается в заранее определенную переменную (по умолчанию все переменные 
    имеют значение 'Не указано'), описывающую конкретную опцию. Функция возвращает кортеж с 
    полученными значениями переменных (опций). 
    Args:
        cat_options (list): список с опциями конкретной категории

    Returns:
        tuple: кортеж со значениями опций в конкретной категории
    """    
    signaling = 'Не указано'
    fback_signaling = 'Не указано'
    central_lock = 'Не указано'
    vol_sens = 'Не указано'
    immobilizer = 'Не указано' 
        
    for elem in cat_options:
        
        # Проверяем, что в текущей строке есть значения
        if elem != 'Не указано':
            
            if elem == 'Сигнализация' or elem == '•Сигнализация':
                signaling = elem
            
            elif elem == 'Сигнализация с обратной связью' or elem == '•Сигнализация с обратной связью':
                fback_signaling = elem            
                
            elif re.search('центральный замок', elem.lower()):
                central_lock = elem
            
            elif re.search('проникновения в салон', elem.lower()):
                vol_sens = elem
            
            elif re.search('иммобилайзер', elem.lower()):
                immobilizer = elem

    return signaling, fback_signaling, central_lock, vol_sens, immobilizer



# Функция для извлечения опций из категории "интерьер"
def get_interior(cat_options):
    """Функция принимает на вход список с опциями из категории "интерьер".   
    В цикле перебираются элементы полученного списка. На каждой итерации выполняется
    проверка на соответствие конкретной опции. В случае нахождения соответствия текущий
    элемент записывается в заранее определенную переменную (по умолчанию все переменные 
    имеют значение 'Не указано'), описывающую конкретную опцию. Функция возвращает кортеж с 
    полученными значениями переменных (опций). 
    Args:
        cat_options (list): список с опциями конкретной категории

    Returns:
        tuple: кортеж со значениями опций в конкретной категории
    """
    in_color_light = 'Не указано'
    in_color_dark = 'Не указано'    
    in_mat_leather  = 'Не указано'
    in_mat_combi = 'Не указано'
    in_mat_artif_l = 'Не указано'
    in_mat_fabric = 'Не указано'
    in_mat_velour = 'Не указано'
    in_mat_alcantara = 'Не указано'    
    seat_drive_h_adj = 'Не указано'
    seat_front_h_adj = 'Не указано'
    seat_drive_e_adj = 'Не указано'
    seat_front_e_adj = 'Не указано'
    seat_rear_e_adj = 'Не указано'
    drive_seat_mem = 'Не указано'
    front_seat_mem = 'Не указано'    
    drive_lumbar_sup = 'Не указано'
    front_lumbar_sup = 'Не указано'    
    front_seat_heat = 'Не указано'
    rear_seat_heat = 'Не указано'    
    front_seat_vent = 'Не указано'
    rear_seat_vent = 'Не указано'    
    seat_count_2 = 'Не указано'
    seat_count_4 = 'Не указано'
    seat_count_5 = 'Не указано'
    seat_count_6 = 'Не указано'
    seat_count_7 = 'Не указано'
    seat_count_8 = 'Не указано'
    seat_count_9 = 'Не указано'    
    sport_seat = 'Не указано'
    luke = 'Не указано'
    panoramic = 'Не указано'
    massage_seat = 'Не указано'
    heated_wheel = 'Не указано'
    leather_wheel = 'Не указано'
    leather_lever = 'Не указано'
    black_ceiling = 'Не указано'
    rear_armrest = 'Не указано'
    third_row_seats = 'Не указано'
    fold_rear_seat = 'Не указано'
    backrest_fold = 'Не указано'
    fold_table = 'Не указано'
    in_light = 'Не указано'
    f_center_armrest = 'Не указано'
    rear_headrest = 'Не указано'
    tinted_glass = 'Не указано'
    shades_rear_doors = 'Не указано'
    shades_rear_window = 'Не указано'
    pedal_pads = 'Не указано'
    door_sills = 'Не указано'

    for elem in cat_options: 
        
        # Проверяем, что в текущей строке есть значения
        if elem != 'Не указано':  
                    
            if re.search('светлый', elem.lower()):
                in_color_light = elem
                
            elif re.search('темный', elem.lower()):
                in_color_dark = elem
                    
            elif elem == 'Кожа (материал салона)' or elem == '•Кожа (материал салона)':
                in_mat_leather = elem
            
            elif re.search('комбинированный', elem.lower()):
                in_mat_combi = elem
            
            elif re.search('искусственная кожа', elem.lower()):
                in_mat_artif_l = elem
                
            elif re.search('ткань', elem.lower()):
                in_mat_fabric = elem
                
            elif re.search('велюр', elem.lower()):
                in_mat_velour = elem
                
            elif re.search('алькантара', elem.lower()):
                in_mat_alcantara = elem               
    
            elif re.search('регулировка сиденья водителя по высоте', elem.lower()):
                seat_drive_h_adj = elem
                
            elif re.search('регулировка передних сидений по высоте', elem.lower()):
                seat_front_h_adj = elem
                
            elif re.search('электрорегулировка сиденья водителя', elem.lower()):
                seat_drive_e_adj = elem
                
            elif re.search('электрорегулировка передних сидений', elem.lower()):
                seat_front_e_adj = elem
                
            elif re.search('электрорегулировка задних сидений', elem.lower()):
                seat_rear_e_adj = elem   
            
            elif re.search('память сиденья водителя', elem.lower()):
                drive_seat_mem = elem
                
            elif re.search('память передних сидений', elem.lower()):
                front_seat_mem = elem
                
            elif re.search('сиденье водителя с поясничной поддержкой', elem.lower()):
                drive_lumbar_sup = elem
            
            elif re.search('передние сиденья с поясничной поддержкой', elem.lower()):
                front_lumbar_sup = elem
            
            elif re.search('подогрев передних сидений', elem.lower()):
                front_seat_heat = elem
                
            elif re.search('подогрев задних сидений', elem.lower()):
                rear_seat_heat = elem
                
            elif re.search('вентиляция передних сидений', elem.lower()):
                front_seat_vent = elem
            
            elif re.search('вентиляция задних сидений', elem.lower()):
                rear_seat_vent = elem
                
            elif re.search('количество мест: 2', elem.lower()):
                seat_count_2 = elem
                
            elif re.search('количество мест: 4', elem.lower()):
                seat_count_4 = elem
                
            elif re.search('количество мест: 5', elem.lower()):
                seat_count_5 = elem
                
            elif re.search('количество мест: 6', elem.lower()):
                seat_count_6 = elem
                
            elif re.search('количество мест: 7', elem.lower()):
                seat_count_7 = elem
                
            elif re.search('количество мест: 8', elem.lower()):
                seat_count_8 = elem
            
            elif re.search('количество мест: 9', elem.lower()):
                seat_count_9 = elem         
                    
            elif re.search('спортив', elem.lower()):
                sport_seat = elem
            
            elif re.search('люк', elem.lower()):
                luke = elem
            
            elif re.search('панорамн', elem.lower()):
                panoramic = elem
            
            elif re.search('массаж', elem.lower()):
                massage_seat = elem
            
            elif re.search('обогрев рулевого колеса', elem.lower()):
                heated_wheel = elem
            
            elif re.search('кожей рулевого', elem.lower()):
                leather_wheel = elem
                
            elif re.search('кожей рычага', elem.lower()):
                leather_lever = elem
                
            elif re.search('потолка черного цвета', elem.lower()):
                black_ceiling = elem
                
            elif re.search('задний подлокотник', elem.lower()):
                rear_armrest = elem
                
            elif re.search('ряд сидений', elem.lower()):
                third_row_seats = elem
                
            elif re.search('складывающееся заднее сиденье', elem.lower()):
                fold_rear_seat = elem
                
            elif re.search('складывания спинки', elem.lower()):
                backrest_fold = elem
            
            elif re.search('складной столик', elem.lower()):
                fold_table = elem
                
            elif re.search('подсветка салона', elem.lower()):
                in_light = elem
            
            elif re.search('передний центральный подлокотник', elem.lower()):
                f_center_armrest = elem
            
            elif re.search('третий задний подголовник', elem.lower()):
                rear_headrest = elem
            
            elif re.search('тонированные стекла', elem.lower()):
                tinted_glass = elem
                
            elif re.search('солнцезащитные шторки в задних дверях', elem.lower()):
                shades_rear_doors = elem
            
            elif re.search('солнцезащитная шторка на заднем стекле', elem.lower()):
                shades_rear_window = elem
            
            elif re.search('накладки на педали', elem.lower()):
                pedal_pads = elem
            
            elif re.search('накладки на пороги', elem.lower()):
                door_sills = elem

    return in_color_light, in_color_dark, in_mat_leather, in_mat_combi, in_mat_artif_l, in_mat_fabric,\
        in_mat_velour, in_mat_alcantara, seat_drive_h_adj, seat_front_h_adj, seat_drive_e_adj,\
        seat_front_e_adj, seat_rear_e_adj, drive_seat_mem, front_seat_mem, drive_lumbar_sup, \
        front_lumbar_sup, front_seat_heat, rear_seat_heat, front_seat_vent, rear_seat_vent,\
        seat_count_2, seat_count_4, seat_count_5, seat_count_6, seat_count_7, seat_count_8,\
        seat_count_9, sport_seat, luke, panoramic,   massage_seat, heated_wheel, leather_wheel,\
        leather_lever, black_ceiling, rear_armrest, third_row_seats, fold_rear_seat, backrest_fold,\
        fold_table, in_light, f_center_armrest, rear_headrest, tinted_glass, shades_rear_doors,\
        shades_rear_window, pedal_pads, door_sills



# Функция для извлечения опций из категории "прочее"
def get_other(cat_options):
    """Функция принимает на вход список с опциями из категории "прочее".   
    В цикле перебираются элементы полученного списка. На каждой итерации выполняется
    проверка на соответствие конкретной опции. В случае нахождения соответствия текущий
    элемент записывается в заранее определенную переменную (по умолчанию все переменные 
    имеют значение 'Не указано'), описывающую конкретную опцию. Функция возвращает кортеж с 
    полученными значениями переменных (опций). 
    Args:
        cat_options (list): список с опциями конкретной категории

    Returns:
        tuple: кортеж со значениями опций в конкретной категории
    """
    active_suspension = 'Не указано'    
    sport_suspension = 'Не указано' 
    air_suspension = 'Не указано'
    full_spare_wheel = 'Не указано'  
    spare_wheel = 'Не указано'    
    towbar = 'Не указано'
    crankcase_prot = 'Не указано'
    
    for elem in cat_options:
        
        # Проверяем, что в текущей строке есть значения
        if elem != 'Не указано':
              
            if re.search('активная', elem.lower()):
                active_suspension = elem
                
            elif re.search('спортивная', elem.lower()):
                sport_suspension = elem
                
            elif re.search('пневмо', elem.lower()):
                air_suspension  = elem
                
            elif re.search('колесо', elem.lower()):    
                full_spare_wheel = elem
                
            elif re.search('докатка', elem.lower()): 
                spare_wheel = elem     
                        
            elif re.search('фаркоп', elem.lower()):
                towbar = elem
            
            elif re.search('картер', elem.lower()):
                crankcase_prot = elem

    return active_suspension, sport_suspension, air_suspension, full_spare_wheel, spare_wheel, towbar, crankcase_prot




# Функция для подсчета количества опций автомомбиля, указанных в объявлении
def get_count_options(df):
    """Функция принимает на вход строку датасета. Удаляет столбцы с категориями
    опций. В цикле перебирает значения опций. Если текущее значение опции не является
    ни нулем  ни "Не указано" - увеличивает счетчик опций на единицу. Функция 
    возвращает количество указанных в объявлении опций автомобиля.

    Args:
        df (Serries): строка датасета с опциями автомобиля

    Returns:
        int: количество опций автомобиля
    """
    # Задаем счетчик количества опций
    count = 0
    
    # Перебираем значения опций исключив признаки с категориями опций
    for elem in df.drop(['option_media',
                         'option_interior',
                         'option_exterior',
                         'option_visibility',
                         'option_safety',
                         'option_comfort',
                         'option_protection',
                         'option_other'
                         ]
                        ).values:
        
        # Если опция указана увеличиваем счетчик на единицу
        if elem != 'Не указано' and elem != 0:
            count += 1
    
    # Возвращаем результат
    return count  



# функция для принятия решения об отклонении нулевой гипотезы
def decision_hypothesis(p, alpha=0.05):
    """Функция принимает p-значение и выводит сообщение 
    о возможности принятия нулевой гипотезы. Значение 
    alpha опционально, по умолчанию задано равным 0.05

    Args:
        p (float): p-значение
        alpha (float, optional): уровень значимости, по умолчанию 0.05.
    """
    print('p-value = {:.3f}'.format(p))
    if p <= alpha:
        print('p-значение меньше, чем заданный уровень значимости {:.2f}. Отвергаем нулевую гипотезу в пользу альтернативной.'.format(alpha))
    else:
        print('p-значение больше, чем заданный уровень значимости {:.2f}. У нас нет оснований отвергнуть нулевую гипотезу.'.format(alpha))



# функция реализующая обработку выбросов методом межквартильных интервалов (метод Тьюки)
def outliers_irq(df, feature):
    """Функция принимает на вход датасет и имя признака, 
    в котором нужно провести обработку выбросов. Производится вычисление 1 и 3 квартилей
    для указанных в списке признаков. Находится межкварьтльный интервал как разность между 
    3ей и 1ой квартилями. Нижняя граница допустимых значений определяется как разность между
    1ой квартилей и 1.5 величиной межквартильного интервала. Верхняя - как сумма 3ей квартили 
    и 1.5 величиной межквартильного интервала. К входящему датасету применяется две маски.
    Для получения выбросов - берутся все значения выходящие за границы допустимых значений.
    Для очищенных данных - все значения находящиесй внутри границ допустимых значений. Функция 
    возвращает два датасета: первый - выбросы, второй - оцищенные данные
           
    Args:
        df (DataFrame): исходный датасет
        feature (list): список признаков для обработки выбросов

    Returns:
        DataFrame: датасет с выбросами, датасет с очищенными данными
    """
    # Выделяем интересующие нас признаки в исходном датасете
    x = df[feature]
    
    # Рассчитываем 1ый и 3ий квартили и опрделяем межвкартильный интервал
    quantile_1, quantile_3 = x.quantile(0.25), x.quantile(0.75)
    iqr = quantile_3 - quantile_1
    
    # Задаем границы допустимых значений
    low_bound = quantile_1 - (iqr*1.5)
    up_bound = quantile_3 + (iqr*1.5)
    
    # Формируем датасеты с выбросами и очищенными данными, применив границы 
    # допустимых значений в качестве макси к исходным данным
    outlaiers = df[(x<low_bound)|(x>up_bound)]
    cleaned =  df[(x>low_bound)&(x<up_bound)]
    
    # возвращаем результат
    return outlaiers, cleaned


def get_score(estimator, X, y):
    """Функция принимает на вход обученную на логарифмированных
    правильных ответах модель, матрицу наблюдений и вектор правильных 
    ответов в логарифмическом масштабе. Делает предсказание с помощью полученной 
    модели и приводит их в нормальный масштаб. Производит расчет средней абсолютной 
    процентной ошибки и возвращает результат.

    Args:
        estimator (object type that implements the 'predict' method): модель, обученная
        на логарифмированных правильных ответах. 
        X ({array-like, sparse matrix} of shape (n_samples, n_features)): матрица наблюдений.
        y (array-like of shape (n_samples,)): вектор правильных ответов в логарифмическом
        масштабе.

    Returns:
        float: Значение метрики МАРЕ расчитання для ответов в обычном масштабе.
    """
    # Делаем предсказания на обеих выборках для контроля переобуения модели
    pred = np.exp(estimator.predict(X))-1
    # Рассчитываем и выводим метрики
    score = metrics.mean_absolute_percentage_error(y_true=np.exp(y)-1, y_pred=pred)*100
    return score


def  plot_learn_curve(model, X, y, cv, scoring, ax=None, title=''):
    """Функция выводит график кривой обучения модели.

    Args:
        model (object type that implements the 'fit' method): модель, подвергаемая проверке.
        X ({array-like, sparse matrix} of shape (n_samples, n_features)): матрица наблюдений.
        y (array-like of shape (n_samples,)): вектор правильных ответов.
        cv (int or cross-validation generator): кросс-валидатор из бибилиотеки sklearn или 
        количество фолдов для разбиения выборки. По умолчанию используется кросс-валидация 
        k-fold на 5 фолдах.
        scoring (str or callable): название метрики в виде строки или функция для ее вычисления.
        ax (matplotlib Axes): оси для построения кривой обучения. По умолчанию - None.
        title (str, optional): Наименование модели для которой строится график. По умолчанию - ''.
    """
    # Вычисляем координаты для построения кривой обучения
    tr_size, tr_score, val_score = model_selection.learning_curve(estimator=model,
                                                                  X=X,
                                                                  y=y,
                                                                  scoring=scoring
                                                                  )
    
    # Ищем среднее значение по фолдам для каждого набора данных
    tr_score_mean = np.mean(tr_score, axis=1)
    val_score_mean = np.mean(val_score, axis=1)
    
    # Если координатная плоскость не задана - создаем свою
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,5))
    
    # Отображаем кривую изменения метрики для каждой выборки
    ax.plot(tr_size, tr_score_mean, label='Train')
    ax.plot(tr_size, val_score_mean, label='Valid')
    
    # Задаем наименование графика и осей
    ax.set_title(f'Learning curve for {title}')
    ax.set_xlabel('Train data size')
    ax.set_ylabel('MAPE score')
    
    # Задаем метки оси Х и отображаем легенду
    ax.xaxis.set_ticks(tr_size)
    ax.legend();
    
    
def get_score_mse(estimator, X, y):
        """Функция принимает на вход обученную на логарифмированных
        правильных ответах модель, матрицу наблюдений и вектор правильных 
        ответов в логарифмическом масштабе. Делает предсказание с помощью полученной 
        модели и приводит их в нормальный масштаб. Производит расчет средней квадратичной 
        ошибки и возвращает результат.

        Args:
            estimator (object type that implements the 'predict' method): модель, обученная
            на логарифмированных правильных ответах.
            X ({array-like, sparse matrix} of shape (n_samples, n_features)): матрица наблюдений.
            y (array-like of shape (n_samples,)): вектор правильных ответов в логарифмическом
            масштабе.

        Returns:
            float: Значение метрики МSЕ расчитання для ответов в обычном масштабе.
        """
        # Делаем предсказания на обеих выборках для контроля переобуения модели
        pred = np.exp(estimator.predict(X))-1
        # Рассчитываем и выводим метрики
        score = metrics.mean_squared_error(y_true=np.exp(y)-1, y_pred=pred)*100
        return score
    

# Создадим класс кастомного трансформера для получения координат по наименованию города
class sity_to_loc_transform(BaseEstimator, TransformerMixin):
    
    # Создаем функцию для инициализации объекта класса
    def __init__(self):
        pass
    
    # Создаем функцию обучения. Обучение нам не требуется, 
    # поэтому функция возвращает сам объект класса
    def fit(self, X, y=None):
        return self
        
    # Определяем функцию преобразования
    def transform(self, X, y=None):
        
        # Загружаем словарь с координатами из файла
        with open ('./file_pkl/coordinate.pkl','rb') as pkl_file:
            coordinate = pickle.load(pkl_file)
            
        # Применяем лямбда функцию к данным для получения координат  
        lat = X['sity'].map(lambda x: coordinate[x][0])
        long = X['sity'].map(lambda x: coordinate[x][1])
        subset = pd.DataFrame({'lat':lat,
                               'long':long}
                              )
        
        # Добавляем данные к исходному датасету 
        X = pd.concat([X, subset], axis=1)
                
        # Возвращаем преобразованный датасет
        return X
    

# Создадим класс кастомного трансформера для кодирования признаков
class encoder_transform(BaseEstimator, TransformerMixin):
    
    # Создаем функцию для инициализации объекта класса   
    def __init__(self):
        pass
    
    # Создаем функцию обучения. Обучение нам не требуется, 
    # поэтому функция возвращает сам объект класса
    def fit(self, X, y=None):
        return self
    
    # Определяем функцию преобразования    
    def transform(self, X, y=None):
        
        # Считываем имена признаков для однократного кодирования из файла
        with open ('./file_pkl/for_dummy_ls.pkl','rb') as pkl_file:
            for_dummy_ls = pickle.load(pkl_file)
                
        # Закодируем признаки из списка для однократного кодирования применив лямбда функцию.
        # Результат кодирования занесем в словарь dict_dummy
        dict_dummy = {}
        for i in for_dummy_ls:
            dict_dummy[i] = X[i].map(lambda x: 0 if x == 'Не указано' else 1)

        # Закодируем и разделим оставшиеся два признака ('steering_wheel', 'customs') на четыре,
        # аналогичной лямбда функцией
        dict_dummy['steering_wheel_l'] = X['steering_wheel'].map(lambda x: 1 if x == 'Левый' else 0)
        dict_dummy['steering_wheel_r'] = X['steering_wheel'].map(lambda x: 1 if x == 'Правый' else 0)
        dict_dummy['customs_pts'] = X['customs'].map(lambda x: 1 if x == 'Растаможен' else 0)
        dict_dummy['customs_no_pts'] = X['customs'].map(lambda x: 1 if x == 'Растаможен, нет ПТС' else 0)

        # Удалим из исходного датасета изначальные (мы их уже закодировали) признаки
        X.drop(for_dummy_ls, inplace=True, axis=1)
        X.drop(['steering_wheel', 'customs', 'sity'], inplace=True, axis=1)

        # Добавим закодированные признаки к исходному датасету
        X = pd.concat([X, pd.DataFrame(dict_dummy)], axis=1)   
        
        # Считываем bin_encoder из файла 
        with open ('./file_pkl/bin_encoder.pkl','rb') as pkl_file:
            bin_encoder = pickle.load(pkl_file)
        
        # Создаем список признаков для бинарного кодирования
        for_binary_ls = ['availability',
                         'transmission',
                         'drive',
                         'condition',
                         'owners',
                         'pts',
                         'eng_type',
                         'body',
                         'color',
                         'brand',
                         'model',
                         'generation'
                         ]
            
        # Кодируем признаки        
        data = bin_encoder.transform(X[for_binary_ls])
        
        # Удаляем исходные признаки и добавляем результат кодирования
        X = pd.concat([X.drop(for_binary_ls, axis=1), data], axis=1)
                
        # Возвращаем датасет с закодированными признаками
        return X
 
# Создадим класс кастомного трансформера для robust преобразования признаков
class robust_scaler(BaseEstimator, TransformerMixin):
    
    # Создаем функцию для инициализации объекта класса
    def __init__(self):
        pass
    
    # Создаем функцию обучения. Обучение нам не требуется, 
    # поэтому функция возвращает сам объект класса
    def fit(self, X, y=None):
        return self
        
    # Определяем функцию преобразования
    def transform(self, X, y=None):
        
        # Загружаем список признаков с высокой корреляцией из файла
        with open ('./file_pkl/feature_corr.pkl','rb') as pkl_file:
            feature_corr = pickle.load(pkl_file)
        
        # Удаляем признаки с высокой корреляцией из набора данных 
        X = X.drop(feature_corr, axis=1)
        
        # Загружаем список c порядком столбцов из файла
        with open ('./file_pkl/col_order.pkl','rb') as pkl_file:
            col_order = pickle.load(pkl_file)
        
        # Восстанавливаем порядок столбцов 
        X = X[col_order]
        
        # Загружаем обученный r_scaler из файла
        with open ('./file_pkl/r_scaler.pkl','rb') as pkl_file:
            r_scaler = pickle.load(pkl_file)
          
        # Нормализуем признаки
        X = pd.DataFrame(r_scaler.transform(X=X), columns=X.columns)
        
        # Возвращаем преобразованный датасет
        return X    
   
    
# Создадим класс кастомного трансформера для выбора признаков
class best_features(BaseEstimator, TransformerMixin):
    
    # Создаем функцию для инициализации объекта класса
    def __init__(self):
        pass
    
    # Создаем функцию обучения. Обучение нам не требуется, 
    # поэтому функция возвращает сам объект класса
    def fit(self, X, y=None):
        return self
        
    # Определяем функцию преобразования
    def transform(self, X, y=None):
        
        # Загружаем список с отобранными признаками из файла
        with open ('./file_pkl/best_dt_importance.pkl','rb') as pkl_file:
            best_dt_importance = pickle.load(pkl_file)
        
        # Возвращаем преобразованный датасет
        return X[best_dt_importance] 