# Applicability of a ranking DSSM-model for Hierarchical Reinforcement Learning

Bachelor thesis of Kirill Tyshchuk at the Department of Mathematics and Computer Science, SPbSU

### Abstract

One of the intriguing open challenges in the field of Reinforcement Learning is the processing of expert demonstrations data with the purpose of more efficient training of new agents to perform similar tasks. One of the possible approaches is the extraction of macro-abstractions or patterns that are common to all of the demonstrated tasks. It is similar to a situation, in which we observe someone playing a game, get a rough concept of what is going on there and, when we get to play, learn very fast. In this work a method based on a ranking DSSM model is implemented along with the environment and instruments for studying its advantages and limitations. Taking into account the results of the analysis, the baseline model is refined. The resulting models show decent quality on a synthetic dataset.

# Применимость ранжирующей DSSM-модели для иерархического обучения с подкремпелнием

Бакалаврская дипломная работа Тыщука Кирилла на СПбГУ МКН

[Текст дипломной работы](https://github.com/Reason239/rl-dssm/blob/master/extra%20materials/Thesis_v2.pdf)

[Данные экспериментов на Comet.ml](https://www.comet.ml/reason239/gridworld-dssm/view/5BBniF9PhAd5KtWADLjcWCc5A)

### Аннотация

Одной из интересных открытых проблем из области обучения с подкреплением является обработка демонстраций эксперта, выполняющего определённую задачу, с целью более эффективного обучения новых агентов выполнять схожие задачи. Один из возможных подходов -- выделение из экспертных данных крупных абстракцией или шаблонов, общих между всеми задачами. Это подобно тому, как мы, понаблюдав за кем-то, занятым игрой, можем в общих чертах понять, что в ней происходит, и, когда нам доведётся играть самим, быстро разобраться в происходящем. В данной работе был разработан метод на основе ранжирующей модели DSSM, а также среда и инструменты для изучения его достоинств и недостатков. По итогам анализа, базовая модель была доработана. Полученные модели показали хорошее качество на синтетических данных.

### Описание файлов и папок

#### [datasets/](https://github.com/Reason239/rl-dssm/tree/master/datasets)

Директория, в которой хранятся сгенерированные датасеты.

#### [experiments/](https://github.com/Reason239/rl-dssm/tree/master/experiments)

Директория с данными экспериментов: сохранённые модели, визуализации и т. д.

#### [extra materials/](https://github.com/Reason239/rl-dssm/tree/master/extra%20materials)

Директория с дополнительными материалами: текст работы, ссылки на использованные материалы о OpenAI Gym.

#### [clustering.py](https://github.com/Reason239/rl-dssm/blob/master/clustering.py)

Скрипт для визуализации свойств кластеров эмбеддингов переходов базовой модели.

#### [dssm.py](https://github.com/Reason239/rl-dssm/blob/master/dssm.py)

Файл с реализацией базовой и квантизованной модели. Также содержит третью модель, находящуюся в разработке.

#### [generate_dataset.py](https://github.com/Reason239/rl-dssm/blob/master/generate_dataset.py)

Скрипт для создания датасетов для обучения, валидации или оценке на "синтетических" данных.

#### [gridworld.py](https://github.com/Reason239/rl-dssm/blob/master/gridworld.py)

Файл с реализацией тестовой среды, согласованной с интерфейсом OpenAI Gym.

####  [inspect_embeds.py](https://github.com/Reason239/rl-dssm/blob/master/inspect_embeds.py)

Скрипт для визуализации "кластеров" векторов, соответствующих каждому из векторов квантизации в квантизованной модели.

#### [main.py](https://github.com/Reason239/rl-dssm/blob/master/main.py)

Основной скрипт, который запускается для обучени одной модели или для проведения серии экспериментов. Параметры задаются вручную в этом файле. Обуение одной модели в течение 60 эпох занимает примерно 20-25 минут на CPU, 10-15 минут на GPU.

#### [next_state_prediction.py](https://github.com/Reason239/rl-dssm/blob/master/next_state_prediction.py)

Скрипт для визуализации предсказания следующих состояний моделью. Высодит состояние и топ возможных следующих состояний по оценку моделью. Не вошёл в тест диплома, потому что я зыбыл про него.

#### [play_env.py](https://github.com/Reason239/rl-dssm/blob/master/play_env.py)

Маленький скрипт для воспроизведения экспертной траектории в среде.

#### [train_dssm.py](https://github.com/Reason239/rl-dssm/blob/master/train_dssm.py)

Файл, в котором реализована процедура обучения модели. Содердит мастер-функцию, которую вызывает main.py.

#### [utils.py](https://github.com/Reason239/rl-dssm/blob/master/utils.py)

Файл с различными вспомогательными классами и функциями. 
