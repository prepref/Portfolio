# OTTO - многоцелевая рекомендательная система
## Цель  
Предсказать следующие действия для пользователя
## Библиотеки  
- Pandas
- Numpy
## Главная метрика оценки качества модели  
Recall@20
## Модель  
Нам надо предсказать, на что кликнет, что положить в корзину и что закажет пользователь
- Все что лежит в корзине, пользователь вероятно закажет
- Все на что кликнул пользователь, вероятно заинтересовало его, и это может оказаться в корзине
- Если пользователь кликал на какой-либо товар, то мы можем ему предложить тот же товар, и вероятно он кликнет на него снова
- Найти 20 самых продаваемых товаров
  
Для каждого действия clicks, carts и orders мы также рекомендуем дополнительно 20 самых продаваемых товаров
