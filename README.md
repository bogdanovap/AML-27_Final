# Итоговая работа по AML-27

## Описание
Модель Deep Q-Learning для торговли криптовалютой.

В ходе обучения модель принимает решения о покупке, продаже и удержании криптовалюты.
Целевая функция - итоговая доходность от торговли на момент завершения обучения.
Проход модели завершается по 2м условиям:
- превышение максимално допустимой просадки по средствам
- достижение конца периода обучения

Если эпизод завершается при достижении конечного периода, то выполнятеся тестирование модели на тестовом периоде.

Критерий завершения обучения: достижение целевого уровня доходности на тестовом периоде.

Класс и функции для работы модели реализованы в main_DQN.py, в том числе:
- QModel
- Learn
- Test
- Load model
- Save model

Для модели сформировна торговая среда - TradingEnv, в которой реализовны все действия, необходимые для подготовки данных, обучения и тестирвоания модели.

Подготовка данных для обучения и тестирования производится через терминал Meta Trader 5.0.
Такой подход выбран для того, чтобы далее упросить интеграцию непосредственно торговой стратегии и модели в единую систему.
Базовый скрипт для подготовки данных находится в папке MQL5 Script.

## Примеры готовых расчетов

Примеры обученых моделей:
 - ETHUSDT
 - BTCUSDT
 - BNBUSDT

Все примеры обучались по одному сценарию:
- начало периода - 01.01.2018, свечи 1 день и 4 часа
- обучение на первых 1500 свечей
- тестирвоание на следующих 500 свечей (фактически, тестирвоание было выполнено на примерно 380 свечей)

Каждый пример включает:
 - файлы торговой среды и модели
 - данные на которых выполнялось обучение и тестирование
 - ноутбук с результатами тестирования

