# Пример реализации KAN (Колмогоровско-Арнольдовской Сети)

В этом репозитории представлен пример **упрощенной** реализации нейронной сети типа KAN (Kolmogorov-Arnold Network) для задачи классификации. Код написан с использованием стандартных библиотек Python и демонстрирует базовую структуру KAN.

**Важное замечание**: Данная реализация является **демонстрационной** и **не включает этап обучения** параметров функций активации (например, узлов сплайнов или параметров RBF). Метод `fit` только инициализирует эти функции случайным образом на основе диапазона входных данных.

## Пример с KANClassifier (Пользовательская Реализация)

**Описание**:
Реализация простого классификатора `KANClassifier`, имитирующего структуру Колмогоровско-Арнольдовской Сети. Класс наследуется от базовых классов Scikit-learn (`BaseEstimator`, `ClassifierMixin`) для частичной совместимости с его API. Пример использует синтетические данные, сгенерированные с помощью `make_classification`.

**Особенности**:
- Демонстрация базовой структуры KAN: слои "внешних" (outer) и "внутренних" (inner) функций.
- Реализация с использованием `BaseEstimator` и `ClassifierMixin` для интеграции в Scikit-learn пайплайны (ограниченно).
- Возможность выбора типа активации: 'spline' (кубические сплайны на основе `scipy.interpolate.interp1d`) или 'rbf' (радиально-базисные функции).
- Функции активации инициализируются **случайно** и **не обучаются** в методе `fit`.
- Подходит для понимания концепции KAN, но не для практического применения без доработки (добавления обучения параметров).

**Требования**:
- Python 3.7+
- scikit-learn 1.0+ (используются базовые классы и утилиты)
- numpy 1.19+
- scipy 1.5+ (для `interp1d`)

**Документация и Ссылки**:
- **Концепция KAN (статья на arXiv)**: [Kolmogorov-Arnold Networks (2404.19756)](https://arxiv.org/abs/2404.19756)
- **Scipy `interp1d`**: [scipy.interpolate.interp1d](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html)
- **Базовые классы Scikit-learn**: [sklearn.base documentation](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.base)
- **Генератор данных `make_classification`**: [sklearn.datasets.make_classification](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html)