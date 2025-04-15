import numpy as np  # Для численных операций и работы с массивами
from sklearn.model_selection import train_test_split  # Для разделения данных
from sklearn.datasets import make_classification  # Для генерации синтетических данных
from sklearn.metrics import accuracy_score  # Для оценки точности модели
from sklearn.preprocessing import StandardScaler  # Для масштабирования данных
from scipy.interpolate import interp1d  # Для создания интерполяционных сплайн-функций
from sklearn.base import BaseEstimator, ClassifierMixin  # Базовые классы для совместимости с Scikit-learn

# Определение класса KANClassifier
# Наследуемся от BaseEstimator и ClassifierMixin для совместимости с инструментами Scikit-learn
# (например, GridSearchCV, Pipeline, стандартные методы fit/predict/score).
class KANClassifier(BaseEstimator, ClassifierMixin):
    """
    Упрощенный классификатор на основе Колмогоровско-Арнольдовских Сетей (KAN).

    Эта реализация использует фиксированные, случайно инициализированные
    функции активации (сплайны или RBF) и не обучает их параметры.
    Представляет собой базовую структуру KAN согласно теореме представления Колмогорова-Арнольда:
    f(x1, ..., xn) = Σ[q=0..2n] Φq ( Σ[p=1..n] ψq,p(xp) )
    Здесь ψ - "внешние" функции (outer), Φ - "внутренние" функции (inner).
    В данной реализации структура несколько упрощена для демонстрации.
    """
    def __init__(self, n_outer=10, n_inner=10, activation='spline', random_state=None):
        """
        Инициализация KAN классификатора.

        Параметры:
        - n_outer (int): Количество "внешних" функций активации ψ на каждый входной признак.
        - n_inner (int): Количество "внутренних" функций активации Φ.
        - activation (str): Тип используемых функций активации ('spline' для кубических сплайнов
                          или 'rbf' для радиально-базисных функций).
        - random_state (int): Сид для генератора случайных чисел для воспроизводимости
                              инициализации функций.
        """
        self.n_outer = n_outer  # Сохраняем количество внешних функций
        self.n_inner = n_inner  # Сохраняем количество внутренних функций
        self.activation = activation  # Сохраняем тип активации
        self.random_state = random_state  # Сохраняем сид случайности

    def _create_activation_functions(self, X, y):
        """
        Внутренний метод для создания и инициализации функций активации (ψ и Φ).
        Функции создаются на основе диапазона значений входных данных X.
        В этой реализации параметры функций (узлы сплайнов, центры RBF)
        инициализируются случайно и НЕ ОБУЧАЮТСЯ в методе `fit`.

        Параметры:
        - X (np.array): Обучающие данные (признаки).
        - y (np.array): Обучающие данные (метки классов) - в этой реализации не используются
                        напрямую для создания функций, но передаются по стандарту fit.
        """
        # Установка сида для воспроизводимости случайной инициализации
        np.random.seed(self.random_state)

        # Получение размерности данных
        n_samples, n_features = X.shape

        # Инициализация списков для хранения функций
        self.outer_functions = []  # Список списков: outer_functions[feature_idx][func_idx]
        self.inner_functions = []  # Список: inner_functions[func_idx]

        # --- Создание "внешних" функций (ψ) ---
        # Создаем свой набор из n_outer функций для каждого входного признака
        for feature_idx in range(n_features):
            feature_functions = []  # Список функций для текущего признака
            # Находим минимальное и максимальное значение для текущего признака во входных данных
            # Это нужно для определения диапазона, в котором будут работать функции активации
            feature_min = X[:, feature_idx].min()
            feature_max = X[:, feature_idx].max()

            # Генерируем n_outer функций для этого признака
            for _ in range(self.n_outer):
                if self.activation == 'spline':
                    # Создание кубического сплайна
                    # Генерируем 10 узловых точек по оси X в диапазоне значений признака
                    x_points = np.linspace(feature_min, feature_max, 10)
                    # Генерируем случайные значения Y для узловых точек (от -1 до 1)
                    y_points = np.random.uniform(-1, 1, 10)
                    # Создаем функцию кубической интерполяции с помощью scipy.interpolate.interp1d
                    # kind='cubic': тип сплайна
                    # fill_value='extrapolate': позволяет функции работать вне диапазона x_points
                    f = interp1d(x_points, y_points, kind='cubic', fill_value='extrapolate')
                else:  # активация 'rbf'
                    # Создание радиально-базисной функции (Гауссоида)
                    # Выбираем случайный центр RBF в диапазоне значений признака
                    center = np.random.uniform(feature_min, feature_max)
                    # Выбираем случайную ширину RBF (от 0.1 до 1.0)
                    width = np.random.uniform(0.1, 1.0)
                    # Определяем RBF как лямбда-функцию
                    f = lambda x, c=center, w=width: np.exp(-((x - c)**2) / (2 * w**2))
                # Добавляем созданную функцию в список для текущего признака
                feature_functions.append(f)
            # Добавляем список функций для текущего признака в общий список внешних функций
            self.outer_functions.append(feature_functions)

        # --- Создание "внутренних" функций (Φ) ---
        # Создаем n_inner внутренних функций
        for _ in range(self.n_inner):
            # Определяем ожидаемый диапазон входных значений для внутренних функций.
            # Вход для Φ - это сумма выходов ψ по всем признакам.
            # Грубая оценка диапазона: от -n_features * n_outer до n_features * n_outer
            # (если считать, что каждый ψ выдает значение около +/- 1).
            # Используем более простой диапазон для сплайнов (-n_features до n_features) для простоты.
            inner_input_min = -n_features
            inner_input_max = n_features
            if self.activation == 'spline':
                # Создание кубического сплайна для внутренней функции
                x_points = np.linspace(inner_input_min, inner_input_max, 10)
                y_points = np.random.uniform(-1, 1, 10)
                f = interp1d(x_points, y_points, kind='cubic', fill_value='extrapolate')
            else:  # активация 'rbf'
                # Создание RBF для внутренней функции
                center = np.random.uniform(inner_input_min, inner_input_max)
                width = np.random.uniform(0.1, 1.0)
                f = lambda x, c=center, w=width: np.exp(-((x - c)**2) / (2 * w**2))
            # Добавляем созданную внутреннюю функцию в список
            self.inner_functions.append(f)

        # --- Создание финальной выходной функции ---
        # Эта функция преобразует сумму выходов внутренних функций в подобие вероятности (0 до 1)
        # для бинарной классификации. Используем линейный сплайн, имитирующий сигмоиду.
        # Определяем диапазон входа для выходной функции (сумма n_inner выходов Φ)
        # Грубая оценка: от -n_inner до n_inner
        output_input_min = -self.n_inner
        output_input_max = self.n_inner
        x_points = np.linspace(output_input_min, output_input_max, 10)
        # Задаем Y-значения так, чтобы получилась S-образная кривая
        y_points = np.array([0, 0, 0, 0.2, 0.5, 0.8, 1, 1, 1, 1])
        # Создаем линейный сплайн
        self.output_function = interp1d(x_points, y_points, kind='linear', fill_value=(0, 1), bounds_error=False)
        # fill_value=(0, 1), bounds_error=False : значения вне диапазона x_points будут ограничены 0 или 1.


    def fit(self, X, y):
        """
        Метод "обучения" KAN модели (в соответствии с API Scikit-learn).
        В данной УПРОЩЕННОЙ реализации этот метод ТОЛЬКО вызывает
        `_create_activation_functions` для инициализации случайных функций активации.
        Параметры функций не настраиваются под данные X, y.

        Параметры:
        - X (np.array): Обучающие данные (признаки).
        - y (np.array): Обучающие данные (метки классов).

        Возвращает:
        - self (object): Экземпляр обученного классификатора.
        """
        # Вызываем метод для создания (инициализации) функций активации
        self._create_activation_functions(X, y)
        # Возвращаем self, как требует стандартный интерфейс Scikit-learn
        return self

    def predict(self, X):
        """
        Выполнение предсказаний для входных данных X.

        Параметры:
        - X (np.array): Данные для предсказания (признаки), форма [n_samples, n_features].

        Возвращает:
        - predictions (np.array): Предсказанные метки классов (0 или 1), форма [n_samples,].
        """
        # Получение количества образцов в данных X
        n_samples = X.shape[0]
        # Инициализация массива для хранения предсказаний (заполнен нулями)
        predictions = np.zeros(n_samples)

        # Итерация по каждому образцу (строке) в данных X
        for i in range(n_samples):
            # --- Шаг 1: Применение "внешних" функций (ψ) ---
            outer_outputs = []  # Список для хранения сумм выходов ψ для каждого признака
            # Итерация по каждому признаку (столбцу) текущего образца
            for feature_idx in range(X.shape[1]):
                # Получение значения текущего признака для текущего образца
                feature_val = X[i, feature_idx]
                # Применение ВСЕХ внешних функций, ассоциированных с ЭТИМ признаком
                feature_outputs = [f(feature_val) for f in self.outer_functions[feature_idx]]
                # Суммирование выходов всех внешних функций для данного признака
                outer_outputs.append(np.sum(feature_outputs))
                # outer_outputs теперь содержит по одному значению для каждого исходного признака,
                # каждое значение - это сумма выходов n_outer функций ψ для этого признака.

            # --- Шаг 2: Применение "внутренних" функций (Φ) ---
            inner_outputs = []  # Список для хранения выходов внутренних функций Φ
            # В этой реализации суммируем выходы ВСЕХ внешних функций перед подачей на каждую внутреннюю
            inner_input = np.sum(outer_outputs)
            # Итерация по всем внутренним функциям Φ
            for inner_func in self.inner_functions:
                # Применение текущей внутренней функции к сумме выходов внешних функций
                inner_outputs.append(inner_func(inner_input))
                # inner_outputs теперь содержит n_inner значений - выходы каждой функции Φ.

            # --- Шаг 3: Применение выходной функции и получение предсказания ---
            # Суммирование выходов всех внутренних функций Φ
            final_input = np.sum(inner_outputs)
            # Применение финальной выходной функции (сигмоидо-подобного сплайна)
            # для получения значения, похожего на вероятность принадлежности к классу 1
            prob = self.output_function(final_input)
            # Преобразование вероятности в бинарное предсказание (0 или 1) по порогу 0.5
            predictions[i] = 1 if prob > 0.5 else 0

        # Возвращение массива предсказаний
        return predictions

# --- Пример Использования KANClassifier ---

# --- Шаг 1: Генерация синтетических данных ---
# Используем ту же функцию, что и в примере с MLP
X, y = make_classification(n_samples=500, n_features=20, n_informative=15,
                           n_redundant=5, n_classes=2, random_state=42)

# --- Шаг 2: Разделение данных ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Шаг 3: Масштабирование данных ---
# Масштабирование также может быть полезно для KAN, особенно если используются RBF
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Шаг 4: Создание и "обучение" KAN модели ---
# Инициализация классификатора KANClassifier
# n_outer=15: используем 15 внешних функций на признак
# n_inner=10: используем 10 внутренних функций
# activation='spline': используем кубические сплайны
# random_state=42: для воспроизводимости инициализации функций
kan = KANClassifier(n_outer=15, n_inner=10, activation='spline', random_state=42)
print("Обучение модели KAN (инициализация функций)...")
# Вызов метода fit, который в данной реализации только создает функции активации
kan.fit(X_train, y_train)
print("Обучение (инициализация) завершено.")

# --- Шаг 5: Предсказание и оценка ---
# Получение предсказаний на тестовых данных
y_pred = kan.predict(X_test)
# Вычисление точности модели путем сравнения предсказанных и истинных меток
accuracy = accuracy_score(y_test, y_pred)
# Вывод точности
print(f"Точность модели KAN на тестовых данных: {accuracy:.2f}")
# Примечание: Точность этой упрощенной модели, скорее всего, будет невысокой,
# так как параметры функций активации не оптимизировались под данные.