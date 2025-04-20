import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os

class SingleNeuronClassifier:
    """
    Реалізація однонейронного класифікатора з налаштовуваними параметрами:
    alpha - швидкість навчання
    nu - момент (для прискорення навчання)
    E0 - цільова точність навчання
    """
    def __init__(self, activation_function='sigmoid'):
        # Ініціалізація ваг і зсуву
        self.weights = None
        self.bias = None
        self.delta_weights_prev = None
        self.delta_bias_prev = None
        
        # Вибір функції активації
        if activation_function == 'sigmoid':
            self.activation = self._sigmoid
            self.activation_derivative = self._sigmoid_derivative
        elif activation_function == 'tanh':
            self.activation = self._tanh
            self.activation_derivative = self._tanh_derivative
        elif activation_function == 'step':
            self.activation = self._step
            self.activation_derivative = self._step_derivative
        else:
            raise ValueError("Підтримуються лише 'sigmoid', 'tanh' і 'step' функції активації")
        
        # Журнал навчання
        self.training_log = {
            'epochs': [],
            'train_error': [],
            'val_error': []
        }
        
        # Алгоритм навчання
        self.train_algorithm = 'gradient_descent'
    
    def _sigmoid(self, z):
        """Сигмоїдна функція активації."""
        return 1 / (1 + np.exp(-z))
    
    def _sigmoid_derivative(self, z):
        """Похідна сигмоїдної функції."""
        s = self._sigmoid(z)
        return s * (1 - s)
    
    def _tanh(self, z):
        """Гіперболічний тангенс."""
        return np.tanh(z)
    
    def _tanh_derivative(self, z):
        """Похідна гіперболічного тангенса."""
        return 1 - np.tanh(z) ** 2
    
    def _step(self, z):
        """Порогова функція активації."""
        return np.where(z >= 0, 1, 0)
    
    def _step_derivative(self, z):
        """Похідна порогової функції (апроксимація)."""
        # Оскільки порогова функція не є диференційованою в 0,
        # використовуємо просту апроксимацію похідної
        return np.where(np.abs(z) < 0.1, 1, 0)
    
    def initialize_weights(self, n_features):
        """Ініціалізація ваг і зсуву."""
        # Ініціалізація малими випадковими значеннями для покращення збіжності
        self.weights = np.random.randn(n_features) * 0.1
        self.bias = np.random.randn() * 0.1
        
        # Ініціалізація змінних для моменту
        self.delta_weights_prev = np.zeros(n_features)
        self.delta_bias_prev = 0
    
    def forward(self, X):
        """Прямий прохід (обчислення прогнозів)."""
        # Обчислення зваженої суми входів
        z = np.dot(X, self.weights) + self.bias
        # Застосування функції активації
        return self.activation(z)
    
    def compute_error(self, y_true, y_pred):
        """Обчислення середньоквадратичної помилки."""
        return np.mean((y_true - y_pred) ** 2)
    
    def fit(self, X_train, y_train, X_val, y_val, alpha=0.01, nu=0.0, E0=0.01, max_epochs=10000):
        """
        Навчання моделі з використанням градієнтного спуску.
        
        Параметри:
        X_train: тренувальні дані (ознаки)
        y_train: тренувальні дані (цільові значення)
        X_val: валідаційні дані (ознаки)
        y_val: валідаційні дані (цільові значення)
        alpha: швидкість навчання
        nu: коефіцієнт моменту (0.0 - без моменту)
        E0: цільова точність (середньоквадратична помилка)
        max_epochs: максимальна кількість епох
        
        Повертає:
        epochs: кількість епох навчання
        """
        # Ініціалізація ваг
        n_samples, n_features = X_train.shape
        self.initialize_weights(n_features)
        
        # Очищення журналу навчання
        self.training_log = {
            'epochs': [],
            'train_error': [],
            'val_error': []
        }
        
        # Цикл навчання
        epoch = 0
        train_error = float('inf')
        val_error = float('inf')
        
        while epoch < max_epochs and train_error > E0:
            # Прямий прохід
            y_pred = self.forward(X_train)
            
            # Обчислення помилки
            train_error = self.compute_error(y_train, y_pred)
            
            # Зворотнє поширення помилки
            error = y_train - y_pred
            
            # Обчислення градієнта для ваг і зсуву
            z = np.dot(X_train, self.weights) + self.bias
            delta = error * self.activation_derivative(z)
            
            dw = np.dot(X_train.T, delta) / n_samples
            db = np.mean(delta)
            
            # Застосування моменту для прискорення навчання
            delta_weights = alpha * dw + nu * self.delta_weights_prev
            delta_bias = alpha * db + nu * self.delta_bias_prev
            
            # Оновлення ваг і зсуву
            self.weights += delta_weights
            self.bias += delta_bias
            
            # Запам'ятовування змін ваг для наступної ітерації (для моменту)
            self.delta_weights_prev = delta_weights
            self.delta_bias_prev = delta_bias
            
            # Обчислення помилки на валідаційному наборі
            y_val_pred = self.forward(X_val)
            val_error = self.compute_error(y_val, y_val_pred)
            
            # Запис інформації в журнал
            self.training_log['epochs'].append(epoch)
            self.training_log['train_error'].append(train_error)
            self.training_log['val_error'].append(val_error)
            
            epoch += 1
        
        return epoch, train_error, val_error
    
    def predict(self, X, threshold=0.5):
        """
        Прогнозування класу для нових даних.
        
        Параметри:
        X: вхідні дані для прогнозування
        threshold: поріг для класифікації (для сигмоїдної і tanh функцій)
        
        Повертає:
        predictions: прогнозовані класи (0 або 1)
        """
        y_pred = self.forward(X)
        
        if self.activation == self._tanh:
            # Для tanh поріг встановлюється на 0
            return np.where(y_pred >= 0, 1, 0)
        else:
            # Для sigmoid і step поріг встановлюється на threshold (зазвичай 0.5)
            return np.where(y_pred >= threshold, 1, 0)
    
    def predict_proba(self, X):
        """
        Прогнозування ймовірності належності до класу 1.
        
        Параметри:
        X: вхідні дані для прогнозування
        
        Повертає:
        probabilities: ймовірності належності до класу 1
        """
        return self.forward(X)
    
    def compute_accuracy(self, X, y_true, threshold=0.5):
        """
        Обчислення точності моделі.
        
        Параметри:
        X: вхідні дані
        y_true: істинні значення класів
        threshold: поріг для класифікації
        
        Повертає:
        accuracy: точність (відсоток правильних прогнозів)
        """
        y_pred = self.predict(X, threshold)
        return np.mean(y_pred == y_true) * 100
    
    def plot_decision_boundary(self, X, y, title='Межа рішення'):
        """
        Візуалізація межі рішення для двовимірних даних.
        
        Параметри:
        X: вхідні дані (ознаки)
        y: цільові значення
        title: заголовок графіка
        """
        # Перевірка, чи дані двовимірні
        if X.shape[1] != 2:
            raise ValueError("Можна візуалізувати межу рішення тільки для двовимірних даних")
        
        # Створення сітки для візуалізації
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        
        # Прогнозування для всіх точок сітки
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Візуалізація
        plt.figure(figsize=(10, 6))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)
        plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', label='Клас 0')
        plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='Клас 1')
        
        # Візуалізація роздільної прямої (w1*x1 + w2*x2 + b = 0)
        slope = -self.weights[0] / self.weights[1]
        intercept = -self.bias / self.weights[1]
        x_points = np.array([x_min, x_max])
        y_points = slope * x_points + intercept
        plt.plot(x_points, y_points, 'g-', linewidth=2, label='Роздільна пряма')
        
        plt.xlabel('Ознака 1')
        plt.ylabel('Ознака 2')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_training_log(self):
        """Візуалізація процесу навчання."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_log['epochs'], self.training_log['train_error'], label='Помилка на тренувальному наборі')
        plt.plot(self.training_log['epochs'], self.training_log['val_error'], label='Помилка на валідаційному наборі')
        plt.xlabel('Епоха')
        plt.ylabel('Середньоквадратична помилка')
        plt.title('Процес навчання')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def get_model_info(self):
        """Повертає інформацію про модель."""
        info = {
            'weights': self.weights,
            'bias': self.bias,
            'activation': self.activation.__name__,
            'final_train_error': self.training_log['train_error'][-1] if self.training_log['train_error'] else None,
            'final_val_error': self.training_log['val_error'][-1] if self.training_log['val_error'] else None,
            'epochs': len(self.training_log['epochs'])
        }
        return info


class DataProcessor:
    """
    Клас для обробки даних: завантаження, нормалізація, поділ на вибірки.
    """
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.scaler = StandardScaler()
    
    def load_data_from_excel(self, file_path):
        """
        Завантаження даних з Excel файлу.
        
        Параметри:
        file_path: шлях до Excel файлу
        
        Повертає:
        True, якщо дані успішно завантажені, False - інакше
        """
        try:
            self.data = pd.read_excel(file_path)
            # Перевірка, чи файл має заголовки колонок, або потрібно їх встановити
            if all(isinstance(col, (int, float)) for col in self.data.columns):
                # Якщо стовпці числові, створюємо імена (feature_1, feature_2, ...)
                new_columns = [f'feature_{i+1}' for i in range(len(self.data.columns)-1)]
                new_columns.append('target')  # Останній стовпець вважаємо цільовим
                self.data.columns = new_columns
            return True
        except Exception as e:
            print(f"Помилка при завантаженні даних: {e}")
            return False
    
    def preprocess_data(self, target_column, train_size=0.6, val_size=0.2, test_size=0.2, normalize=True):
        """
        Попередня обробка даних: поділ на ознаки та цільову змінну,
        поділ на тренувальну, валідаційну та тестову вибірки, нормалізація.
        
        Параметри:
        target_column: назва стовпця з цільовою змінною або його індекс
        train_size: частка даних для тренування
        val_size: частка даних для валідації
        test_size: частка даних для тестування
        normalize: чи потрібно нормалізувати дані
        
        Повертає:
        True, якщо дані успішно оброблені, False - інакше
        """
        if self.data is None:
            print("Спочатку завантажте дані")
            return False
        
        # Перевірка, що сума розмірів дорівнює 1
        if abs(train_size + val_size + test_size - 1.0) > 1e-10:
            print("Сума розмірів вибірок повинна дорівнювати 1")
            return False
        
        try:
            # Виділення ознак та цільової змінної
            # Перевіряємо, чи target_column є індексом або назвою колонки
            if isinstance(target_column, (int, float)) or target_column.isdigit():
                # Якщо це індекс, конвертуємо в int
                target_idx = int(float(target_column)) if isinstance(target_column, str) else int(target_column)
                if target_idx >= len(self.data.columns):
                    print(f"Індекс стовпця {target_idx} виходить за межі даних")
                    return False
                
                # Отримуємо назву стовпця за індексом
                target_column = self.data.columns[target_idx]
            
            if target_column in self.data.columns:
                self.y = self.data[target_column].values
                self.X = self.data.drop(columns=[target_column]).values
            else:
                print(f"Стовпець {target_column} не знайдено в даних")
                # Спробуємо використати останній стовпець як цільовий
                print("Використовуємо останній стовпець як цільовий")
                target_column = self.data.columns[-1]
                self.y = self.data[target_column].values
                self.X = self.data.iloc[:, :-1].values
            
            # Поділ на тренувальну та тимчасову вибірки
            X_train, X_temp, y_train, y_temp = train_test_split(
                self.X, self.y, test_size=(val_size + test_size), random_state=42)
            
            # Поділ тимчасової вибірки на валідаційну та тестову
            val_ratio = val_size / (val_size + test_size)
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=(1 - val_ratio), random_state=42)
            
            # Нормалізація даних (якщо потрібно)
            if normalize:
                self.X_train = self.scaler.fit_transform(X_train)
                self.X_val = self.scaler.transform(X_val)
                self.X_test = self.scaler.transform(X_test)
            else:
                self.X_train = X_train
                self.X_val = X_val
                self.X_test = X_test
            
            self.y_train = y_train
            self.y_val = y_val
            self.y_test = y_test
            
            return True
        except Exception as e:
            print(f"Помилка при обробці даних: {e}")
            return False
    
    def get_data_info(self):
        """Повертає інформацію про дані."""
        info = {
            'total_samples': len(self.X) if self.X is not None else 0,
            'train_samples': len(self.X_train) if self.X_train is not None else 0,
            'val_samples': len(self.X_val) if self.X_val is not None else 0,
            'test_samples': len(self.X_test) if self.X_test is not None else 0,
            'features': self.X.shape[1] if self.X is not None else 0,
            'classes': np.unique(self.y) if self.y is not None else []
        }
        return info
    
    def normalize_new_data(self, X):
        """
        Нормалізація нових даних з використанням того ж скейлера,
        що використовувався для тренувальних даних.
        
        Параметри:
        X: нові дані для нормалізації
        
        Повертає:
        X_normalized: нормалізовані дані
        """
        if self.scaler is None:
            print("Спочатку треба навчити скейлер на тренувальних даних")
            return X
        
        return self.scaler.transform(X)


class NeuralNetworkApp:
    """
    Графічний інтерфейс для роботи з нейромережею.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Однонейронний класифікатор")
        self.root.geometry("800x600")
        
        # Створення об'єктів для обробки даних і нейромережі
        self.data_processor = DataProcessor()
        self.neural_network = SingleNeuronClassifier()
        
        # Створення інтерфейсу
        self.create_ui()
    
    def create_ui(self):
        """Створення графічного інтерфейсу."""
        # Створення вкладок
        self.tab_control = ttk.Notebook(self.root)
        
        self.tab_data = ttk.Frame(self.tab_control)
        self.tab_training = ttk.Frame(self.tab_control)
        self.tab_testing = ttk.Frame(self.tab_control)
        self.tab_visualization = ttk.Frame(self.tab_control)
        
        self.tab_control.add(self.tab_data, text="Дані")
        self.tab_control.add(self.tab_training, text="Навчання")
        self.tab_control.add(self.tab_testing, text="Тестування")
        self.tab_control.add(self.tab_visualization, text="Візуалізація")
        
        self.tab_control.pack(expand=1, fill="both")
        
        # Вкладка "Дані"
        self.create_data_tab()
        
        # Вкладка "Навчання"
        self.create_training_tab()
        
        # Вкладка "Тестування"
        self.create_testing_tab()
        
        # Вкладка "Візуалізація"
        self.create_visualization_tab()
    
    def create_data_tab(self):
        """Створення вкладки "Дані"."""
        # Рамка для завантаження даних
        load_frame = ttk.LabelFrame(self.tab_data, text="Завантаження даних")
        load_frame.pack(padx=10, pady=10, fill="x")
        
        # Кнопка для вибору файлу
        self.btn_load = ttk.Button(load_frame, text="Вибрати Excel файл", command=self.load_data)
        self.btn_load.pack(padx=10, pady=10)
        
        # Відображення шляху до файлу
        self.lbl_file_path = ttk.Label(load_frame, text="Файл не вибрано")
        self.lbl_file_path.pack(padx=10, pady=5)
        
        # Рамка для попередньої обробки даних
        preprocess_frame = ttk.LabelFrame(self.tab_data, text="Попередня обробка даних")
        preprocess_frame.pack(padx=10, pady=10, fill="x")
        
        # Вибір стовпця з цільовою змінною
        ttk.Label(preprocess_frame, text="Цільова змінна:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.combo_target = ttk.Combobox(preprocess_frame, state="readonly")
        self.combo_target.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Розміри вибірок
        ttk.Label(preprocess_frame, text="Тренувальна вибірка (%):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.entry_train_size = ttk.Entry(preprocess_frame)
        self.entry_train_size.insert(0, "60")
        self.entry_train_size.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        ttk.Label(preprocess_frame, text="Валідаційна вибірка (%):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.entry_val_size = ttk.Entry(preprocess_frame)
        self.entry_val_size.insert(0, "20")
        self.entry_val_size.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        
        ttk.Label(preprocess_frame, text="Тестова вибірка (%):").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.entry_test_size = ttk.Entry(preprocess_frame)
        self.entry_test_size.insert(0, "20")
        self.entry_test_size.grid(row=3, column=1, padx=5, pady=5, sticky="w")
        
        # Чекбокс для нормалізації
        self.var_normalize = tk.BooleanVar(value=True)
        self.chk_normalize = ttk.Checkbutton(
            preprocess_frame, text="Нормалізувати дані", variable=self.var_normalize)
        self.chk_normalize.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        
        # Кнопка для обробки даних
        self.btn_preprocess = ttk.Button(
            preprocess_frame, text="Обробити дані", command=self.preprocess_data)
        self.btn_preprocess.grid(row=5, column=0, columnspan=2, padx=5, pady=10)
        
        # Інформація про дані
        info_frame = ttk.LabelFrame(self.tab_data, text="Інформація про дані")
        info_frame.pack(padx=10, pady=10, fill="both", expand=True)
        
        self.text_data_info = tk.Text(info_frame, height=10, width=50)
        self.text_data_info.pack(padx=5, pady=5, fill="both", expand=True)
    
    def create_training_tab(self):
        """Створення вкладки "Навчання"."""
        # Рамка для параметрів навчання
        params_frame = ttk.LabelFrame(self.tab_training, text="Параметри навчання")
        params_frame.pack(padx=10, pady=10, fill="x")
        
        # Функція активації
        ttk.Label(params_frame, text="Функція активації:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.combo_activation = ttk.Combobox(params_frame, values=["sigmoid", "tanh", "step"], state="readonly")
        self.combo_activation.current(0)
        self.combo_activation.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Швидкість навчання (alpha)
        ttk.Label(params_frame, text="Швидкість навчання (alpha):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.entry_alpha = ttk.Entry(params_frame)
        self.entry_alpha.insert(0, "0.01")
        self.entry_alpha.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        # Момент (nu)
        ttk.Label(params_frame, text="Момент (nu):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.entry_nu = ttk.Entry(params_frame)
        self.entry_nu.insert(0, "0.0")
        self.entry_nu.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        
        # Цільова точність (E0)
        ttk.Label(params_frame, text="Цільова точність (E0):").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.entry_e0 = ttk.Entry(params_frame)
        self.entry_e0.insert(0, "0.01")
        self.entry_e0.grid(row=3, column=1, padx=5, pady=5, sticky="w")
        
        # Максимальна кількість епох
        ttk.Label(params_frame, text="Максимальна кількість епох:").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.entry_max_epochs = ttk.Entry(params_frame)
        self.entry_max_epochs.insert(0, "10000")
        self.entry_max_epochs.grid(row=4, column=1, padx=5, pady=5, sticky="w")
        
        # Кнопка для запуску навчання
        self.btn_train = ttk.Button(params_frame, text="Навчити модель", command=self.train_model)
        self.btn_train.grid(row=5, column=0, columnspan=2, padx=5, pady=10)
        
        # Інформація про навчання
        training_info_frame = ttk.LabelFrame(self.tab_training, text="Інформація про навчання")
        training_info_frame.pack(padx=10, pady=10, fill="both", expand=True)
        
        self.text_training_info = tk.Text(training_info_frame, height=10, width=50)
        self.text_training_info.pack(padx=5, pady=5, fill="both", expand=True)
    
    def create_testing_tab(self):
        """Створення вкладки "Тестування"."""
        # Рамка для оцінки моделі на тестовій вибірці
        test_frame = ttk.LabelFrame(self.tab_testing, text="Оцінка моделі на тестовій вибірці")
        test_frame.pack(padx=10, pady=10, fill="x")
        
        self.btn_evaluate = ttk.Button(test_frame, text="Оцінити модель на тестовій вибірці", command=self.evaluate_model)
        self.btn_evaluate.pack(padx=10, pady=10)
        
        # Інформація про тестування
        test_info_frame = ttk.LabelFrame(self.tab_testing, text="Результати тестування")
        test_info_frame.pack(padx=10, pady=10, fill="both", expand=True)
        
        self.text_test_info = tk.Text(test_info_frame, height=10, width=50)
        self.text_test_info.pack(padx=5, pady=5, fill="both", expand=True)
        
        # Рамка для прогнозування нових даних
        predict_frame = ttk.LabelFrame(self.tab_testing, text="Прогнозування для нових даних")
        predict_frame.pack(padx=10, pady=10, fill="x")
        
        # Вибір кількості ознак
        self.num_features = 2  # За замовчуванням 2 ознаки
        
        # Створимо контейнер для введення значень ознак
        self.features_frame = ttk.Frame(predict_frame)
        self.features_frame.pack(padx=5, pady=5, fill="x")

    # Функція для динамічного створення полів введення для ознак
        def create_feature_inputs(self, n_features):
            # Очищаємо попередні віджети
            for widget in self.features_frame.winfo_children():
                widget.destroy()
            
            self.feature_entries = []
            
            # Створюємо поля введення для кожної ознаки
            for i in range(n_features):
                ttk.Label(self.features_frame, text=f"Ознака {i+1}:").grid(row=i, column=0, padx=5, pady=2, sticky="w")
                entry = ttk.Entry(self.features_frame)
                entry.grid(row=i, column=1, padx=5, pady=2, sticky="w")
                self.feature_entries.append(entry)
        
        # Кнопка для прогнозування
        self.btn_predict = ttk.Button(predict_frame, text="Прогнозувати", command=self.predict_new_data)
        self.btn_predict.pack(padx=10, pady=10)
        
        # Результати прогнозування
        predict_result_frame = ttk.LabelFrame(predict_frame, text="Результат прогнозування")
        predict_result_frame.pack(padx=5, pady=5, fill="x")
        
        self.lbl_prediction = ttk.Label(predict_result_frame, text="")
        self.lbl_prediction.pack(padx=5, pady=5)
    
    def create_visualization_tab(self):
        """Створення вкладки "Візуалізація"."""
        # Кнопки для різних візуалізацій
        btn_frame = ttk.Frame(self.tab_visualization)
        btn_frame.pack(padx=10, pady=10, fill="x")
        
        self.btn_plot_data = ttk.Button(btn_frame, text="Візуалізувати дані", command=self.plot_data)
        self.btn_plot_data.pack(padx=5, pady=5, fill="x")
        
        self.btn_plot_decision = ttk.Button(btn_frame, text="Візуалізувати межу рішення", command=self.plot_decision_boundary)
        self.btn_plot_decision.pack(padx=5, pady=5, fill="x")
        
        self.btn_plot_training = ttk.Button(btn_frame, text="Візуалізувати процес навчання", command=self.plot_training_process)
        self.btn_plot_training.pack(padx=5, pady=5, fill="x")
    
    def load_data(self):
        """Завантаження даних з Excel файлу."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Excel files", "*.xlsx *.xls")])
        
        if not file_path:
            return
        
        if self.data_processor.load_data_from_excel(file_path):
            self.lbl_file_path.config(text=f"Файл: {os.path.basename(file_path)}")
            
            # Оновлення випадаючого списку для вибору цільової змінної
            self.combo_target['values'] = self.data_processor.data.columns.tolist()
            if len(self.data_processor.data.columns) > 0:
                self.combo_target.current(len(self.data_processor.data.columns) - 1)
            
            # Оновлення інформації про дані
            self.text_data_info.delete(1.0, tk.END)
            self.text_data_info.insert(tk.END, f"Файл завантажено успішно.\n")
            self.text_data_info.insert(tk.END, f"Кількість рядків: {len(self.data_processor.data)}\n")
            self.text_data_info.insert(tk.END, f"Кількість стовпців: {len(self.data_processor.data.columns)}\n")
            self.text_data_info.insert(tk.END, f"Стовпці: {', '.join(self.data_processor.data.columns)}\n\n")
            self.text_data_info.insert(tk.END, "Перші 5 рядків даних:\n")
            self.text_data_info.insert(tk.END, f"{self.data_processor.data.head().to_string()}\n")
        else:
            messagebox.showerror("Помилка", "Не вдалося завантажити файл")
    
    def preprocess_data(self):
        """Попередня обробка даних."""
        if self.data_processor.data is None:
            messagebox.showerror("Помилка", "Спочатку завантажте дані")
            return
        
        target_column = self.combo_target.get()
        
        try:
            train_size = float(self.entry_train_size.get()) / 100
            val_size = float(self.entry_val_size.get()) / 100
            test_size = float(self.entry_test_size.get()) / 100
            
            if abs(train_size + val_size + test_size - 1.0) > 1e-10:
                messagebox.showerror("Помилка", "Сума розмірів вибірок повинна дорівнювати 100%")
                return
            
            normalize = self.var_normalize.get()
            
            if self.data_processor.preprocess_data(target_column, train_size, val_size, test_size, normalize):
                # Оновлення інформації про дані
                info = self.data_processor.get_data_info()
                
                self.text_data_info.delete(1.0, tk.END)
                self.text_data_info.insert(tk.END, "Дані оброблено успішно.\n\n")
                self.text_data_info.insert(tk.END, f"Загальна кількість зразків: {info['total_samples']}\n")
                self.text_data_info.insert(tk.END, f"Кількість ознак: {info['features']}\n")
                self.text_data_info.insert(tk.END, f"Класи: {', '.join(map(str, info['classes']))}\n\n")
                self.text_data_info.insert(tk.END, f"Тренувальна вибірка: {info['train_samples']} зразків ({train_size*100:.1f}%)\n")
                self.text_data_info.insert(tk.END, f"Валідаційна вибірка: {info['val_samples']} зразків ({val_size*100:.1f}%)\n")
                self.text_data_info.insert(tk.END, f"Тестова вибірка: {info['test_samples']} зразків ({test_size*100:.1f}%)\n")
                
                # Оновлення кількості ознак для прогнозування
                self.num_features = info['features']
                self.create_feature_inputs(self.num_features)
                
                messagebox.showinfo("Успіх", "Дані оброблено успішно")
            else:
                messagebox.showerror("Помилка", "Виникла помилка при обробці даних")
        
        except ValueError:
            messagebox.showerror("Помилка", "Введіть коректні числові значення для розмірів вибірок")
    
    def train_model(self):
        """Навчання моделі."""
        if self.data_processor.X_train is None:
            messagebox.showerror("Помилка", "Спочатку обробіть дані")
            return
        
        try:
            # Отримання параметрів навчання
            activation = self.combo_activation.get()
            alpha = float(self.entry_alpha.get())
            nu = float(self.entry_nu.get())
            e0 = float(self.entry_e0.get())
            max_epochs = int(self.entry_max_epochs.get())
            
            # Створення нової моделі з вибраною функцією активації
            self.neural_network = SingleNeuronClassifier(activation)
            
            # Навчання моделі
            epochs, train_error, val_error = self.neural_network.fit(
                self.data_processor.X_train,
                self.data_processor.y_train,
                self.data_processor.X_val,
                self.data_processor.y_val,
                alpha=alpha,
                nu=nu,
                E0=e0,
                max_epochs=max_epochs
            )
            
            # Оновлення інформації про навчання
            self.text_training_info.delete(1.0, tk.END)
            self.text_training_info.insert(tk.END, "Модель навчено успішно.\n\n")
            self.text_training_info.insert(tk.END, f"Функція активації: {activation}\n")
            self.text_training_info.insert(tk.END, f"Кількість епох: {epochs}\n")
            
            # Форматування ваг та зсуву
            weights_str = ", ".join([f"{w:.4f}" for w in self.neural_network.weights])
            
            self.text_training_info.insert(tk.END, f"Ваги: [{weights_str}]\n")
            self.text_training_info.insert(tk.END, f"Зсув: {self.neural_network.bias:.4f}\n\n")
            
            self.text_training_info.insert(tk.END, f"Помилка на тренувальній вибірці: {train_error:.6f}\n")
            self.text_training_info.insert(tk.END, f"Помилка на валідаційній вибірці: {val_error:.6f}\n\n")
            
            # Обчислення та відображення точності
            train_accuracy = self.neural_network.compute_accuracy(
                self.data_processor.X_train, self.data_processor.y_train)
            val_accuracy = self.neural_network.compute_accuracy(
                self.data_processor.X_val, self.data_processor.y_val)
            
            self.text_training_info.insert(tk.END, f"Точність на тренувальній вибірці: {train_accuracy:.2f}%\n")
            self.text_training_info.insert(tk.END, f"Точність на валідаційній вибірці: {val_accuracy:.2f}%\n")
            
            messagebox.showinfo("Успіх", f"Модель навчено успішно за {epochs} епох")
        
        except ValueError:
            messagebox.showerror("Помилка", "Введіть коректні числові значення для параметрів")
    
    def evaluate_model(self):
        """Оцінка моделі на тестовій вибірці."""
        if self.neural_network.weights is None:
            messagebox.showerror("Помилка", "Спочатку навчіть модель")
            return
        
        if self.data_processor.X_test is None:
            messagebox.showerror("Помилка", "Тестові дані не знайдено")
            return
        
        # Обчислення прогнозів для тестової вибірки
        y_pred = self.neural_network.predict(self.data_processor.X_test)
        
        # Обчислення точності на тестовій вибірці
        test_accuracy = self.neural_network.compute_accuracy(
            self.data_processor.X_test, self.data_processor.y_test)
        
        # Обчислення помилки на тестовій вибірці
        test_error = self.neural_network.compute_error(
            self.data_processor.y_test,
            self.neural_network.predict_proba(self.data_processor.X_test)
        )
        
        # Оновлення інформації про тестування
        self.text_test_info.delete(1.0, tk.END)
        self.text_test_info.insert(tk.END, "Результати на тестовій вибірці:\n\n")
        self.text_test_info.insert(tk.END, f"Кількість тестових зразків: {len(self.data_processor.X_test)}\n")
        self.text_test_info.insert(tk.END, f"Точність: {test_accuracy:.2f}%\n")
        self.text_test_info.insert(tk.END, f"Середньоквадратична помилка: {test_error:.6f}\n\n")
        
        # Відображення перших 10 прогнозів
        n_samples = min(10, len(self.data_processor.X_test))
        
        self.text_test_info.insert(tk.END, f"Перші {n_samples} прогнозів:\n\n")
        self.text_test_info.insert(tk.END, "Істинний клас | Прогнозований клас | Імовірність\n")
        self.text_test_info.insert(tk.END, "-" * 50 + "\n")
        
        proba = self.neural_network.predict_proba(self.data_processor.X_test[:n_samples])
        
        for i in range(n_samples):
            true_class = self.data_processor.y_test[i]
            pred_class = y_pred[i]
            probability = proba[i]
            
            self.text_test_info.insert(
                tk.END,
                f"{true_class:12d} | {pred_class:18d} | {probability:.6f}\n"
            )
    
    def predict_new_data(self):
        """Прогнозування для нових даних."""
        if self.neural_network.weights is None:
            messagebox.showerror("Помилка", "Спочатку навчіть модель")
            return
        
        try:
            # Зчитування нових даних з полів введення
            features = []
            for entry in self.feature_entries:
                features.append(float(entry.get()))
            
            if len(features) != self.num_features:
                messagebox.showerror("Помилка", f"Необхідно ввести {self.num_features} ознак")
                return
            
            # Перетворення на масив numpy і нормалізація
            features = np.array([features])
            features_normalized = self.data_processor.normalize_new_data(features)
            
            # Прогнозування
            predicted_class = self.neural_network.predict(features_normalized)[0]
            probability = self.neural_network.predict_proba(features_normalized)[0]
            
            # Відображення результату
            self.lbl_prediction.config(
                text=f"Прогнозований клас: {predicted_class} (імовірність: {probability:.4f})"
            )
        
        except ValueError:
            messagebox.showerror("Помилка", "Введіть коректні числові значення для ознак")
    
    def plot_data(self):
        """Візуалізація даних."""
        if self.data_processor.X is None:
            messagebox.showerror("Помилка", "Спочатку завантажте та обробіть дані")
            return
        
        # Перевірка, чи дані двовимірні (для візуалізації)
        if self.data_processor.X.shape[1] != 2:
            messagebox.showinfo("Інформація", "Візуалізація можлива тільки для двовимірних даних")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Візуалізація тренувальних даних
        plt.scatter(
            self.data_processor.X_train[self.data_processor.y_train == 0, 0],
            self.data_processor.X_train[self.data_processor.y_train == 0, 1],
            color='blue', label='Клас 0 (тренувальні)'
        )
        plt.scatter(
            self.data_processor.X_train[self.data_processor.y_train == 1, 0],
            self.data_processor.X_train[self.data_processor.y_train == 1, 1],
            color='red', label='Клас 1 (тренувальні)'
        )
        
        # Візуалізація валідаційних даних
        plt.scatter(
            self.data_processor.X_val[self.data_processor.y_val == 0, 0],
            self.data_processor.X_val[self.data_processor.y_val == 0, 1],
            color='lightblue', marker='^', label='Клас 0 (валідаційні)'
        )
        plt.scatter(
            self.data_processor.X_val[self.data_processor.y_val == 1, 0],
            self.data_processor.X_val[self.data_processor.y_val == 1, 1],
            color='lightcoral', marker='^', label='Клас 1 (валідаційні)'
        )
        
        # Візуалізація тестових даних
        plt.scatter(
            self.data_processor.X_test[self.data_processor.y_test == 0, 0],
            self.data_processor.X_test[self.data_processor.y_test == 0, 1],
            color='cyan', marker='s', label='Клас 0 (тестові)'
        )
        plt.scatter(
            self.data_processor.X_test[self.data_processor.y_test == 1, 0],
            self.data_processor.X_test[self.data_processor.y_test == 1, 1],
            color='salmon', marker='s', label='Клас 1 (тестові)'
        )
        
        plt.xlabel('Ознака 1')
        plt.ylabel('Ознака 2')
        plt.title('Візуалізація набору даних')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_decision_boundary(self):
        """Візуалізація межі рішення."""
        if self.neural_network.weights is None:
            messagebox.showerror("Помилка", "Спочатку навчіть модель")
            return
        
        if self.data_processor.X.shape[1] != 2:
            messagebox.showinfo("Інформація", "Візуалізація межі рішення можлива тільки для двовимірних даних")
            return
        
        # Візуалізація межі рішення для всього набору даних
        self.neural_network.plot_decision_boundary(self.data_processor.X, self.data_processor.y)
    
    def plot_training_process(self):
        """Візуалізація процесу навчання."""
        if self.neural_network.weights is None or not self.neural_network.training_log['epochs']:
            messagebox.showerror("Помилка", "Спочатку навчіть модель")
            return
        
        # Візуалізація процесу навчання
        self.neural_network.plot_training_log()


if __name__ == "__main__":
    # Створення головного вікна програми
    root = tk.Tk()
    app = NeuralNetworkApp(root)
    root.mainloop()