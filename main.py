import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import threading
import time


import decision_tree_regression
import random_forest_regression
import hist_crude_oil
import draw_graph
import correlation_matrix

class OilPriceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Oil Price Prediction")
        self.root.geometry("1528x1000")

        self.df = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.style = ttk.Style()
        self.style.theme_use('classic')

        self.setup_ui()

    def setup_ui(self):
        # Create main frames
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Left panel - Controls
        left_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))

        # Right panel - Display
        right_frame = ttk.LabelFrame(main_frame, text="Visualization", padding="10")
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)

        # === LEFT PANEL CONTROLS ===

        # File loading section
        file_frame = ttk.Frame(left_frame)
        file_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Button(file_frame, text="Load CSV File",
                   command=self.load_file).grid(row=0, column=0, sticky=(tk.W, tk.E))

        self.file_label = ttk.Label(file_frame, text="No file loaded")
        self.file_label.grid(row=1, column=0, sticky=(tk.W, tk.E))

        # Data info
        self.data_info = tk.Text(left_frame, height=5, width=30)
        self.data_info.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # Feature selection
        feature_frame = ttk.LabelFrame(left_frame, text="Feature Selection", padding="5")
        feature_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(feature_frame, text="Features (X):").grid(row=0, column=0, sticky=tk.W)
        self.feature_listbox = tk.Listbox(feature_frame, selectmode=tk.MULTIPLE, height=4)
        self.feature_listbox.grid(row=1, column=0, sticky=(tk.W, tk.E))

        ttk.Label(feature_frame, text="Target (y):").grid(row=2, column=0, sticky=tk.W)
        self.target_var = tk.StringVar()
        self.target_combo = ttk.Combobox(feature_frame, textvariable=self.target_var, state="readonly")
        self.target_combo.grid(row=3, column=0, sticky=(tk.W, tk.E))

        # Model selection
        model_frame = ttk.LabelFrame(left_frame, text="Model Training", padding="5")
        model_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(model_frame, text="Model:").grid(row=0, column=0, sticky=tk.W)
        self.model_var = tk.StringVar(value="Decision Tree")
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_var,
                                   values=["Decision Tree", "Random Forest"], state="readonly")
        model_combo.grid(row=0, column=1, sticky=(tk.W, tk.E))

        ttk.Label(model_frame, text="Max Depth:").grid(row=1, column=0, sticky=tk.W)
        self.depth_var = tk.StringVar(value="10")
        ttk.Entry(model_frame, textvariable=self.depth_var).grid(row=1, column=1, sticky=(tk.W, tk.E))

        ttk.Label(model_frame, text="Test Size:").grid(row=2, column=0, sticky=tk.W)
        self.test_size_var = tk.StringVar(value="0.2")
        ttk.Entry(model_frame, textvariable=self.test_size_var).grid(row=2, column=1, sticky=(tk.W, tk.E))

        self.train_btn = ttk.Button(model_frame, text="Train Model", command=self.train_model)
        self.train_btn.grid(row=3, column=0, columnspan=2, pady=(5, 0))

        self.train_progress = ttk.Progressbar(model_frame, mode='indeterminate')
        self.train_progress.grid(row=4, column=0, columnspan=2,  sticky=(tk.W, tk.E), pady=(5, 0))

        # Visualization buttons
        viz_frame = ttk.LabelFrame(left_frame, text="Visualization", padding="5")
        viz_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Button(viz_frame, text="Histogram",
                   command=self.show_histogram).grid(row=1, column=0, sticky=(tk.W, tk.E), pady=2)
        ttk.Button(viz_frame, text="Correlation Matrix",
                   command=self.show_correlation).grid(row=0, column=0, sticky=(tk.W, tk.E), pady=2)
        ttk.Button(viz_frame , text="Pair Plot",
                   command=self.show_pair_plot).grid(row=2, column=0, sticky=(tk.W, tk.E), pady=2)
        ttk.Button(viz_frame, text="Prediction Plot",
                   command=self.show_prediction_plot).grid(row=3, column=0, sticky=(tk.W, tk.E), pady=2)
        # ttk.Button(viz_frame, text="Prediction Results",
        #            command=self.show_predictions).grid(row=3, column=0, sticky=(tk.W, tk.E), pady=2)
        # Prediction section
        pred_frame = ttk.LabelFrame(left_frame, text="Manual Prediction", padding="5")
        pred_frame.grid(row=5, column=0, sticky=(tk.W, tk.E))

        self.pred_inputs = {}
        self.pred_frame_inner = ttk.Frame(pred_frame)
        self.pred_frame_inner.grid(row=0, column=0, sticky=(tk.W, tk.E))

        self.predict_btn = ttk.Button(pred_frame, text="Predict",
                                      command=self.manual_predict, state="disabled")
        self.predict_btn.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))

        self.pred_result = ttk.Label(pred_frame, text="", foreground="blue")
        self.pred_result.grid(row=2, column=0, sticky=(tk.W, tk.E))

        # Evaluation section
        eval_frame = ttk.LabelFrame(left_frame, text="Model Evaluation", padding="5")
        eval_frame.grid(row=6, column=0, sticky=(tk.W, tk.E), pady=(10, 0))

        self.eval_text = tk.Text(eval_frame, height=6, width=30)
        self.eval_text.grid(row=0, column=0, sticky=(tk.W, tk.E))

        # === RIGHT PANEL DISPLAY ===

        # Chart title
        self.chart_title = ttk.Label(right_frame, text="Select a visualization option", font=('Arial', 12, 'bold'))
        self.chart_title.grid(row=0, column=0, sticky=(tk.W, tk.E))

        # Chart frame
        self.chart_frame = ttk.Frame(right_frame)
        self.chart_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_frame.rowconfigure(1, weight=1)
        right_frame.columnconfigure(0, weight=1)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.file_label.config(text=f"Loaded: {file_path.split('/')[-1]}")
                self.update_data_info()
                self.update_feature_lists()
                messagebox.showinfo("Success", f"Loaded {len(self.df)} rows with {len(self.df.columns)} columns")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")

    def update_data_info(self):
        info_text = f"Rows: {len(self.df)}\nColumns: {len(self.df.columns)}\n\nColumns:\n"
        for col in self.df.columns:
            info_text += f"- {col}\n"
        self.data_info.delete(1.0, tk.END)
        self.data_info.insert(1.0, info_text)

    def update_feature_lists(self):
        # Clear existing lists
        self.feature_listbox.delete(0, tk.END)
        self.target_combo['values'] = ()

        # Populate with column names
        for col in self.df.columns:
            self.feature_listbox.insert(tk.END, col)

        self.target_combo['values'] = tuple(self.df.columns)
        if len(self.df.columns) > 0:
            self.target_var.set(self.df.columns[-1])  # Set last column as default target

    def train_model(self):
        if self.df is None:
            messagebox.showerror("Error", "Please load a CSV file first")
            return

        # Get selected features and target
        selected_features = [self.feature_listbox.get(i) for i in self.feature_listbox.curselection()]
        target = self.target_var.get()

        if not selected_features:
            messagebox.showerror("Error", "Please select at least one feature")
            return

        if not target:
            messagebox.showerror("Error", "Please select a target column")
            return

        if target in selected_features:
            messagebox.showerror("Error", "Target column cannot be in features")
            return

        # Start training in separate thread
        self.train_btn.config(state="disabled")
        self.train_progress.start()

        thread = threading.Thread(target=self._train_model_thread,
                                  args=(selected_features, target))
        thread.daemon = True
        thread.start()

    def _train_model_thread(self, selected_features, target):
        try:
            # Xử lý datetime columns trong training
            processed_df = self.df.copy()
            self.datetime_features = []

            for feature in selected_features:
                if processed_df[feature].dtype == 'object':
                    try:
                        # Thử convert sang datetime
                        processed_df[feature] = pd.to_datetime(processed_df[feature], errors='coerce')
                        # Kiểm tra xem có convert thành công không
                        if not processed_df[feature].isna().all():
                            # Chuyển thành timestamp và convert sang float
                            processed_df[feature] = (processed_df[feature].astype('int64') // 10 ** 9).astype(
                                np.float64)
                            self.datetime_features.append(feature)
                            print(f"Converted datetime feature: {feature} to float")
                        else:
                            print(f"Feature {feature} cannot be converted to datetime")
                    except Exception as e:
                        print(f"Error converting {feature} to datetime: {str(e)}")

            # Prepare data - ĐẢM BẢO TẤT CẢ LÀ FLOAT
            X = processed_df[selected_features].astype(np.float64).values
            y = processed_df[target].astype(np.float64).values

            print(f"X shape: {X.shape}, X dtype: {X.dtype}")
            print(f"y shape: {y.shape}, y dtype: {y.dtype}")
            print(f"Datetime features: {self.datetime_features}")

            # Split data
            test_size = float(self.test_size_var.get())
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            # Train model
            max_depth = int(self.depth_var.get())
            if self.model_var.get() == "Decision Tree":
                self.model = decision_tree_regression.DecisionTree(max_depth=max_depth)
            else:
                self.model = random_forest_regression.RandomForestRegression(max_depth=max_depth)

            self.model.fit(self.X_train, self.y_train)

            print("Model training completed successfully")

            # Update UI in main thread
            self.root.after(0, self._training_complete)

        except Exception as e:
            print(f"Training error: {str(e)}")
            self.root.after(0, lambda: self._training_error(str(e)))

        except Exception as e:
            self.root.after(0, lambda: self._training_error(str(e)))

    def _training_complete(self):
        self.train_progress.stop()
        self.train_btn.config(state="normal")
        self.predict_btn.config(state="normal")
        self.update_prediction_inputs()
        self.update_evaluation()
        messagebox.showinfo("Success", "Model training completed!")

    def _training_error(self, error_msg):
        self.train_progress.stop()
        self.train_btn.config(state="normal")
        messagebox.showerror("Error", f"Training failed: {error_msg}")

    def update_prediction_inputs(self):
        # Clear existing inputs
        for widget in self.pred_frame_inner.winfo_children():
            widget.destroy()

        self.pred_inputs = {}
        self.datetime_features = []  # Lưu danh sách các feature là datetime

        # Get feature names
        selected_features = [self.feature_listbox.get(i) for i in self.feature_listbox.curselection()]

        # Create input fields for each feature
        row_counter = 0
        for feature in selected_features:
            # Kiểm tra nếu cột là datetime
            is_datetime = False
            if self.df[feature].dtype == 'object':
                try:
                    # Thử convert sang datetime để kiểm tra
                    pd.to_datetime(self.df[feature].head())
                    is_datetime = True
                    self.datetime_features.append(feature)  # Lưu lại feature datetime
                except:
                    pass

            ttk.Label(self.pred_frame_inner, text=f"{feature}:").grid(row=row_counter, column=0, sticky=tk.W)

            if is_datetime:
                # Tạo entry cho datetime với placeholder
                var = tk.StringVar()
                entry = ttk.Entry(self.pred_frame_inner, textvariable=var)
                entry.grid(row=row_counter, column=1, sticky=(tk.W, tk.E), padx=(5, 0))

                # Thêm hint cho format datetime
                hint_label = ttk.Label(self.pred_frame_inner,
                                       text="(YYYY-MM-DD or DD/MM/YYYY)",
                                       foreground="gray", font=('Arial', 8))
                hint_label.grid(row=row_counter, column=2, sticky=tk.W, padx=(5, 0))

            else:
                var = tk.StringVar()
                entry = ttk.Entry(self.pred_frame_inner, textvariable=var)
                entry.grid(row=row_counter, column=1, sticky=(tk.W, tk.E), padx=(5, 0))

            self.pred_inputs[feature] = var
            row_counter += 1

    def manual_predict(self):
        if self.model is None:
            messagebox.showerror("Error", "Please train a model first")
            return

        try:
            # Get input values
            input_values = []
            selected_features = [self.feature_listbox.get(i) for i in self.feature_listbox.curselection()]

            print(f"Selected features: {selected_features}")
            print(f"Datetime features: {getattr(self, 'datetime_features', [])}")

            for feature in selected_features:
                value_str = self.pred_inputs[feature].get().strip()

                if not value_str:
                    messagebox.showerror("Error", f"Please enter value for {feature}")
                    return

                # Kiểm tra nếu feature là datetime
                is_datetime = hasattr(self, 'datetime_features') and feature in self.datetime_features

                if is_datetime:
                    # Xử lý datetime input
                    try:
                        date_val = pd.to_datetime(value_str, dayfirst=True, errors='coerce')
                        if pd.isna(date_val):
                            date_val = pd.to_datetime(value_str, errors='coerce')

                        if pd.isna(date_val):
                            messagebox.showerror("Error",
                                                 f"Invalid date format for {feature}. Use YYYY-MM-DD or DD/MM/YYYY")
                            return

                        # Chuyển thành timestamp và convert sang float
                        timestamp = float(date_val.timestamp())
                        input_values.append(timestamp)
                        print(f"Datetime feature '{feature}': '{value_str}' -> {timestamp}")

                    except Exception as e:
                        messagebox.showerror("Error",
                                             f"Error processing date for {feature}: {str(e)}")
                        return
                else:
                    # Xử lý numerical input - ĐẢM BẢO LÀ FLOAT
                    try:
                        numeric_value = float(value_str)
                        input_values.append(numeric_value)
                        print(f"Numerical feature '{feature}': '{value_str}' -> {numeric_value}")
                    except ValueError:
                        messagebox.showerror("Error",
                                             f"Please enter valid number for {feature}")
                        return

            # Debug: kiểm tra input values
            print(f"Final input values: {input_values}")
            print(f"Input types: {[type(x) for x in input_values]}")

            # ĐẢM BẢO INPUT LÀ NUMPY ARRAY VỚI DTYPE FLOAT
            X_pred = np.array([input_values], dtype=np.float64)
            print(f"Input array shape: {X_pred.shape}")
            print(f"Input array dtype: {X_pred.dtype}")

            # Make prediction
            prediction = self.model.predict(X_pred)[0]

            # Hiển thị kết quả
            result_text = f"Predicted: {prediction:.4f}"

            # Hiển thị thông tin datetime đã nhập (nếu có)
            datetime_inputs = []
            for feature in selected_features:
                if hasattr(self, 'datetime_features') and feature in self.datetime_features:
                    datetime_inputs.append(f"{feature}: {self.pred_inputs[feature].get()}")

            if datetime_inputs:
                result_text += f"\nDate inputs: {', '.join(datetime_inputs)}"

            self.pred_result.config(text=result_text)
            print(f"Prediction result: {prediction}")

        except Exception as e:
            print(f"Prediction error: {str(e)}")
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")

    def update_evaluation(self):
        if self.model is None or self.X_test is None:
            return

        # Make predictions
        y_pred = self.model.predict(self.X_test)

        # Calculate metrics
        r2 = r2_score(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)

        eval_text = f"R² Score: {r2:.4f}\n"
        eval_text += f"MSE: {mse:.4f}\n"
        eval_text += f"RMSE: {rmse:.4f}\n\n"
        eval_text += f"Test samples: {len(self.y_test)}\n"
        eval_text += f"Train samples: {len(self.y_train)}"

        self.eval_text.delete(1.0, tk.END)
        self.eval_text.insert(1.0, eval_text)

    def clear_chart(self):
        for widget in self.chart_frame.winfo_children():
            widget.destroy()

    def show_histogram(self):
        if self.df is None:
            messagebox.showerror("Error", "Please load a CSV file first")
            return

        self.clear_chart()
        self.chart_title.config(text="Histograms of Numerical Features")

        try:

            # Tạo figure tổng hợp
            fig = plt.figure(figsize=(12, 8))

            # Lấy các cột cần vẽ
            numerical_cols = self.df.select_dtypes(include=[np.number]).columns
            use_cols = [col for col in numerical_cols if col in hist_crude_oil.DEFAULT_COLS]

            if not use_cols:
                use_cols = numerical_cols[:4]

            # Vẽ từng subplot
            for i, col in enumerate(use_cols[:4]):
                if col in self.df.columns:
                    plt.subplot(2, 2, i + 1)
                    series = self.df[col].dropna()
                    if not series.empty:
                        plt.hist(series, bins=50, alpha=0.7, color=f'C{i}')
                        plt.title(f'Phân phối {col}')
                        plt.xlabel(col)
                        plt.ylabel('Tần suất')
                        plt.grid(True, alpha=0.3)

            plt.tight_layout()

            # Embed in tkinter
            canvas = FigureCanvasTkAgg(fig, self.chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to create histogram: {str(e)}")

    def show_pair_plot(self):
        if self.df is None:
            messagebox.showerror("Error", "Please load a CSV file first")
            return

        self.clear_chart()
        self.chart_title.config(text="Pair Plot")

        try:
            # Kiểm tra các cột cần thiết
            required_cols = ['open', 'close', 'high', 'low', 'volume']
            available_cols = [col for col in required_cols if col in self.df.columns]

            if len(available_cols) < 2:
                messagebox.showwarning("Warning",
                                       f"Need at least 2 numerical columns from {required_cols}")
                return

            # SỬ DỤNG FUNCTION VỚI show_plot=False
            fig = draw_graph.draw_pair_plot(self.df[available_cols], show_plot=False)
            fig.set_size_inches(12, 10)

            # Embed trong tkinter
            canvas = FigureCanvasTkAgg(fig, self.chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to create pair plot: {str(e)}")

    def show_prediction_plot(self):
        if self.model is None or self.X_test is None:
            messagebox.showerror("Error", "Please train a model first")
            return

        self.clear_chart()
        self.chart_title.config(text="Prediction Plots")

        try:
            # Lấy predictions
            y_pred = self.model.predict(self.X_test)
            y_true = self.y_test

            # SỬ DỤNG FUNCTION VỚI show_plot=False
            fig = draw_graph.draw_prediction_plot(y_true, y_pred, show_plot=False)

            # Embed trong tkinter
            canvas = FigureCanvasTkAgg(fig, self.chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to create prediction plot: {str(e)}")

    def show_correlation(self):
        if self.df is None:
            messagebox.showerror("Error", "Please load a CSV file first")
            return

        self.clear_chart()
        self.chart_title.config(text="Correlation Matrix")

        try:
            # Tạo popup để chọn phương pháp correlation
            self.corr_popup = tk.Toplevel(self.root)
            self.corr_popup.title("Select Correlation Method")
            self.corr_popup.geometry("300x200")

            tk.Label(self.corr_popup, text="Select correlation method:").pack(pady=10)

            # Radio buttons để chọn method
            self.corr_method = tk.StringVar(value="pearson")

            ttk.Radiobutton(self.corr_popup, text="Pearson",
                            variable=self.corr_method, value="pearson").pack(anchor=tk.W, padx=20)
            ttk.Radiobutton(self.corr_popup, text="Spearman",
                            variable=self.corr_method, value="spearman").pack(anchor=tk.W, padx=20)
            # Hàng cấm dùng
            #ttk.Radiobutton(self.corr_popup, text="Kendall",
                            #variable=self.corr_method, value="kendall").pack(anchor=tk.W, padx=20)

            def calculate_correlation():
                method = self.corr_method.get()
                self._create_correlation_plot(method)
                self.corr_popup.destroy()

            ttk.Button(self.corr_popup, text="Calculate Correlation",
                       command=calculate_correlation).pack(pady=10)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to create correlation matrix: {str(e)}")

    def _create_correlation_plot(self, method):
        """Tạo correlation matrix plot sử dụng function từ correlation_matrix"""
        self.clear_chart()
        self.chart_title.config(text=f"Correlation Matrix ({method.capitalize()})")

        try:
            # SỬ DỤNG FUNCTION CORR TỪ correlation_matrix
            corr_matrix = correlation_matrix.corr(self.df, method=method, min_periods=1)

            # Tạo heatmap
            fig, ax = plt.subplots(figsize=(10, 8))

            # Vẽ heatmap
            im = ax.imshow(corr_matrix.values, cmap='coolwarm', vmin=-1, vmax=1)

            # Hiển thị giá trị trên heatmap
            for i in range(len(corr_matrix.columns)):
                for j in range(len(corr_matrix.columns)):
                    text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.3f}',
                                   ha="center", va="center", color="black", fontsize=10)

            # Thiết lập ticks và labels
            ax.set_xticks(range(len(corr_matrix.columns)))
            ax.set_yticks(range(len(corr_matrix.columns)))
            ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
            ax.set_yticklabels(corr_matrix.columns)

            # Thêm colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Correlation Coefficient')

            ax.set_title(f'Correlation Matrix ({method.capitalize()})')
            plt.tight_layout()

            # Embed trong tkinter
            canvas = FigureCanvasTkAgg(fig, self.chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to create correlation matrix: {str(e)}")

    # def show_predictions(self):
    #     if self.model is None or self.X_test is None:
    #         messagebox.showerror("Error", "Please train a model first")
    #         return
    #
    #     self.clear_chart()
    #     self.chart_title.config(text="Prediction Results")
    #
    #     # Make predictions
    #     y_pred = self.model.predict(self.X_test)
    #
    #     # Create comparison plot
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    #
    #     # Plot 1: True vs Predicted
    #     ax1.scatter(self.y_test, y_pred, alpha=0.5)
    #     ax1.plot([self.y_test.min(), self.y_test.max()],
    #              [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
    #     ax1.set_xlabel('True Values')
    #     ax1.set_ylabel('Predicted Values')
    #     ax1.set_title('True vs Predicted Values')
    #
    #     # Plot 2: Prediction line plot
    #     sample_indices = range(min(50, len(self.y_test)))
    #     ax2.plot(sample_indices, self.y_test[:50], 'b-', label='True', alpha=0.7)
    #     ax2.plot(sample_indices, y_pred[:50], 'r-', label='Predicted', alpha=0.7)
    #     ax2.set_xlabel('Sample Index')
    #     ax2.set_ylabel('Value')
    #     ax2.set_title('True vs Predicted (First 50 samples)')
    #     ax2.legend()
    #
    #     plt.tight_layout()
    #
    #     # Embed in tkinter
    #     canvas = FigureCanvasTkAgg(fig, self.chart_frame)
    #     canvas.draw()
    #     canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

#main
root = tk.Tk()
app = OilPriceApp(root)
root.mainloop()
