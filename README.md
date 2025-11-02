# 📦 Dynamic Fulfillment Optimizer

deployded URL link -https://data-fulfillment-6.onrender.com

The Dynamic Fulfillment Optimizer is an interactive web application built with Streamlit that analyzes logistics data and predicts order delivery outcomes. It provides a comprehensive dashboard for visualizing key performance indicators, exploring raw data, and simulating delivery statuses using a pre-trained Random Forest model.

This tool is designed for logistics managers, data analysts, and supply chain professionals to gain insights into their fulfillment network, identify bottlenecks, assess carrier performance, and make data-driven decisions to improve on-time delivery rates.

## ✨ Key Features

*   **Interactive Analytics Dashboard**: Visualize core logistics metrics, including delivery performance, cost breakdowns, carrier reliability, and warehouse inventory levels.
*   **Fulfillment Prediction Simulator**: Input various order parameters (e.g., origin, destination, value, carrier) to predict the likelihood of an order being 'On-Time', 'Slightly-Delayed', or 'Severely-Delayed'.
*   **Route Visualization**: An integrated map displays the selected origin and destination for a more intuitive simulation experience.
*   **ML-Powered Insights**: Utilizes a Random Forest Classifier trained on historical data to provide accurate predictions. A key engineered feature, `Reliability_Score`, combines customer ratings and delivery delays to a single metric for carrier evaluation.
*   **Data Explorer**: View and download the underlying raw and merged datasets directly from the application.

## 📊 Dataset

The application is powered by a set of interconnected CSV files that model a complete logistics ecosystem:

*   `orders.csv`: Contains primary order details such as customer segment, priority, value, and product category.
*   `delivery_performance.csv`: Tracks delivery outcomes, including carrier, promised vs. actual delivery days, and customer ratings.
*   `cost_breakdown.csv`: Provides a detailed breakdown of logistics costs for each order (fuel, labor, maintenance, etc.).
*   `routes_distance.csv`: Includes route-specific information like distance, traffic delays, and weather impact.
*   `customer_feedback.csv`: Collects qualitative feedback and ratings from customers.
*   `warehouse_inventory.csv`: Manages stock levels, reorder points, and storage costs across different warehouses.
*   `vehicle_fleet.csv`: Details the fleet of vehicles, including type, capacity, and current status.

## 🤖 Machine Learning Model

The predictive functionality is driven by a Random Forest Classifier model saved in `fulfillment_rf_model.pkl`.

*   **Objective**: To classify an order's final delivery status into one of three categories: `On-Time`, `Slightly-Delayed`, or `Severely-Delayed`.
*   **Features**: The model is trained on a combination of numerical and categorical features, including:
    *   `Order_Value_INR`
    *   `Distance_KM`
    *   `Total_Logistics_Cost`
    *   `Reliability_Score` (Engineered Feature)
    *   Categorical features like `Priority`, `Customer_Segment`, `Origin`, `Destination`, and `Weather_Impact`.
*   **Encoding**: Categorical features are one-hot encoded for the model, and the target variable (`Delivery_Status`) is managed using the `fulfillment_label_encoder.pkl`.

## ⚙️ Technology Stack

*   **Backend**: Python
*   **Web Framework**: Streamlit
*   **Data Manipulation**: Pandas, NumPy
*   **Visualization**: Plotly Express
*   **Machine Learning**: Scikit-learn
*   **UI Components**: streamlit-option-menu

## 🚀 How to Run Locally

To set up and run this project on your local machine, follow these steps.

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/siddharthjais299/data_fulfillment-.git
    cd data_fulfillment-
    ```

2.  **Create a Virtual Environment**
    It's recommended to create a virtual environment to manage dependencies.
    ```bash
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    The required packages are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
    If `requirements.txt` is missing, you can install the necessary packages manually:
    ```bash
    pip install streamlit pandas plotly scikit-learn numpy streamlit-option-menu requests
    ```

4.  **Run the Streamlit App**
    Execute the following command in your terminal:
    ```bash
    streamlit run app.py
    ```
    The application will open in your default web browser.

## 📂 File Structure

```
└── data_fulfillment-/
    ├── app.py                      # Main Streamlit application file
    ├── cost_breakdown.csv          # Dataset: Cost components for orders
    ├── customer_feedback.csv       # Dataset: Customer ratings and feedback
    ├── delivery_performance.csv    # Dataset: Delivery carrier performance
    ├── fulfillment_label_encoder.pkl # Saved LabelEncoder for the target variable
    ├── fulfillment_rf_model.pkl    # Pre-trained Random Forest model
    ├── orders.csv                  # Dataset: Core order information
    ├── requirements.txt            # Python dependencies
    ├── routes_distance.csv         # Dataset: Route and distance information
    ├── vehicle_fleet.csv           # Dataset: Vehicle fleet details
    └── warehouse_inventory.csv     # Dataset: Warehouse stock levels
