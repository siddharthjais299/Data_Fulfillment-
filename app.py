import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import numpy as np
import base64
from io import BytesIO
from streamlit_option_menu import option_menu
import requests  # Used for geocoding

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Dynamic Fulfillment Optimizer")
st.markdown("""
<style>
/* --- Sidebar Container Styling --- */
[data-testid="stSidebar"] {
    background-color: #25023B;
    background-image: linear-gradient(180deg, #25023B 0%, #62059C 100%);
    color: white;
}

/* --- Main Page Background --- */
[data-testid="stAppViewContainer"] {
    background-color: #0F021A;
    background-image: linear-gradient(180deg, #0F021A 0%, #1E0033 100%);
    color: white;
}

/* --- Top Navigation Bar / Header --- */
[data-testid="stHeader"] { background: transparent; }

/* --- Main Content Area --- */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    background: rgba(255, 255, 255, 0.02);
    border-radius: 10px;
}

/* --- Headings and Text --- */
h1,h2,h3,h4,h5,h6 { color: #FFFFFF; font-weight: 600; }
p, div, span, label { color: #E5E5E5; }

/* --- Buttons (General) --- */
.stButton > button {
    background-color: #6A0DAD;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
    transition: all 0.3s ease;
}
.stButton > button:hover { background-color: #9B30FF; transform: scale(1.03); }

/* --- Download Button Styling --- */
.stDownloadButton a {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 0.25rem 0.75rem;
    border-radius: 0.5rem;
    min-height: 38px;
    line-height: 1.6;
    color: white;
    background-color: #4CAF50;
    border: 1px solid #4CAF50;
    text-decoration: none;
    transition: background-color 0.3s ease;
}
.stDownloadButton a:hover { background-color: #45a049; border-color: #45a049; }

/* --- Tables --- */
[data-testid="stDataFrame"] {
    background-color: rgba(255, 255, 255, 0.05);
    border-radius: 10px;
}

/* --- Hide Streamlit Default Menu, Footer --- */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* --- Custom Fullscreen Loader (CSS-only hide) --- */
#custom-loader {
    position: fixed;
    inset: 0;
    width: 100vw;
    height: 100vh;
    background: radial-gradient(circle at center, rgba(15,2,26,0.95) 0%, rgba(10,0,16,1) 100%);
    z-index: 999999;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    opacity: 1;
    pointer-events: auto;
    /* Start the fade-out animation which ends at 10s and stays hidden (forwards) */
    animation: hideLoader 5s linear forwards;
}

/* Spinner */
.spinner {
    border: 8px solid rgba(255, 255, 255, 0.2);
    border-top: 8px solid #9B30FF;
    border-radius: 50%;
    width: 80px;
    height: 80px;
    animation: spin 1.2s linear infinite;
    box-shadow: 0 0 25px #9B30FF;
}

/* Loading text */
.loading-text {
    color: #FFFFFF;
    font-size: 20px;
    font-weight: 600;
    margin-top: 18px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    animation: glow 1.5s ease-in-out infinite alternate;
}

/* Animations */
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
@keyframes glow {
    from { text-shadow: 0 0 8px #9B30FF; }
    to { text-shadow: 0 0 20px #D58FFF; }
}

/* hideLoader: stays visible until 99% then sets opacity 0 and visibility hidden at 100%.
   animation-fill-mode: forwards applied via 'forwards' in animation shorthand above. */
@keyframes hideLoader {
    0%   { opacity: 1; visibility: visible; }
    99%  { opacity: 1; visibility: visible; }
    100% { opacity: 0; visibility: hidden; pointer-events: none; }
}

/* Extra safety: once hidden ensure no clicks */
#custom-loader[style*="visibility: hidden"] { pointer-events: none; }
</style>

<div id="custom-loader">
    <div class="spinner"></div>
    <div class="loading-text">Loading...</div>
</div>
""", unsafe_allow_html=True)

# Define the penalty factor used in the ML training script
PENALTY_FACTOR = 0.5


# --- Helper Functions ---

# Function to get city coordinates (Approximation for demo purposes)
@st.cache_data
def get_city_coordinates(cities):
    """
    Simulates getting latitude and longitude for a list of cities.
    UPDATED to include Delhi, Mumbai, Bangalore, Pune, Dubai, Bangkok, and Singapore.
    """



    coords = {
        'Mumbai': (19.0760, 72.8777),
        'Delhi': (28.7041, 77.1025),
        'Bangalore': (12.9716, 77.5946),
        'Pune': (18.5204, 73.8567),
        'Dubai': (25.276987, 55.296249),
        'Bangkok': (13.7563, 100.5018),
        'Singapore': (1.3521, 103.8198)
    }

    city_data = []
    for city in cities:
        # Use only the specified cities. If a city is in the original data but not in this list,
        # it will default to a neutral point (center of India).
        if city in coords:
            lat, lon = coords[city]
            city_data.append({'City': city, 'lat': lat, 'lon': lon})
        else:
            # Handle cities in the original CSV that are not in the new list (e.g., Hyderabad, Chennai)
            # These cities will still be included in the dataframe but assigned a default coordinate.
            lat, lon = (20.5937, 78.9629)  # Default to center of India if city is missing
            city_data.append({'City': city, 'lat': lat, 'lon': lon})

    return pd.DataFrame(city_data)


# Function to load and merge data
@st.cache_data
def load_and_merge_data():
    """Loads all CSVs, merges them, and applies feature engineering (Reliability Score)."""
    try:
        # Load core data
        orders = pd.read_csv("orders.csv")
        routes = pd.read_csv("routes_distance.csv")
        delivery = pd.read_csv("delivery_performance.csv")
        costs = pd.read_csv("cost_breakdown.csv")
        inventory = pd.read_csv("warehouse_inventory.csv")
        feedback = pd.read_csv("customer_feedback.csv")
        fleet = pd.read_csv("vehicle_fleet.csv")

        # 1. Cost Aggregation
        cost_cols = ['Fuel_Cost', 'Labor_Cost', 'Vehicle_Maintenance', 'Insurance', 'Packaging_Cost',
                     'Technology_Platform_Fee', 'Other_Overhead']
        costs['Total_Logistics_Cost'] = costs[cost_cols].sum(axis=1)

        # 2. Merge DataFrames
        df = orders.merge(routes, on='Order_ID', how='inner')  # Inner join to match training script merge
        df = df.merge(delivery, on='Order_ID', how='inner')
        df = df.merge(costs[['Order_ID', 'Total_Logistics_Cost']], on='Order_ID', how='left')

        # Fill any missing Total_Logistics_Cost with Delivery_Cost_INR (as done in ML script)
        df['Total_Logistics_Cost'].fillna(df['Delivery_Cost_INR'], inplace=True)

        # 3. Feature Engineering (Reliability Score)
        df['Delay_Days'] = df['Actual_Delivery_Days'] - df['Promised_Delivery_Days']
        df['Delay_Days'] = df['Delay_Days'].fillna(0)

        carrier_perf = df.groupby('Carrier').agg(
            Avg_Rating=('Customer_Rating', 'mean'),
            Avg_Delay_Days=('Delay_Days', 'mean')
        ).reset_index()
        carrier_perf['Reliability_Score'] = carrier_perf['Avg_Rating'] - (
                carrier_perf['Avg_Delay_Days'] * PENALTY_FACTOR)

        # Merge the Reliability Score back into the main dataset
        df = df.merge(carrier_perf[['Carrier', 'Reliability_Score']], on='Carrier', how='left')

        # 4. Geolocation data for mapping (NEW ADDITION)
        all_cities = pd.concat([df['Origin'], df['Destination']]).unique()
        df_coords = get_city_coordinates(all_cities)

        # Merge coords for Origin
        df = df.merge(df_coords.rename(columns={'City': 'Origin', 'lat': 'Origin_lat', 'lon': 'Origin_lon'}),
                      on='Origin', how='left')
        # Merge coords for Destination
        df = df.merge(
            df_coords.rename(columns={'City': 'Destination', 'lat': 'Destination_lat', 'lon': 'Destination_lon'}),
            on='Destination', how='left')

        # 5. Final Cleaning
        critical_cols = [
            'Delivery_Status', 'Total_Logistics_Cost', 'Distance_KM',
            'Order_Value_INR', 'Priority', 'Customer_Segment', 'Product_Category',
            'Carrier', 'Reliability_Score', 'Origin', 'Destination', 'Special_Handling', 'Weather_Impact'
        ]
        # Include new coordinate columns in dropna subset for safety, though they should be filled by get_city_coordinates
        df.dropna(subset=critical_cols + ['Origin_lat', 'Origin_lon', 'Destination_lat', 'Destination_lon'],
                  inplace=True)

        return df, inventory, feedback, carrier_perf

    except FileNotFoundError as e:
        st.error(f"Required file not found: {e.filename}. Please ensure all uploaded files are accessible.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred during data loading or merging: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


# Function to load ML model and encoder
@st.cache_resource
def load_ml_assets():
    """Loads the pre-trained RF model and Label Encoder."""
    try:
        with open('fulfillment_rf_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('fulfillment_label_encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        return model, encoder
    except FileNotFoundError:
        st.error("ML model or encoder file not found. Prediction functionality disabled.")
        return None, None
    except Exception as e:
        st.error(f"Error loading ML assets: {e}")
        return None, None


# Function to generate a download link for data
def get_table_download_link(df, filename, label):
    """Generates a link allowing the data in a DataFrame to be downloaded as a CSV."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="stDownloadButton">{label}</a>'
    return href


# Function to generate a download link for a plotly figure
def get_chart_download_link(fig, filename, label):
    """Generates a link to download the Plotly chart as an HTML file."""
    buffer = BytesIO()
    html_bytes = fig.to_html(full_html=False).encode('utf-8')
    buffer.write(html_bytes)

    b64 = base64.b64encode(buffer.getvalue()).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="{filename}.html" class="stDownloadButton">{label}</a>'
    return href


# --- Data Loading ---
df_main, df_inventory, df_feedback, df_carrier_perf = load_and_merge_data()
rf_model, le_encoder = load_ml_assets()

# --- Streamlit App Layout ---
st.title("📦 Dynamic Fulfillment Optimizer")
st.markdown("### Analyzing and Predicting Optimal Order Fulfillment")

if df_main.empty:
    st.warning("Cannot proceed with analysis. Please ensure all necessary CSV files are uploaded and correct.")
    st.stop()

# --- Sidebar Navigation (Modern Style using Option Menu) ---
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["Data Overview & Analysis", "Fulfillment Prediction Simulator", "Raw Data Viewer"],
        icons=["bar-chart", "robot", "table"],  # Bootstrap icons
        menu_icon="compass",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#180126"},
            "icon": {"color": "#4CAF50", "font-size": "20px"},
            "nav-link": {
                "font-size": "15px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#262730"
            },
            "nav-link-selected": {"background-color": "#39035c", "color": "white"},
        },
    )

page = selected

# --- Page 1: Data Overview & Analysis ---
if page == "Data Overview & Analysis":
    st.header("1. Core Logistics Data Analysis")

    # --- Key Metrics ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Orders Analyzed", f"{len(df_main):,}")
    col2.metric("Avg. Order Value (INR)", f"₹{df_main['Order_Value_INR'].mean():,.2f}")
    col3.metric("Avg. Distance (KM)", f"{df_main['Distance_KM'].mean():,.0f} km")
    col4.metric("On-Time Delivery Rate", f"{(df_main['Delivery_Status'] == 'On-Time').mean() * 100:.2f}%")
    st.markdown("---")

    st.header("2. Key Visualizations")

    # --- Visualization 1 ---
    st.subheader("Delivery Performance Breakdown")
    status_counts = df_main['Delivery_Status'].value_counts().reset_index()
    status_counts.columns = ['Delivery Status', 'Count']
    fig1 = px.bar(status_counts, x='Delivery Status', y='Count',
                  title='Overall Delivery Status Distribution',
                  color='Delivery Status',
                  color_discrete_map={'On-Time': 'green', 'Slightly-Delayed': 'orange', 'Severely-Delayed': 'red'})
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown(get_chart_download_link(fig1, "delivery_status_breakdown", "Download Delivery Status Chart (HTML)"),
                unsafe_allow_html=True)

    # --- Visualization 2 ---
    st.subheader("Logistics Cost vs. Distance by Delivery Status")
    fig2 = px.scatter(df_main, x='Distance_KM', y='Total_Logistics_Cost',
                      color='Delivery_Status',
                      hover_data=['Order_ID', 'Route', 'Order_Value_INR'],
                      title='Logistics Cost vs. Distance (Colored by Delivery Status)',
                      template='plotly_white')
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown(get_chart_download_link(fig2, "cost_vs_distance_scatter", "Download Cost vs. Distance Chart (HTML)"),
                unsafe_allow_html=True)

    # --- Visualization 3 ---
    st.subheader("Carrier Reliability Scores vs. Customer Ratings")
    fig3 = px.box(df_main, x='Carrier', y='Customer_Rating',
                  color='Carrier',
                  title='Customer Rating Distribution by Carrier',
                  points='all')
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown(get_chart_download_link(fig3, "customer_rating_by_carrier", "Download Rating by Carrier Chart (HTML)"),
                unsafe_allow_html=True)

    # --- Visualization 4 ---
    if not df_inventory.empty:
        st.subheader("Warehouse Inventory Health: Stock vs. Reorder Level")
        inv_agg = df_inventory.groupby('Product_Category')[['Current_Stock_Units', 'Reorder_Level']].sum().reset_index()
        inv_agg['Stock_Health'] = inv_agg['Current_Stock_Units'] - inv_agg['Reorder_Level']
        fig4 = px.bar(inv_agg, x='Product_Category', y=['Current_Stock_Units', 'Reorder_Level'],
                      barmode='group',
                      title='Total Inventory Stock vs. Reorder Level by Product Category',
                      labels={'value': 'Units', 'variable': 'Metric'})
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown(get_chart_download_link(fig4, "inventory_health_chart", "Download Inventory Health Chart (HTML)"),
                    unsafe_allow_html=True)

    # --- Visualization 5 ---
    st.subheader("Average Logistics Cost by Customer Segment and Priority")
    cost_agg = df_main.groupby(['Customer_Segment', 'Priority'])['Total_Logistics_Cost'].mean().reset_index()
    cost_agg.columns = ['Customer_Segment', 'Priority', 'Average_Cost']
    fig5 = px.bar(cost_agg, x='Customer_Segment', y='Average_Cost',
                  color='Priority',
                  barmode='group',
                  title='Average Total Logistics Cost by Customer Segment and Priority',
                  labels={'Average_Cost': 'Average Logistics Cost (INR)'},
                  template='plotly_white')
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown(
        get_chart_download_link(fig5, "avg_cost_by_segment_priority", "Download Cost by Segment/Priority Chart (HTML)"),
        unsafe_allow_html=True)


# --- Page 2: Fulfillment Prediction Simulator ---
elif page == "Fulfillment Prediction Simulator":
    st.header("Dynamic Fulfillment Prediction")
    st.markdown("Input the order parameters to simulate the likely Delivery Status (Target: On-Time).")

    if rf_model is None or le_encoder is None:
        st.warning("Cannot run prediction: ML model assets failed to load.")
    else:
        feature_names = rf_model.feature_names_in_

        # Use the union of origins and destinations that have defined coordinates for selection
        available_cities = ['Delhi', 'Mumbai', 'Bangalore', 'Pune', 'Dubai', 'Bangkok', 'Singapore']
        # Filter the lists of cities to only include those defined in coords
        origins = sorted([c for c in df_main['Origin'].unique() if c in available_cities])
        destinations = sorted([c for c in df_main['Destination'].unique() if c in available_cities])

        priorities = sorted(df_main['Priority'].unique())
        segments = sorted(df_main['Customer_Segment'].unique())
        products = sorted(df_main['Product_Category'].unique())
        handling = sorted(df_main['Special_Handling'].unique())
        weather = sorted(df_main['Weather_Impact'].unique())
        carriers = sorted(df_carrier_perf['Carrier'].unique())

        # Determine initial selection for map
        default_origin = 'Mumbai' if 'Mumbai' in origins else origins[0] if origins else ''
        default_destination = 'Delhi' if 'Delhi' in destinations else destinations[0] if destinations else ''

        # Stop if no valid cities are found, though this shouldn't happen with the current setup
        if not origins or not destinations:
            st.error("No valid city data found for route selection. Please check data files.")
            st.stop()

        with st.form("prediction_form"):
            col_a, col_b = st.columns(2)

            with col_a:
                st.subheader("Order Details")
                order_value = st.number_input("Order Value (INR)", min_value=1.0, value=100.0, format="%.2f")
                distance = st.number_input("Estimated Distance (KM)", min_value=1.0, value=1.0, format="%.1f")
                total_cost = st.number_input("Estimated Total Logistics Cost (INR)", min_value=1.0, value=800.0,
                                             format="%.2f")

                st.subheader("Route Details")
                origin = st.selectbox("Origin City", origins,
                                      index=origins.index(default_origin) if default_origin in origins else 0,
                                      key='origin_select')
                destination = st.selectbox("Destination City", destinations,
                                           index=destinations.index(
                                               default_destination) if default_destination in destinations else 0,
                                           key='dest_select')
                weather_impact = st.selectbox("Expected Weather Impact", weather,
                                              index=weather.index('None') if 'None' in weather else 0)

            with col_b:
                st.subheader("Customer & Product Data")
                priority = st.selectbox("Order Priority", priorities)
                segment = st.selectbox("Customer Segment", segments)
                product_category = st.selectbox("Product Category", products)
                special_handling = st.selectbox("Special Handling", handling,
                                                index=handling.index('None') if 'None' in handling else 0)

                st.subheader("Fulfillment Choice (Carrier)")
                selected_carrier = st.selectbox("Hypothetical Carrier Choice", carriers)
                reliability_score_lookup = \
                    df_carrier_perf[df_carrier_perf['Carrier'] == selected_carrier]['Reliability_Score'].iloc[0]
                st.info(
                    f"Using Carrier's historical Reliability Score: **{reliability_score_lookup:.4f}** (Based on ML training logic)")

            submitted = st.form_submit_button("Predict Delivery Status")

        st.markdown("---")
        st.subheader("Selected Route Visualization")

        # --- NEW MAP VISUALIZATION LOGIC (Updated to ensure coordinates exist for selected cities) ---

        # Get coordinates for the selected cities
        origin_coords = df_main[df_main['Origin'] == origin][['Origin_lat', 'Origin_lon']].iloc[0]
        dest_coords = df_main[df_main['Destination'] == destination][['Destination_lat', 'Destination_lon']].iloc[0]

        # Define the path for Plotly (Origin -> Destination)
        path_df = pd.DataFrame({
            'lat': [origin_coords['Origin_lat'], dest_coords['Destination_lat']],
            'lon': [origin_coords['Origin_lon'], dest_coords['Destination_lon']],
            'City': [origin, destination]
        })

        # Create a Plotly Scatter Mapbox
        fig_map = px.line_mapbox(path_df, lat="lat", lon="lon", color_discrete_sequence=["#FF5733"], zoom=4, height=400)

        # Add the points (Origin and Destination) as markers
        fig_map.add_trace(px.scatter_mapbox(path_df, lat="lat", lon="lon",
                                            hover_name="City",
                                            color=path_df['City'].apply(
                                                lambda x: 'Origin' if x == origin else 'Destination'),
                                            color_discrete_map={'Origin': '#4CAF50', 'Destination': '#6A0DAD'},
                                            size=[15, 15],
                                            size_max=15).data[0])

        # Calculate map center for better view
        map_center_lat = path_df['lat'].mean()
        map_center_lon = path_df['lon'].mean()

        # Adjust zoom level based on distance for a better fit
        # Simple heuristic: higher zoom for closer points, lower zoom for farther points
        max_coord_diff = max(abs(path_df['lat'].max() - path_df['lat'].min()),
                             abs(path_df['lon'].max() - path_df['lon'].min()))

        # Simple logic:
        # Very close (e.g., < 0.5 deg diff): zoom 9
        # Regional (e.g., < 5 deg diff): zoom 6
        # Country/Continent (e.g., < 20 deg diff): zoom 4
        # Intercontinental: zoom 2
        if max_coord_diff < 0.5:
            map_zoom = 9
        elif max_coord_diff < 5:
            map_zoom = 6
        elif max_coord_diff < 20:
            map_zoom = 4
        else:
            map_zoom = 2

        # Configure the map layout
        fig_map.update_layout(
            mapbox_style="carto-darkmatter",  # Dark map style
            mapbox_zoom=map_zoom,
            mapbox_center={"lat": map_center_lat, "lon": map_center_lon},
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                title='Location'
            )
        )

        st.plotly_chart(fig_map, use_container_width=True)
        # --- END OF NEW MAP VISUALIZATION LOGIC ---

        if submitted:
            input_features = pd.DataFrame(0, index=[0], columns=feature_names)
            input_features['Order_Value_INR'] = order_value
            input_features['Distance_KM'] = distance
            input_features['Total_Logistics_Cost'] = total_cost
            input_features['Reliability_Score'] = reliability_score_lookup

            categorical_inputs = {
                'Priority': priority,
                'Customer_Segment': segment,
                'Product_Category': product_category,
                'Origin': origin,
                'Destination': destination,
                'Special_Handling': special_handling,
                'Weather_Impact': weather_impact
            }

            for feature, value in categorical_inputs.items():
                col_name = f'{feature}_{value}'
                if col_name in input_features.columns:
                    input_features[col_name] = 1

            try:
                prediction_encoded = rf_model.predict(input_features)
                prediction_proba = rf_model.predict_proba(input_features)[0]
                predicted_status = le_encoder.inverse_transform(prediction_encoded)[0]

                st.markdown("---")
                st.subheader("Prediction Result")

                if predicted_status == 'On-Time':
                    st.success(f"**Predicted Delivery Status: {predicted_status}** 🎉")
                    st.balloons()
                elif predicted_status == 'Slightly-Delayed':
                    st.warning(f"**Predicted Delivery Status: {predicted_status}** ⚠️")
                else:
                    st.error(f"**Predicted Delivery Status: {predicted_status}** 🚨")

                st.markdown("---")
                st.subheader("Status Probability Breakdown")
                proba_df = pd.DataFrame({
                    'Status': le_encoder.classes_,
                    'Probability': prediction_proba
                }).sort_values(by='Probability', ascending=False)
                proba_df['Probability'] = (proba_df['Probability'] * 100).round(2).astype(str) + '%'
                st.dataframe(proba_df, use_container_width=True, hide_index=True)

            except ValueError as e:
                st.error(
                    f"Prediction Error: The input features do not match the trained model's features. Details: {e}")

# --- Page 3: Raw Data Viewer ---
elif page == "Raw Data Viewer":
    st.header("Raw Data Explorer")
    st.markdown("Select a dataset to view the raw data and download it as CSV.")

    data_options = {
        "Orders & Fulfillment (Merged)": df_main,
        "Warehouse Inventory": df_inventory,
        "Customer Feedback": df_feedback,
        "Carrier Performance (ML Feature)": df_carrier_perf
    }

    selected_data_name = st.selectbox("Choose Dataset to View", list(data_options.keys()))
    selected_df = data_options[selected_data_name]

    st.subheader(f"Dataset: {selected_data_name}")
    st.dataframe(selected_df, use_container_width=True)

    if not selected_df.empty:
        st.markdown(get_table_download_link(selected_df, f"{selected_data_name.replace(' ', '_').lower()}.csv",
                                            f"Download {selected_data_name} CSV"), unsafe_allow_html=True)
    else:
        st.info("No data available to download.")
