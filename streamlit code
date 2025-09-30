import streamlit as st
import pandas as pd
import joblib
import gzip
import os
from snowflake.snowpark.context import get_active_session

# ======================================================
# Reuse Snowflake session
# ======================================================
try:
    session = get_active_session()
except Exception:
    session = None  # fallback if running locally

# ======================================================
# Load model from stage
# ======================================================
local_model_path = "supply_chain_model_2.pkl.gz"

if session:
    stage_file = '@"LOGISTICS_DB"."LOGISTICS_SCHEMA"."LOGISTICS_STAGE"/supply_chain_model_2.pkl.gz'
    session.file.get(stage_file, ".")
    local_model_path = os.path.join(".", "supply_chain_model_2.pkl.gz")

with gzip.open(local_model_path, "rb") as f:
    model = joblib.load(f)

st.title("📦 On-Time Delivery Prediction")


with st.sidebar:
    # Logo
    st.image(
        "https://booleandata.ai/wp-content/uploads/2022/09/Boolean-logo_Boolean-logo-USA-1.png",
        use_container_width=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # # Model accuracy
    # st.subheader("📊 Model Accuracy")
    # st.metric("Hold-out Accuracy", f"{holdout_acc:.2%}")

    # Spacer
    st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)

    # About Us
    st.markdown("""
<div style="font-size:15px; line-height:1.4; color:#333;text-align: center;">
<h5 style="font-size:18px;">🚀 About Us</h5>
We leverage Snowflake to plan and design emerging data architectures that facilitate incorporation of high-quality and flexible data. 
<br><br>
These solutions lower costs and enhance output, designed to transform smoothly as your enterprise, and your data continue to increase over time.
</div>
    """, unsafe_allow_html=True)

    # Social media links
    st.markdown("""
<div style="text-align:center; display:flex; justify-content:center; gap:15px; margin-top:10px;">
<a href="https://booleandata.ai/" target="_blank">🌐</a>
<a href="https://www.facebook.com/Booleandata" target="_blank">
<img src="https://cdn-icons-png.flaticon.com/24/1384/1384005.png" width="24">
</a>
<a href="https://www.youtube.com/channel/UCd4PC27NqQL5v9-1jvwKE2w" target="_blank">
<img src="https://cdn-icons-png.flaticon.com/24/1384/1384060.png" width="24">
</a>
<a href="https://www.linkedin.com/company/boolean-data-systems" target="_blank">
<img src="https://cdn-icons-png.flaticon.com/24/145/145807.png" width="24">
</a>
</div>
    """, unsafe_allow_html=True)
    

# ======================================================
# User Input Section
# ======================================================

# 🔹 Let user directly choose a weekday (0=Monday … 6=Sunday)
weekday_map = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2,
    "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
}
selected_day = st.selectbox("📅 Shipping day", list(weekday_map.keys()))
shipping_weekday = weekday_map[selected_day]

order_item_quantity = st.number_input("🛒 Order Item Quantity", min_value=1, value=5)
order_item_total_amount = st.number_input("💰 Order Item Total Amount", min_value=1.0, value=200.0)
order_item_discount_rate = st.number_input("🏷️ Discount Rate (%)", min_value=0.0, max_value=1.0, value=0.1)
sales = st.number_input("📈 Sales", min_value=1.0, value=250.0)
product_price = st.number_input("💵 Product Price", min_value=1.0, value=50.0)
order_profit_per_order = st.number_input("📊 Order Profit per Order", value=30.0)

shipping_mode = st.selectbox("🚢 Shipping Mode", ["Standard Class", "First Class", "Same Day", "Second Class"])
order_region = st.selectbox("🌍 Order Region", ["East", "West", "Central", "South"])

payment_type = st.selectbox("💳 Payment Type", ["Credit Card", "Debit Card", "COD", "Net Banking", "UPI"])
customer_city = st.text_input("🏙️ Customer City", value="New York")

# ======================================================
# Prediction
# ======================================================
if st.button("🔮 Predict Delivery Status"):
    input_df = pd.DataFrame([{
        "ORDER_ITEM_QUANTITY": order_item_quantity,
        "ORDER_ITEM_TOTAL_AMOUNT": order_item_total_amount,
        "ORDER_ITEM_DISCOUNT_RATE": order_item_discount_rate,
        "SALES": sales,
        "PRODUCT_PRICE": product_price,
        "ORDER_PROFIT_PER_ORDER": order_profit_per_order,
        "SHIPPING_MODE": shipping_mode,
        "ORDER_REGION": order_region,
        "PAYMENT_TYPE": payment_type,
        "CUSTOMER_CITY": customer_city,
        "SHIPPING_WEEKDAY": shipping_weekday
    }])

    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    st.subheader("📊 Prediction Result")
    st.write(f"**Prediction:** {'✅ On-Time Delivery' if prediction == 1 else '⚠️ Delay Expected'}")
    st.write(f"**Probability:** On-Time = {proba[1]*100:.2f}% | Delay = {proba[0]*100:.2f}%")
