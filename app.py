import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# --- إعداد عنوان الصفحة ---
st.set_page_config(page_title="Smart Inventory Prediction System")
st.title("📊 Inventory Demand Prediction System")
st.write("This application uses AI to predict required product quantities and calculate model accuracy.")

# --- 1. تجهيز بيانات وهمية (بيانات المبيعات) ---
# سنقوم بإنشاء بيانات تعتمد على (سعر المنتج، ميزانية الإعلانات)
np.random.seed(42)
data_size = 100
data = {
    'Price': np.random.uniform(10, 100, data_size),        # سعر المنتج
    'Ad_Budget': np.random.uniform(100, 1000, data_size), # ميزانية الإعلانات
    'Stock_Demand': []                                    # الكمية المطلوبة (الهدف)
}

# معادلة وهمية لإنشاء الهدف: الطلب يزيد بزيادة الإعلانات ويقل بزيادة السعر
for i in range(data_size):
    demand = (data['Ad_Budget'][i] * 0.5) - (data['Price'][i] * 1.2) + np.random.normal(0, 10)
    data['Stock_Demand'].append(max(10, demand)) # التأكد أن الطلب لا يقل عن 10 قطع

df = pd.DataFrame(data)

# --- 2. بناء نموذج تعلم الآلة ---
# تحديد المدخلات (X) والمخرج المطلوب (y)
X = df[['Price', 'Ad_Budget']]
y = df['Stock_Demand']

# تقسيم البيانات: 80% للتدريب و 20% للاختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# استخدام خوارزمية الغابة العشوائية للتنبؤ بالأرقام (Regressor)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train) # تدريب النموذج

# عمل توقعات بناءً على بيانات الاختبار
y_pred = model.predict(X_test)

# حساب نسبة النجاح باستخدام R2 Score
score = r2_score(y_test, y_pred)

# --- 3. عرض النتائج على ويب ---

# عرض نسبة النجاح في "كارت" ملون
st.subheader("✅ Model Performance Evaluation")
st.metric(label="Prediction Accuracy (R2 Score)", value=f"{score * 100:.2f}%")

# --- 4. واجهة تفاعلية للمستخدم ---
st.divider()
st.subheader("🔮 Try the Prediction Yourself")
col1, col2 = st.columns(2)

with col1:
    user_price = st.number_input("Enter Product Price ($):", min_value=10, max_value=100, value=50)
with col2:
    user_ads = st.number_input("Enter Advertising Budget ($):", min_value=100, max_value=1000, value=500)

# التنبؤ بناءً على مدخلات المستخدم
user_prediction = model.predict([[user_price, user_ads]])

st.success(f"The predicted quantity to be ordered is: **{int(user_prediction[0])} units**")

# --- 5. الرسم البياني للنتائج ---
st.divider()
st.subheader("📈 Predictions vs Reality")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, color='blue', alpha=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # خط التوقع المثالي
ax.set_xlabel('Actual Demand')
ax.set_ylabel('Model Prediction')
st.pyplot(fig)

st.info("Note: The closer the blue dots are to the red dashed line, the more accurate the model is!")
