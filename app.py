import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("AI 股價趨勢預估系統")

ticker = st.text_input("股票代號", "DIS")

if st.button("開始分析"):
    data = yf.download(ticker, start="2022-01-01")

    data["MA5"] = data["Close"].rolling(5).mean()
    data["MA10"] = data["Close"].rolling(10).mean()
    data["Return"] = data["Close"].pct_change()
    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
    data = data.dropna()

    X = data[["MA5", "MA10", "Return"]]
    y = data["Target"]

    # 切分資料
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    future_pred = model.predict(X.tail(1))[0]

    st.subheader("股價走勢")
    st.line_chart(data["Close"])

    st.subheader("模型準確率")
    st.write(f"{acc:.2%}")

    st.subheader("AI 預測結果")
    if future_pred == 1:
        st.success("短期趨勢偏向上漲")
    else:
        st.warning("短期趨勢偏向下跌")

    st.caption("僅供學術研究，非投資建議")
