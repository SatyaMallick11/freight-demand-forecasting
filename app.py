from flask import Flask, jsonify, render_template, request
from ml_model import train_models, forecast_horizon
from optimizer import optimize_capacity

app = Flask(__name__)
df, beta, rf = train_models()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/forecast", methods=["POST"])
def api_forecast():
    data = request.get_json()
    horizon = int(data.get("horizon", 25))
    fuel = float(data.get("fuel", 104.5))
    congestion = float(data.get("congestion", 0.24))

    forecasts = forecast_horizon(df, beta, rf, horizon, fuel, congestion)
    result = optimize_capacity(forecasts)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)