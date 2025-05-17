from flask import Flask, render_template, jsonify, request
from problems.generators import generate_problem

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_problem", methods=["POST"])
def get_problem():
    data = request.get_json()
    topic = data.get("topic")
    result = generate_problem(topic)

    # Support plot for Lagrange polynomials
    if topic == "lagrange":
        problem, solution, plot_path = result
        return jsonify({
            "problem": problem,
            "solution": solution,
            "plot": plot_path
        })

    problem, solution = result
    return jsonify({"problem": problem, "solution": solution})


if __name__ == "__main__":
    app.run(debug=True)
