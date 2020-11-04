from flask import Flask, render_template
from make_matrix import *
import numpy as np
import os

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/linear-equations.html')
def linear():
    A, b = make_unique_sle(3)
    x = np.linalg.solve(A,b).astype('int64')

    formatted = [[format_coefficient(A, row, col) for col in range(A.shape[1])] for row in range(A.shape[0])]

    xparts = [str(coeff) for coeff in x]
    solution = f"x = ({', '.join(xparts)})"


    return render_template("linear-equations.html", A=A, b=b, formatted=formatted, solution=solution)


def format_coefficient(A, row, col):
    "Returns tuple (op, coefficient, variable)."
    variable = f"x_{col+1}"
    is_first_nonzero_column = all(A[row][c] == 0 for c in range(col))

    if A[row,col] == 0:
        return "", "", ""

    elif abs(A[row,col]) == 1:
        op = "" if is_first_nonzero_column else signop(A[row,col])
        return op, "", variable

    else:
        op = "" if is_first_nonzero_column else signop(A[row,col])
        coeff = A[row,col] if is_first_nonzero_column else abs(A[row,col])
        return op, coeff, variable


def signop(value):
    return "-" if value < 0 else "+"



if os.getenv("PORT"):
    app.run(debug=True, port=int(os.getenv("PORT")))
else:
    app.run(debug=True)