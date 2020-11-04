from flask import Flask, render_template
from make_matrix import *
import numpy as np
import argparse

from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop


parser = argparse.ArgumentParser()
parser.add_argument("--port", "-p", type=int, default=5000, help="run on this port")
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()



app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/linear-equations.html')
def linear():
    mode = np.random.choice(3, 1, p=[0.5, 0.25, 0.25])[0]  # unique / unsolvable / infinite

    if mode == 0:
        # uniquely solvable SLE
        A, b = make_unique_sle(3)
        x = np.linalg.solve(A,b).astype('int64')
        xparts = [str(coeff) for coeff in x]
        solution = f"x = ({', '.join(xparts)})"

    elif mode == 1:
        # unsolvable SLE
        A, b = make_unsolvable_sle(3, max_val=4)
        solution = "unsolvable"

    else:
        # SLE with infinitely many solutions
        A, b, x = make_underconstrained_sle(3, max_val=4, max_solution_val=3)
        xparts = [str(coeff) for coeff in x]
        solution = f"infinitely many solutions; one solution is x = ({', '.join(xparts)})"

    formatted = [[format_coefficient(A, row, col) for col in range(A.shape[1])] for row in range(A.shape[0])]
    return render_template("linear-equations.html", A=A, b=b, formatted=formatted, solution=solution)


def format_coefficient(A, row, col):
    "Formats the given entry of the matrix into a string tuple (op, coefficient, variable) that is suitable for printing."
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


if args.debug:
    app.run(debug=True, host='0.0.0.0', port=args.port)
else:
    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(args.port)
    IOLoop.instance().start()

