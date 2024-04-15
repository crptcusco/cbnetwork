import parsl
from parsl import python_app, bash_app


@python_app
def hello_python(message):
    return 'Hello %s' % message


@bash_app
def hello_bash(message, stdout='hello-stdout'):
    return 'echo "Hello %s"' % message


with parsl.load():
    # invoke the Python app and print the result
    print(hello_python('World (Python)').result())
    # invoke the Bash app and read the result from a file
    hello_bash('World (Bash)').result()
with open('hello-stdout', 'r') as f:
    print(f.read())
