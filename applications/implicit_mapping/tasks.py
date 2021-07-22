import invoke


@invoke.task
def black(c):
    c.run('poetry run black --check imap tests')


@invoke.task
def flake8(c):
    c.run('poetry run flake8 imap tests')


@invoke.task
def isort(c):
    c.run('poetry run isort --check-only imap tests')


@invoke.task
def mypy(c):
    c.run('poetry run mypy imap')


@invoke.task
def test(c):
    c.run(
        'poetry run pytest tests --cov=craftrip --cov-report term-missing '
        '--durations 5'
    )