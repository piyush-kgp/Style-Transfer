```
pip install twine
```

```
python setup.py sdist bdist_wheel
python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

```
python -m twine upload dist/*
```

```
python3 -m pip install --index-url https://test.pypi.org/simple/ example-pkg-your-username
```
