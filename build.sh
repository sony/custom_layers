rm -rf dist/* && python setup.py bdist_wheel && twine upload -r artifactory dist/*whl && pip install --no-cache --force-reinstall custom_layers
